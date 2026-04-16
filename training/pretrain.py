import torch
import torch.nn as nn
from torchvision.ops import roi_align
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolo_anchor import YOLOAnchor
from models.delta_encoder import DeltaDetEncoder
from torch.optim import AdamW

class PretrainAligner(nn.Module):
    """
    Stage 1: Δ-Det Encoder Pre-training (Alignment).
    Trains the Delta Encoder to minimize MSE between its per-object Δ-tokens
    and the YOLO backbone's per-box features (via RoI-Align).
    """
    def __init__(self, yolo_size='yolov8m.pt', embed_dim=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.yolo_anchor = YOLOAnchor(model_size=yolo_size, device=device)
        self.delta_encoder = DeltaDetEncoder(embed_dim=embed_dim).to(device)
        
        # We need a projection head to map YOLO features down to embed_dim
        # YOLOv8m P3 feature map has 192 channels
        self.proj_yolo = nn.Sequential(
            nn.Conv2d(192, embed_dim, kernel_size=1),
            nn.ReLU()
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, iframe_rgb, pframe_mvs, pframe_app, target_boxes, batch_ids):
        """
        iframe_rgb: Not strictly needed if we assume P-frames have fully decoded counterparts for training
                    But the prompt says: "extract codec primitives AND fully decoded RGB for P-frame"
        target_boxes: [N, 4] Ground truth boxes in the P-frame.
        batch_ids: [N] assigning boxes to batch items.
        """
        # 1. Get GT Box Features via YOLO backbone
        with torch.no_grad():
            yolo_feats = self.yolo_anchor.extract_backbone_features(iframe_rgb) # Now iframe_rgb receives pframe_rgbs
            # Use largest feature map (P3) which has stride 8 usually
            # Depends on YOLO architecture exactly, we take index 0
            p3_feat = yolo_feats[0]
            
        # Optional projection
        proj_p3 = self.proj_yolo(p3_feat.to(self.device))
        
        # Format boxes for roi_align: [batch_idx, x1, y1, x2, y2]
        roi_boxes = torch.cat([batch_ids.unsqueeze(1).float(), target_boxes], dim=1)
        
        # RoI Align YOLO features
        yolo_box_feats = roi_align(proj_p3, roi_boxes, output_size=(7, 7), spatial_scale=1/8.0)
        
        # Flatten and we want to compare with delta tokens.
        # However, Delta tokens are sequence tokens [N, 8, 256]. 
        # YOLO box feats are [N, C, 7, 7]. We pool them to [N, 256].
        yolo_pooled = torch.mean(yolo_box_feats, dim=(2, 3)) # [N, 256]
        
        # 2. Extract Delta Tokens
        delta_tokens = self.delta_encoder(pframe_mvs, pframe_app, target_boxes, batch_ids) # [N, 8, 256]
        
        # Pool or align delta tokens to match YOLO feature dimension
        # Since Delta Tokens has length 8, we can mean-pool for MSE comparison
        delta_pooled = torch.mean(delta_tokens, dim=1) # [N, 256]
        
        # 3. Compute Loss
        loss = self.mse_loss(delta_pooled, yolo_pooled.detach())
        return loss

def train_pretrain_epoch(dataloader, model, optimizer):
    model.train()
    total_loss = 0
    from tqdm import tqdm
    
    pbar = tqdm(dataloader, desc="Pretraining Delta Encoder")
    for batch in pbar:
        video_names, iframe_rgbs, pframe_rgbs, pframe_mvs, pframe_res, pframe_depths, pframe_modes, targets = batch
        
        # Flatten batch (since each batch item contains multiple P-frames)
        # target_boxes need to be flattened and associated with a batch index across the flattened P-frames
        all_pframe_rgbs = []
        all_mvs = []
        all_app = []
        all_boxes = []
        all_batch_ids = []
        
        global_pframe_idx = 0
        for b_idx in range(len(video_names)):
            num_pframes = pframe_rgbs[b_idx].shape[0]
            if num_pframes == 0:
                continue
                
            all_pframe_rgbs.append(pframe_rgbs[b_idx].to(model.device))
            
            # Format codec features (N, H, W, C) -> (N, C, H, W)
            mvs = pframe_mvs[b_idx].permute(0, 3, 1, 2).to(model.device)
            res = pframe_res[b_idx].permute(0, 3, 1, 2).to(model.device)
            depths = pframe_depths[b_idx].permute(0, 3, 1, 2).to(model.device).float()
            modes = pframe_modes[b_idx].permute(0, 3, 1, 2).to(model.device).float()
            
            # App features expects 3 channels
            app = torch.cat([res, depths, modes], dim=1)
            
            all_mvs.append(mvs)
            all_app.append(app)
            
            # Extract targets for each P-frame (skip I-frame at index 0)
            target_list = targets[b_idx]
            for p_idx in range(num_pframes):
                t = target_list[p_idx + 1] # targets offset by +1 since 0 is I-frame
                boxes = t['boxes'].to(model.device).clone()
                
                if boxes.shape[0] > 0:
                    # Convert [x1, y1, w, h] to [x1, y1, x2, y2]
                    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                
                # Append to flat lists
                all_boxes.append(boxes)
                # Assign batch_id linking to the flattened spatial tensor
                all_batch_ids.append(torch.full((len(boxes),), global_pframe_idx, dtype=torch.long, device=model.device))
                global_pframe_idx += 1
                
        if len(all_pframe_rgbs) == 0:
            continue
            
        flat_pframe_rgbs = torch.cat(all_pframe_rgbs, dim=0)
        flat_mvs = torch.cat(all_mvs, dim=0)
        flat_app = torch.cat(all_app, dim=0)
        flat_boxes = torch.cat(all_boxes, dim=0)
        flat_ids = torch.cat(all_batch_ids, dim=0)
        
        if len(flat_boxes) == 0:
            continue
            
        optimizer.zero_grad()
        loss = model(flat_pframe_rgbs, flat_mvs, flat_app, flat_boxes, flat_ids)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from data.dataset import BDD100KCoPEDataset, collate_fn
    import os
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Executing Pretraining Sequence on {device}...")
    
    # 1. Dataset
    dataset = BDD100KCoPEDataset(root_dir='D:/cope-delta-det2/data/bdd100k', split='train', gop_length=16, annotated_only=True)
    
    # 2. Dataloader with custom collate_fn
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    print(f"Dataset active. Loaded {len(dataset)} GOPs.")
    
    # 3. Model
    # yolo_size can be 'yolov8n.pt' or 'yolov8m.pt'. Using 'm' assuming standard weight
    model = PretrainAligner(yolo_size='yolov8m.pt', embed_dim=256, device=device).to(device)
    
    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    os.makedirs('checkpoints', exist_ok=True)
    num_epochs = 10
    
    print("Starting Epochs...")
    for epoch in range(num_epochs):
        loss = train_pretrain_epoch(dataloader, model, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {loss:.4f}")
        torch.save(model.delta_encoder.state_dict(), f"checkpoints/delta_encoder_epoch_{epoch+1}.pt")
    
    print("Pretraining complete. Checkpoints saved to /checkpoints.")
