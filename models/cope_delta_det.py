import torch
import torch.nn as nn
from .yolo_anchor import YOLOAnchor
from .delta_encoder import DeltaDetEncoder
from .temporal_fusion import TemporalFusionHead

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.box_utils import xyxy_to_xywh, xywh_to_xyxy

class CoPEDeltaDet(nn.Module):
    """
    CoPE-Δ-Det: Full pipeline combining YOLO I-frame anchoring,
    Δ-Det Encoder for P-frame primitives, and Temporal Fusion Head.
    """
    def __init__(self, yolo_size='yolov8m.pt', embed_dim=256, num_classes=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        
        # 1. Anchor Detector (YOLOv8)
        self.anchor_detector = YOLOAnchor(model_size=yolo_size, device=device)
        
        # 2. Δ-Encoder
        self.delta_encoder = DeltaDetEncoder(embed_dim=embed_dim)
        
        # 3. Temporal Fusion Head
        self.fusion_head = TemporalFusionHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, iframe_rgb, pframe_mvs, pframe_app, gop_length=None):
        """
        Forward pass for a GOP.
        
        iframe_rgb: [B, 3, H, W] - fully decoded keyframe
        pframe_mvs: list of length (gop_length-1). Each element is [B, 2, H, W]
        pframe_app: list of length (gop_length-1). Each element is [B, 3, H, W] for (res, depth, mode)
        
        Returns:
            list of N predictions (one per frame in GOP).
            Each prediction is a tuple: (boxes, confidences, class_probs)
        """
        batch_size = iframe_rgb.shape[0]
        device = iframe_rgb.device
        
        results_per_frame = []
        
        # --- Frame 0 (I-frame) ---
        # Get anchor boxes from YOLO
        # For simplicity in batching here, we process individually or rely on YOLO's internal batching
        anchor_results = self.anchor_detector.get_anchor_boxes(iframe_rgb)
        
        # anchor_results is a list of tensors of shape [Num_Objects, 6] (x, y, w, h, conf, cls) per batch item
        # We need to structure it nicely to pass to subsequent frames
        
        frame_0_boxes = []
        frame_0_confs = []
        frame_0_classes = []
        
        for b_idx in range(batch_size):
            res = anchor_results[b_idx] # [N, 6]
            if res.shape[0] > 0:
                frame_0_boxes.append(res[:, :4])
                frame_0_confs.append(res[:, 4:5])
                # Convert YOLO class ID [N, 1] to one-hot logits [N, num_classes]
                # so frame 0 classes have the same format as P-frame cls_scores
                cls_ids = res[:, 5].long()
                cls_onehot = torch.zeros(res.shape[0], self.num_classes, device=device)
                cls_onehot.scatter_(1, cls_ids.unsqueeze(1), 5.0)  # high logit for detected class
                frame_0_classes.append(cls_onehot)
            else:
                frame_0_boxes.append(torch.empty((0, 4), device=device))
                frame_0_confs.append(torch.empty((0, 1), device=device))
                frame_0_classes.append(torch.empty((0, self.num_classes), device=device))
                
        results_per_frame.append({
            'boxes': frame_0_boxes,
            'confs': frame_0_confs,
            'classes': frame_0_classes
        })
        
        # Keep track of active boxes to propagate
        current_boxes = [b.clone() for b in frame_0_boxes]
        current_confs = [c.clone() for c in frame_0_confs]
        current_classes = [cl.clone() for cl in frame_0_classes]
        
        # --- Frames 1 to N (P-frames) ---
        num_pframes = len(pframe_mvs)
        
        for t in range(num_pframes):
            mvs = pframe_mvs[t] # [B, 2, H, W]
            app = pframe_app[t] # [B, 3, H, W]
            
            # Prepare inputs for Delta Encoder (flattening across batch for RoI Align)
            flat_boxes = []
            batch_ids = []

            for b_idx in range(batch_size):
                b_boxes = current_boxes[b_idx]
                if b_boxes.shape[0] > 0:
                    flat_boxes.append(b_boxes)
                    batch_ids.append(torch.full((b_boxes.shape[0],), b_idx, dtype=torch.long, device=device))
                    
            if len(flat_boxes) > 0:
                flat_boxes_t = torch.cat(flat_boxes, dim=0)
                batch_ids_t = torch.cat(batch_ids, dim=0)
                flat_confs_t = torch.cat([c for _, c in zip(flat_boxes, [current_confs[b] for b in range(batch_size) if current_boxes[b].shape[0] > 0])], dim=0)

                # Convert xyxy -> xywh for fusion head (expects [cx, cy, w, h, conf])
                boxes_xywh = xyxy_to_xywh(flat_boxes_t)
                flat_anchors_5d_t = torch.cat([boxes_xywh, flat_confs_t], dim=1)

                # 1. Delta Encoder (takes xyxy boxes for RoI-align)
                delta_tokens = self.delta_encoder(mvs, app, flat_boxes_t, batch_ids_t)

                # 2. Temporal Fusion (takes xywh + conf)
                box_deltas, conf_updates, cls_scores = self.fusion_head(flat_anchors_5d_t, delta_tokens)

                # 3. Update State: apply deltas in xywh space, convert back to xyxy
                updated_xywh = boxes_xywh + box_deltas
                updated_boxes_flat = xywh_to_xyxy(updated_xywh)
                # conf_t = conf_{t-1} * confidence_update
                updated_confs_flat = flat_confs_t * conf_updates
                
                # Unpack back to batch lists structure
                unpacked_boxes = []
                unpacked_confs = []
                unpacked_classes = []

                ptr = 0
                for b_idx in range(batch_size):
                    num_obj = current_boxes[b_idx].shape[0]
                    if num_obj > 0:
                        unpacked_boxes.append(updated_boxes_flat[ptr:ptr+num_obj])
                        unpacked_confs.append(updated_confs_flat[ptr:ptr+num_obj])
                        # Use fusion head class scores [N, num_classes] instead of
                        # the raw YOLO class ID, so the classification branch
                        # is trainable and the loss receives real logits.
                        unpacked_classes.append(cls_scores[ptr:ptr+num_obj])
                        ptr += num_obj
                    else:
                        unpacked_boxes.append(torch.empty((0, 4), device=device))
                        unpacked_confs.append(torch.empty((0, 1), device=device))
                        unpacked_classes.append(torch.empty((0, self.num_classes), device=device))
                        
                current_boxes = unpacked_boxes
                current_confs = unpacked_confs
                current_classes = unpacked_classes
                
            # Store results for this frame
            results_per_frame.append({
                'boxes': current_boxes,
                'confs': current_confs,
                'classes': current_classes
            })
            
        return results_per_frame

if __name__ == '__main__':
    # Test
    model = CoPEDeltaDet(yolo_size='yolov8n.pt').to('cpu')
    print("Model initialized successfully.")
