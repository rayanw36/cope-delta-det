import torch
import torch.nn as nn
from torchvision.ops import roi_align

class PatchEmbedding(nn.Module):
    """Patchifies an HxWxD tensor to a grid of patches and embeds them."""
    def __init__(self, in_channels, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Using a Convolution to implement non-overlapping patchification + linear projection
        # This is equivalent to flattening a patch and applying a linear layer (shared weights)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x is [B, C, H, W]
        x = self.proj(x) # [B, embed_dim, H/16, W/16]
        # B, C, H, W -> B, H, W, C for linear layer
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)
        # B, H, W, C -> B, C, H, W for RoI-Align
        x = x.permute(0, 3, 1, 2)
        return x

class TokenTransformer(nn.Module):
    """2-layer transformer with learnable query tokens."""
    def __init__(self, num_queries=4, embed_dim=256, num_heads=4):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        # Transformer Decoder Layer can act as cross-attention
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=2)

    def forward(self, features):
        # features: [Num_Objects, sequence_length (49), embed_dim]
        # queries: [Num_Objects, num_queries, embed_dim]
        num_objs = features.shape[0]
        queries = self.query_tokens.repeat(num_objs, 1, 1)
        
        # PyTorch TransformerDecoder takes target, memory
        # tgt = queries, memory = features
        out = self.transformer(tgt=queries, memory=features)
        return out # [Num_Objects, num_queries, embed_dim]

class DeltaDetEncoder(nn.Module):
    """
    Δ-Det Encoder.
    Processes P-frame primitives into per-object Δ-tokens representing motion and appearance updates.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        
        # --- Motion Branch ---
        # Input MV tensor is at 1/16 resolution (45x80 for 720x1280), already block-level
        # Use patch_size=1 to preserve spatial resolution (1x1 conv projection)
        self.motion_embed = PatchEmbedding(in_channels=2, patch_size=1, embed_dim=embed_dim)
        self.motion_transformer = TokenTransformer(num_queries=4, embed_dim=embed_dim)
        
        # --- Appearance/Residual Branch ---
        # Input: res_energy(1), part_depth(1), pred_mode(1) -> 3 channels
        # Input is already at 1/16 resolution (45x80), so use a lightweight conv stack
        # that preserves spatial resolution instead of ResNet18 (which would reduce to ~2x5)
        self.app_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.app_transformer = TokenTransformer(num_queries=4, embed_dim=embed_dim)

    def forward(self, mvs, app_inputs, boxes, batch_ids):
        """
        mvs: [B, 2, H, W]
        app_inputs: [B, 3, H, W] (res_energy, part_depth, pred_mode)
        boxes: List of bounding boxes [num_objects, 4] format [x_min, y_min, x_max, y_max]
        batch_ids: [num_objects] tensor mapping box to batch index
        """
        # 1. Min-max normalize MVs to [-1, 1] 
        # (Assuming they aren't pre-normalized, we can clamp/scale dynamically or assume pre-scaling)
        mvs_norm = torch.clamp(mvs / 64.0, -1.0, 1.0) # Assuming 64 pixel max displacement
        
        # 2. Extract spatial feature grids
        mot_grid = self.motion_embed(mvs_norm) # [B, 256, H/16, W/16]
        app_grid = self.app_features(app_inputs) # [B, 256, H/16, W/16]
        
        # 3. RoI-align for each object
        # RoI align expects boxes in format [batch_idx, x1, y1, x2, y2]
        # boxes coordinate space is original frame scale. Since our grids are 1/16, spatial_scale=1/16
        roi_boxes = torch.cat([batch_ids.unsqueeze(1).float(), boxes], dim=1)
        
        # Expand boxes by 20% context padding (optional but specified in prompt)
        # width = roi_boxes[:, 3] - roi_boxes[:, 1]
        # height = roi_boxes[:, 4] - roi_boxes[:, 2]
        # roi_boxes[:, 1] -= width * 0.1
        # roi_boxes[:, 2] -= height * 0.1
        # roi_boxes[:, 3] += width * 0.1
        # roi_boxes[:, 4] += height * 0.1
        
        mot_roi = roi_align(mot_grid, roi_boxes, output_size=(7, 7), spatial_scale=1/16.0) # [N, 256, 7, 7]
        app_roi = roi_align(app_grid, roi_boxes, output_size=(7, 7), spatial_scale=1/16.0) # [N, 256, 7, 7]
        
        # Flatten RoI features to sequence: [N, 49, 256]
        N, C, H_roi, W_roi = mot_roi.shape
        mot_seq = mot_roi.view(N, C, H_roi * W_roi).permute(0, 2, 1)
        app_seq = app_roi.view(N, C, H_roi * W_roi).permute(0, 2, 1)
        
        # 4. Attention mechanism to get K tokens per object
        mot_tokens = self.motion_transformer(mot_seq) # [N, 4, 256]
        app_tokens = self.app_transformer(app_seq) # [N, 4, 256]
        
        # 5. Concatenate
        delta_tokens = torch.cat([mot_tokens, app_tokens], dim=1) # [N, 8, 256]
        
        return delta_tokens

if __name__ == '__main__':
    # Test with 1/16 resolution inputs (45x80 for 720x1280 video)
    model = DeltaDetEncoder()
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    b = 2
    mvs = torch.randn(b, 2, 45, 80)   # 1/16 resolution codec primitives
    app = torch.randn(b, 3, 45, 80)    # res_energy + part_depth + pred_mode

    # 3 objects in batch 0, 2 in batch 1 (pixel coordinates)
    boxes = torch.tensor([
        [100, 100, 200, 200],
        [300, 300, 400, 400],
        [500, 500, 600, 600],
        [50, 50, 150, 150],
        [600, 100, 700, 200]
    ], dtype=torch.float32)
    batch_ids = torch.tensor([0, 0, 0, 1, 1])

    out = model(mvs, app, boxes, batch_ids)
    print("Output shape:", out.shape) # Expected: [5, 8, 256]
