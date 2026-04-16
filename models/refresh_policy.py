import torch
import torch.nn as nn
import torch.nn.functional as F

class SaliencyRefreshPolicy(nn.Module):
    """
    Saliency-Driven Refresh Policy.
    Monitors codec primitive statistics (Residual Energy and MV divergence)
    to trigger a full frame re-decode when new objects appear.
    """
    def __init__(self, alpha=0.7, init_threshold=0.5):
        super().__init__()
        self.alpha = alpha
        
        # Learnable parameters (trained in Stage 3)
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        
        self.reset_states()

    def reset_states(self):
        """Reset EWMA state (e.g. at start of GOP)"""
        self.s_bar = None

    def compute_mv_divergence(self, mvs):
        """
        Computes motion vector divergence per tile.
        mvs: [B, 2, H, W] containing (vx, vy)
        divergence = | d(vx)/dx + d(vy)/dy |
        """
        # Simple finite differences
        vx = mvs[:, 0:1, :, :]
        vy = mvs[:, 1:2, :, :]
        
        # padding to keep shape
        dvx_dx = F.pad(vx[:, :, :, 1:] - vx[:, :, :, :-1], (0, 1, 0, 0))
        dvy_dy = F.pad(vy[:, :, 1:, :] - vy[:, :, :-1, :], (0, 0, 0, 1))
        
        div = torch.abs(dvx_dx + dvy_dy)
        return div

    def forward(self, res_energy, mvs, tracked_boxes, image_size=(720, 1280), tile_size=32):
        """
        res_energy: [B, 1, H, W] (H,W are at 1/16 resolution, so 45x80 for 720x1280)
        mvs: [B, 2, H, W]
        tracked_boxes: List of [N, 4] bounding boxes per batch item.
        
        Note: The formula states "per 32x32 spatial tile". Since our primitives
        are already at 16x16 resolution, a 32x32 spatial tile corresponds to a 2x2 group in primitive grid.
        """
        B, _, H_p, W_p = res_energy.shape
        
        # 1. Compute E_HF_t (High Frequency Energy) - Average over 2x2 blocks
        # using average pooling stride 2
        e_hf = F.avg_pool2d(res_energy, kernel_size=2, stride=2) # [B, 1, H_p/2, W_p/2]
        
        # 2. Compute M_div_t (Motion Divergence)
        m_div_grid = self.compute_mv_divergence(mvs)
        m_div = F.avg_pool2d(m_div_grid, kernel_size=2, stride=2) # [B, 1, H_p/2, W_p/2]
        
        # 3. Compute S_t for each tile
        # ReLU used to ensure saliency is non-negative even if w1, w2 drift
        s_t = F.relu(self.w1 * e_hf + self.w2 * m_div)
        
        # 4. Temporal Smoothing (EWMA)
        if self.s_bar is None or self.s_bar.shape != s_t.shape:
            self.s_bar = s_t
        else:
            self.s_bar = self.alpha * self.s_bar + (1 - self.alpha) * s_t
            
        # 5. Mask out areas covered by tracked boxes
        H_t, W_t = self.s_bar.shape[-2:] # Tile grid sizes
        mask = torch.ones_like(self.s_bar) # 1 means we check this tile
        
        for b_idx in range(B):
            b_boxes = tracked_boxes[b_idx]
            for box in b_boxes:
                x1, y1, x2, y2 = box.long()
                # Map image coords to tile coords
                tx1 = torch.clamp(x1 // tile_size, 0, W_t - 1)
                ty1 = torch.clamp(y1 // tile_size, 0, H_t - 1)
                tx2 = torch.clamp(x2 // tile_size, 0, W_t - 1)
                ty2 = torch.clamp(y2 // tile_size, 0, H_t - 1)
                
                # Zero out the mask inside tracked boxes
                mask[b_idx, 0, ty1:ty2+1, tx1:tx2+1] = 0
                
        # 6. Evaluate trigger
        triggered_tiles = (self.s_bar * mask) > self.threshold
        
        # If any tile triggers, we flag this frame for refresh
        refresh_flags = triggered_tiles.view(B, -1).any(dim=1)
        
        return refresh_flags, self.s_bar

if __name__ == '__main__':
    policy = SaliencyRefreshPolicy()
    B = 2
    res = torch.rand(B, 1, 45, 80)
    mvs = torch.rand(B, 2, 45, 80)
    
    boxes = [
        torch.tensor([[100, 100, 200, 200]]),  # Batch 0
        torch.tensor([[500, 500, 600, 600]])   # Batch 1
    ]
    
    flags, s_bar = policy(res, mvs, boxes)
    print("Refresh Flags:", flags)
    print("S_bar shape:", s_bar.shape) # Ex: [2, 1, 22, 40]
