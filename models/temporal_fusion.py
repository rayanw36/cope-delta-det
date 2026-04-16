import torch
import torch.nn as nn

class TemporalFusionHead(nn.Module):
    """
    Temporal Fusion Head for CoPE-Δ-Det.
    A lightweight transformer decoder that fuses anchor box embeddings (query) 
    with per-object Δ-tokens (key, value) to predict box refinements across P-frames.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=3, num_classes=10):
        super().__init__()
        self.embed_dim = embed_dim
        
        # MLP to encode initial anchor box: [x, y, w, h, conf] -> embed_dim
        self.anchor_encoder = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Prediction Heads
        # 1. Box refinement: (Δx, Δy, Δw, Δh)
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 4)
        )
        
        # 2. Confidence update: Scalar probability multiplier
        self.conf_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 3. Class consistency score (optional)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, anchor_boxes, delta_tokens):
        """
        anchor_boxes: [Num_Objects, 5] format (x, y, w, h, conf)
        delta_tokens: [Num_Objects, sequence_length (8), embed_dim]
        """
        # 1. Embed anchor boxes to use as query
        # Shape: [Num_Objects, 1, embed_dim]
        q_anchors = self.anchor_encoder(anchor_boxes).unsqueeze(1)
        
        # 2. Transformer decoding
        # query = anchors, memory = delta_tokens
        out_fused = self.transformer(tgt=q_anchors, memory=delta_tokens) 
        
        # Squeeze out the sequence dim: [Num_Objects, embed_dim]
        out_fused = out_fused.squeeze(1)
        
        # 3. Predict updates
        box_deltas = self.box_head(out_fused)     # [Num_Objects, 4]
        conf_updates = self.conf_head(out_fused)  # [Num_Objects, 1]
        cls_scores = self.cls_head(out_fused)     # [Num_Objects, num_classes]
        
        return box_deltas, conf_updates, cls_scores

if __name__ == '__main__':
    # Test
    model = TemporalFusionHead(num_classes=10)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    
    # 5 objects in this frame
    anchors = torch.rand(5, 5) # 5 objects, [x, y, w, h, conf]
    deltas = torch.randn(5, 8, 256) # 8 tokens per object of dim 256
    
    box_deltas, conf_updates, cls_scores = model(anchors, deltas)
    
    print("Box Deltas:", box_deltas.shape)       # Exp: [5, 4]
    print("Conf Updates:", conf_updates.shape)   # Exp: [5, 1]
    print("Class Scores:", cls_scores.shape)     # Exp: [5, 10]
