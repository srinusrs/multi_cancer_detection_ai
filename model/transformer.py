import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularBiomarkerTransformer(nn.Module):
    """Option 1: Transformer Architecture for continuous genomic tabular inputs."""
    def __init__(self, num_features=100, embed_dim=32, num_heads=4, num_layers=2, num_classes=4):
        super().__init__()
        # Map every independent continuous scalar feature into a structured vector space
        self.projections = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_features)])
        
        # Learnable structural Classification Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=64,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        embeddings = []
        
        # Construct sequential feature tokens from flat tables
        for i, proj_layer in enumerate(self.projections):
            feature_column = x[:, i].unsqueeze(-1) # [Batch, 1]
            embeddings.append(proj_layer(feature_column).unsqueeze(1)) # [Batch, 1, EmbedDim]
            
        x_space = torch.cat(embeddings, dim=1) # [Batch, NumFeatures, EmbedDim]
        
        # Append classification token to head sequence position
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_input = torch.cat((cls_tokens, x_space), dim=1) # [Batch, NumFeatures + 1, EmbedDim]
        
        x_out = self.transformer(x_input)
        cls_rep = self.norm(x_out[:, 0]) # Extract structural sequence output layer
        return self.classifier(cls_rep)
