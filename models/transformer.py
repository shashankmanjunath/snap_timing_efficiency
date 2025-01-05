import math

import torch.nn as nn
import torch


__all__ = ["TransformerTimeSeries"]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of [max_len, d_model] representing positions and dimensions
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout_prob: float) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob
        self.attention = nn.MultiheadAttention(
            self.embed_dim,
            self.n_heads,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
        )
        self.dropout = nn.Dropout(self.dropout_prob)
        self.ln1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        fc_out = self.fc(x)
        x = x + self.dropout(fc_out)
        x = self.ln2(x)

        return x


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        input_dim_features: int,
        embed_dim: int,
    ):
        super().__init__()
        self.feat_dim = input_dim_features
        self.embed_dim = embed_dim
        self.feat_projection = nn.Linear(self.feat_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = self.feat_projection(x)
        x = x.reshape(batch_size, seq_len, -1)
        return x


class TransformerTimeSeries(nn.Module):
    def __init__(self, embed_dim, n_heads, n_blocks, n_categorical):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_feat_dim = 22 * self.embed_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.dropout_prob = 0.5
        self.n_categorical = n_categorical
        self.forecast_input_dim = self.seq_feat_dim + self.n_categorical

        self.embedding_layer = EmbeddingLayer(input_dim_features=13, embed_dim=16)
        #  self.pos_encoding = PositionalEncoding(self.seq_feat_dim, max_len=1000)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.seq_feat_dim,
                    self.n_heads,
                    self.dropout_prob,
                )
                for n in range(self.n_blocks)
            ]
        )

        self.forecast_head = nn.Sequential(
            nn.Linear(self.forecast_input_dim, self.seq_feat_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.seq_feat_dim // 2, self.seq_feat_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(self.seq_feat_dim // 4, 1),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_mask: torch.Tensor,
        x_category: torch.Tensor,
    ) -> torch.Tensor:
        x_embed = self.embedding_layer(x_seq)
        #  x_embed = self.pos_encoding(x_embed)
        for block in self.blocks:
            x_embed = block(x_embed, x_mask)

        # Mean pooling
        x_pred = x_embed.mean(dim=1)
        x_pred = torch.cat((x_pred, x_category), dim=-1)
        x_pred = self.forecast_head(x_pred)
        return x_pred
