import math
import torch
from torch import nn
from ..utils.config import Config
from ..network.linear import FeedForward


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe: torch.Tensor = self.pe
        x = x + pe[:, :x.size(1)]
        return self.dropout(x)


class EmbedSequence(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.embed.d_model
        self.vocab_size = config.VOCAB_SIZE
        self.num_heads = config.seq.num_heads
        self.encoder_layers = config.seq.encoder_num_layers
        self.max_seq_len = config.seq.max_seq_len
        self.device = config.device

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=self.max_seq_len + 10)
        
        self.embedding_stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.embed.d_model,
                nhead=self.num_heads,
                batch_first=True,
                dim_feedforward=config.embed.d_model * 4,
                dropout=0.1,
                activation='gelu'
            ),
            self.encoder_layers
        )

        self.output_projection = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, embedding: torch.Tensor, tgt: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None):
        if tgt is None:
            return self._generate_autoregressive(embedding)
        
        return self._forward_teacher_forcing(embedding, tgt, tgt_key_padding_mask)

    def _forward_teacher_forcing(self, memory: torch.Tensor, tgt: torch.Tensor, tgt_key_padding_mask: torch.Tensor = None):
        batch_size = tgt.size(0)
        seq_len = tgt.size(1)
        
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(self.device)
        
        output = self.embedding_stack(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        logits = self.output_projection(output)
        return logits

    def _generate_autoregressive(self, memory: torch.Tensor) -> torch.Tensor:
        batch_size = memory.size(0)
        
        generated = torch.full((batch_size, 1), 22, dtype=torch.long, device=self.device)
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for _ in range(self.max_seq_len):
            tgt_emb = self.token_embedding(generated) * math.sqrt(self.d_model)
            tgt_emb = self.positional_encoding(tgt_emb)
            
            seq_len = generated.size(1)
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(self.device)
            
            output = self.embedding_stack(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask
            )
            
            logits = self.output_projection(output[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            finished = finished | (next_token.squeeze(-1) == 23)
            if finished.all():
                break
        
        return generated[:, 1:]

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=self.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
