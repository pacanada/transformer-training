import torch.nn as nn
import torch


class Head(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        head_size = config.embedding_dim//config.n_head
        self.key = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.query = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.value = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        att = (q @ k.transpose(-2, -1)) *  C**-0.5
        att = att.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        att = torch.functional.F.softmax(att, dim=-1)
        att =self.dropout(att)
        y = att @ v 
        return y
    
class TokenHead(nn.Module):
    """ This TokenHead acts as the transformer token embedding but instead transform the input
    dimension to out_dimensions. It can be seen as applying some known markers to the univariate
    value of the crypto """
    def __init__(self, input_dim, out_embedding, config) -> None:
        super().__init__()
        self.key = nn.Linear(input_dim, out_embedding, bias=False)
        self.query = nn.Linear(input_dim, out_embedding, bias=False)
        self.value = nn.Linear(input_dim, out_embedding, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        att = (q @ k.transpose(-2, -1)) *  C**-0.5
        att = att.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        att = torch.functional.F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v 
        return y
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return torch.functional.F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in  range(config.n_head)])
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):

        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(x))
    
class FeedForward(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        self.l1 = nn.Linear(embedding_dim, 4*embedding_dim, bias=True)
        self.l2 = nn.Linear(4*embedding_dim, embedding_dim, bias=True)
        self.dropout=nn.Dropout(0.1)
    def forward(self, x):
        x = self.l2(torch.functional.F.relu(self.l1(x)))
        return self.dropout(x)
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffw = FeedForward(embedding_dim=config.embedding_dim)
        self.ln_1 = LayerNorm(config.embedding_dim)
        self.ln_2 = LayerNorm(config.embedding_dim)
    def forward(self, x):
        x = x + self.sa(self.ln_1(x))
        x = x + self.ffw(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """Only considers last token to make predictions"""
    def __init__(self, config) -> None:
        super().__init__()
        self.config=config
        self.token_embedding = TokenHead(input_dim=1, out_embedding=config.embedding_dim, config=config)
        self.positional_encoding = nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.embedding_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_blocks)],
            LayerNorm(ndim=config.embedding_dim),
        )
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
    def forward(self, x):
        x = self.token_embedding(x.view(-1,self.config.block_size,1))+ self.positional_encoding(torch.arange(self.config.block_size, device=x.device))
        x = self.blocks(x)
        x = self.lm_head(x)
        # Cross entropy already have softmax
        return x
    
class Baseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.block_size, 1000)
        self.l2 = nn.Linear(1000, 500)
        self.l3 = nn.Linear(500, 200)
        self.l4 = nn.Linear(200, config.vocab_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x
