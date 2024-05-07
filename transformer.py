import torch
import torch.nn as nn
from torch.nn import functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# Heads
class Head(nn.Module):
    '''
    One head of self-attention
    '''
    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        input of size (batch, time-step, channels)
        output of size (batch, time-step, had size)

        What Mask_fill doing:
        suppose we have 
        [1, 0, 0]
        [1, 0.6, 0]
        [1, 0.6, 0.4]
        mask fill do:
        [1, -inf, -inf]
        [1, 0.6, -inf]
        [1, 0.6, 0.4]
        '''
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        attn = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T), for softmax calculation
        attn = F.softmax(attn, dim=-1) # (B, T, T), Highlight the attention score
        attn = self.dropout(attn)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, hs)
        out = attn @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

# Multi-head attention
class MultiHeadAttention(nn.Module):
    '''
    Multiple head fo self-attention in parallel
    '''
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        # ModuleList put every computing in parallel, which is different from Sequential(). 
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd) # prjection or transfomation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # （B，T, F) -> (B, T, [h0, h0, h0, h0, h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        out = self.dropout(self.proj(out))
        return out

# Feed forward layer
class FeedFoward(nn.Module):
    '''
    a simple linear layer followed by a non-linearity
    '''
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# Decoder block
class Block(nn.Module):
    '''
    Transformer block, communication followed by computation
    '''
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        # n_embd: embedding dimension
        # n_head: the number of heads we choose
        super().__init__()
        head_size = n_embd // n_head
        self.MHAt = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.feedfwd = FeedFoward(n_embd, dropout)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.MHAt(x)
        x = self.norm1(x+y)
        y = self.feedfwd(x)
        x = self.norm2(x+y)
        return x

# Transformer arhitecutre
class largelanguagemodel(nn.Module):
    def __init__(self, vocab_size, n_embd=384, n_layer=4, n_head=4, block_size=8, dropout=0.2):
        super().__init__()
        '''
        for example of embedding vector
        "sad", embedding_vector = [0.1, 0.8]
        "happy", embedding_vector = [0.9, 0.1]
        '''
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)]) # decoder blocks
        # Transformation of features
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, index, targets=None):
        B, T = index.shape
        # idx and targets are both (B, T) tensor of integers
        tok_em = self.token_embedding(index) # (B, T, C)
        pos_em = self.position_embedding(torch.arange(T, device=device)) # (T,C)
        x = tok_em + pos_em # (B,T,C)        
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)    
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, index, max_new_tokens, block_size = 8):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:] # crop dix to the last block size tokens
            logits, loss = self.forward(index_cond) # get predictions
            logits = logits[:, -1, :] # focus only on the lost time step, becomes (B, C)
            probs = F.softmax(logits, dim=-1) # apply softmax to get probabilities, (B, C)
            index_next= torch.multinomial(probs, num_samples=1) # sample from the distribution, (B, 1)
            index = torch.cat((index, index_next), dim=1) # append sampled index to the running sequence (B, T+1)
        return index






