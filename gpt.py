#!/usr/bin/env python3
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x: torch.Tensor):
        return self.c_proj(self.gelu(self.c_fc(x)))

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"n_embd must be divisible by n_head, but got n_embd = {config.n_embd} and n_head = {config.n_head}"
        # key, query, value projections for all heads but in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask",
                             torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the second dimension
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # The large (T, T) matrix for all queries and keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd), # weights of token embeddings
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd), # weights of positional embeddings
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
            ln_f = nn.LayerNorm(self.config.n_embd)
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        # x is of shape (B, T)
        B, T = x.size()
        assert T <= self.config.block_size, f"input sequence length {T} is greater than block size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos) # positional embeddings (T, n_embd)
        tok_emb = self.transformer.wte(x)   # token embeddings (B, T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)            # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(('.attn.bias', '.attn.mask'))] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(('.attn.masked_bias', '.attn.bias'))] # ignore these, just a buffer
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        if len(sd_keys_hf) != len(sd_keys):
            print("WARNING: number of parameters in source and target GPT models differ")
            print("source:", len(sd_keys_hf), "target:", len(sd_keys))
            source_set = set(sd_keys_hf)
            target_set = set(sd_keys)
            intersection = source_set & target_set
            source_missing = target_set - intersection
            target_missing = source_set - intersection
            print("missing in source:", source_missing)
            print("missing in target:", target_missing)
            raise ValueError("could not align source and target model parameters")
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


if __name__ == '__main__':
    num_return_sequences = 5
    max_length = 30

    model = GPT.from_pretrained('gpt2')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"loaded model to {device}")

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)


    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        while x.size() < max_length:
            logits = model(x)
            # take the logits at the last position
            logits = logits[:, -1, :]
            # get probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (hf default)
            # topk_probs here becomes (5, 50), topk_indices becomes (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            x = torch.multinomial(topk_probs, num_samples=1)
