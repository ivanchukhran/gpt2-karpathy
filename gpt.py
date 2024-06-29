#!/usr/bin/env python3
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
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
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

    @classmethod
    def from_pretrained(cls, model_type: str):
        '''
        Load a pretrained model from Huggingface
        '''
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large'], f"model_type must be one of ['gpt2', 'gpt2-medium', 'gpt2-large'], but got {model_type}"
        from transformers import GPT2LMHeadModel
        print(f"Loading pretrained model {model_type} from Huggingface")

        config_args = {
            'gpt2': GPTConfig(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': GPTConfig(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': GPTConfig(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': GPTConfig(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = cls(config)
        state_dict = model.state_dict()
        state_dict_keys = state_dict.keys()
        state_dict_keys = [k for k in state_dict_keys if not k.endswith('.attn.bias')] # remove bias for attention layers

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        state_dict_hf = model_hf.state_dict()
        state_dict_hf_keys = state_dict_hf.keys()
        state_dict_hf_keys = [k for k in state_dict_hf_keys if not k.endswith('.attn.masked_bias')]
        state_dict_hf_keys = [k for k in state_dict_hf_keys if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(state_dict_hf_keys) == len(state_dict_keys), f"mismatched keys: {len(state_dict_hf_keys)} != {len(state_dict_keys)}"
        for k in state_dict_hf_keys:
            if any(k.endswith(t) for t in transposed):
                assert state_dict_hf[k].shape == state_dict[k].T.shape, f"mismatched shape: {state_dict_hf[k].shape} != {state_dict[k].T.shape}"
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_hf[k].T)
            else:
                assert state_dict_hf[k].shape == state_dict[k].shape, f"mismatched shape: {state_dict_hf[k].shape} != {state_dict[k].shape}"
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_hf[k])
        return model
