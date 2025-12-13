import torch
import torch.nn as nn
import math

class Linear(nn.Module):
  def __init__(self,fan_out:int,fan_in:int,device=None,dtype=None):
    super().__init__()
    self.fan_in=fan_in
    self.fan_out=fan_out
    self.device=device
    self.dtype=dtype
    self.W=nn.Parameter(torch.empty(fan_out, fan_in,device=device, dtype=dtype))
    self.std=math.sqrt(2/(fan_in+fan_out))
    nn.init.trunc_normal_(self.W,0,self.std,-3*self.std,3*self.std)

  def forward(self,x):
    return x@self.W.transpose(0,1)

class Embedding(nn.Module):
  def __init__(self,num_embeddings:int, embedding_dim:int,device=None,dtype=None):
    super().__init__()
    self.num_embeddings=num_embeddings
    self.embedding_dim=embedding_dim
    self.device=device
    self.dtype=dtype
    self.W=nn.Parameter(torch.empty(num_embeddings, embedding_dim,device=device, dtype=dtype))
    self.std=math.sqrt(2/(embedding_dim+ num_embeddings))
    nn.init.trunc_normal_(self.W,0,self.std,-3*self.std,3*self.std)

  def forward(self,id_tensor):
    return self.W[id_tensor]#*math.sqrt(self.embedding_dim)

class RMSnorm(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    super().__init__()
    self.d_model=d_model
    self.eps=eps
    self.g=nn.Parameter(torch.ones(d_model,device=device, dtype=dtype))

  def forward(self,x):
    root_avg=x.pow(2).mean(dim=-1, keepdim=True)
    x=x/torch.sqrt(root_avg+self.eps)
    return x*self.g

class positionwise_feedforward(nn.Module):
  def __init__(self,d_model:int,dff=0,device=None, dtype=None):
    super().__init__()
    self.d_model=d_model
    if dff>0:
      self.dff=dff
    else:
      self.dff=int(((8*self.d_model/3)//64+1))*64
    self.W1=nn.Parameter(torch.empty(self.dff, self.d_model,device=device, dtype=dtype))
    self.W2=nn.Parameter(torch.empty(self.d_model, self.dff,device=device, dtype=dtype))
    self.W3=nn.Parameter(torch.empty(self.dff, self.d_model,device=device, dtype=dtype))
    self.std=math.sqrt(2/(self.dff+ self.d_model))

    nn.init.trunc_normal_(self.W1,0,self.std,-3*self.std,3*self.std)
    nn.init.trunc_normal_(self.W2,0,self.std,-3*self.std,3*self.std)
    nn.init.trunc_normal_(self.W3,0,self.std,-3*self.std,3*self.std)

  def SILU(self,x):
    return x/(1+torch.exp(-x))

  def forward(self,x):
    w1x=x@self.W1.transpose(0,1)
    w3x=x@self.W3.transpose(0,1)
    w1x=self.SILU(w1x)
    w1x=w1x*w3x
    w1x=w1x@self.W2.transpose(0,1)
    return w1x

class RoPE(nn.Module):
  def __init__(self,theta: float,d_k:int,max_seq_len:int,device=None):
    super().__init__()
    self.theta=theta
    self.d_k=d_k
    self.max_seq_len=max_seq_len

    assert self.d_k%2==0, "d_k is not divisible by 2"

    self.d=self.d_k//2

    inv_freq = self.theta ** (-2 * torch.arange(self.d, device=device, dtype=torch.float32) / self.d_k)

    self.register_buffer('inv_freq',inv_freq)

  def forward(self,x,token_positions):
    token_positions = token_positions.to(self.inv_freq.device).long()
    angles=token_positions[..., None].float() * self.inv_freq[None, :]
    cos=angles.cos()
    sin=angles.sin()
    d_k=x.shape[-1]

    x_ = x.view(*x.shape[:-1], d_k // 2, 2)
    x_even = x_[..., 0]
    x_odd = x_[..., 1]

    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_even * sin + x_odd * cos

    x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)
    x_rot = x_rot.reshape(*x.shape)
    return x_rot

def softmax(t, dim=-1):
    m = t.max(dim=dim, keepdim=True).values
    x = t - m
    ex = x.exp()
    d = ex.sum(dim=dim, keepdim=True)
    return ex / d

def scaled_dot_product_attention(q,k,v,mask=None):
  d_k=q.shape[-1]
  x=q @ k.transpose(-2,-1)
  x=x/math.sqrt(d_k)

  if mask is not None:
    x = x.masked_fill(mask==False, float("-inf"))

  x = torch.clamp(x, min=-1e5, max=1e5)

  x=softmax(x,-1)
  return x @ v

class multihead_self_attention(nn.Module):
  def __init__(self,d_model:int,h:int,theta:float,max_seq_len:int,device=None, dtype=None):
    super().__init__()
    self.d_model=d_model
    self.h=h
    self.theta=theta
    self.max_seq_len=max_seq_len

    assert d_model%h==0, "d_model is not divisible by h"

    self.d_k=d_model//h

    self.WQ=Linear(self.d_model,self.d_model,device=device, dtype=dtype)
    self.WK=Linear(self.d_model,self.d_model,device=device, dtype=dtype)
    self.WV=Linear(self.d_model,self.d_model,device=device, dtype=dtype)
    self.WO=Linear(self.d_model,self.d_model,device=device, dtype=dtype)

    self.rope=RoPE(self.theta, self.d_k, self.max_seq_len, device=device)

  def forward(self,x,mask=None):
    q=self.WQ(x)
    k=self.WK(x)
    v=self.WV(x)
    seq_len=x.shape[-2]

    old_shape=q.shape[:-1]

    q=q.view(*old_shape,self.h,self.d_k).transpose(-3,-2)
    k=k.view(*old_shape,self.h,self.d_k).transpose(-3,-2)
    v=v.view(*old_shape,self.h,self.d_k).transpose(-3,-2)

    token_positions = torch.arange(seq_len,device=q.device, dtype=torch.long)
    token_positions = token_positions.view(*([1] * (q.ndim - 2)),seq_len).expand(*q.shape[:-1])

    q=self.rope(q,token_positions)
    k=self.rope(k,token_positions)

    if mask is None:
      # Causal lower-triangular mask on the SAME device as Q, bool dtype
      mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
      mask = mask.view(1, 1, seq_len, seq_len)

    scores=scaled_dot_product_attention(q,k,v,mask)

    scores=scores.transpose(-3,-2).contiguous().view(*old_shape,self.d_model)

    return self.WO(scores)

class transformer_block(nn.Module):
  def __init__(self,d_model:int,h:int,d_ff:int,theta:float,max_seq_len:int,mask=None,device=None,dtype=None):
    super().__init__()
    self.d_model=d_model
    self.h=h
    self.d_ff=d_ff
    self.theta=theta
    self.max_seq_len=max_seq_len
    self.mask=mask
    self.layer_1=RMSnorm(self.d_model,device=device, dtype=dtype)
    self.layer_2=multihead_self_attention(self.d_model,self.h,self.theta,self.max_seq_len,device=device, dtype=dtype)
    self.layer_3=RMSnorm(self.d_model,device=device, dtype=dtype)
    self.layer_4=positionwise_feedforward(self.d_model,self.d_ff,device=device, dtype=dtype)

  def forward(self,x):
    y=x+self.layer_2(self.layer_1(x))
    return y+self.layer_4(self.layer_3(y))

class transformer_lm(nn.Module):
  def __init__(self,d_model:int,h:int,d_ff:int,vocab_size:int, context_length:int, num_layers:int,theta:float,mask=None,device=None,dtype=None):
    super().__init__()
    self.d_model=d_model
    self.h=h
    self.d_ff=d_ff
    self.vocab_size=vocab_size
    self.context_length=context_length
    self.num_layers=num_layers
    self.theta=theta
    self.mask=mask

    self.embedding_layer=Embedding(self.vocab_size,self.d_model,device=device, dtype=dtype)
    self.attention_blocks = nn.ModuleList([
      transformer_block(self.d_model, self.h, self.d_ff, self.theta, self.context_length,self.mask,device=device, dtype=dtype)
      for _ in range(num_layers)
    ])
    self.norm_layer=RMSnorm(self.d_model,device=device, dtype=dtype)
    self.linear_layer=Linear(self.vocab_size,self.d_model,device=device, dtype=dtype)

  def forward(self,x):
    x=self.embedding_layer(x)
    for layer in self.attention_blocks:
      x=layer(x)
    x=self.norm_layer(x)
    x=self.linear_layer(x)
    return x