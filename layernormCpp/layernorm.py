import torch
import struct
import numpy as np

eps = 1e-5

class LayerNorm:
    """前向传播和反向传播的LayerNorm实现

    Returns:
        _type_: (out, cache) -> out, cache
    """
    @staticmethod
    def forward(x, w, b):
        B, T, C = x.size()
        mean = x.sum(-1, keepdim=True) / C # B,T,1
        xshift = x - mean # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        out = norm * w + b # B,T,C

        cache = (x, w, mean, rstd)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        # gradients for weights, bias
        db = dout.sum((0, 1))
        dw = (dout * norm).sum((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)
        dx *= rstd
        return dx, dw, db

# create a small dummy example and check w.r.t PyTorch backward
# B: batch size, T: sequence length, C: channels
# 创建一个小的虚拟示例，并检查PyTorch反向传播
B = 2
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
b = torch.randn(C, requires_grad=True)
out, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
dx, dw, db = LayerNorm.backward(dout, cache)

# compare to PyTorch autograd
# 与PyTorch autograd比较
fakeloss = (out * dout).sum()
fakeloss.backward()
print("dx error:", (x.grad - dx).abs().max().item())
print("dw error:", (w.grad - dw).abs().max().item())
print("db error:", (b.grad - db).abs().max().item())

# for reference checking in Cpp also
# 用于Cpp中的参考检查
x, w, mean, rstd = cache

def write(tensor, handle):
    handle.write(tensor.detach().numpy().astype("float64").tobytes()) #使用float64类型写入文件以符合C++的double类型

# Write to file
# 写入文件
with open('ln.bin', 'wb') as file:
    write(x, file) # (B, T, C)
    write(w, file) # (C, )
    write(b, file) # (C, )
    write(out, file) # (B, T, C)
    write(mean, file) # (B, T)
    write(rstd, file) # (B, T)
    write(dout, file) # (B, T, C)
    write(dx, file) # (B, T, C)
    write(dw, file) # (C, )
    write(db, file) # (C, )