#!/usr/bin/env python3
"""Compile a Triton add_kernel and dump its assembly."""
import torch
import triton
import triton.language as tl
import os
import glob


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


n = 1024
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
out = torch.empty_like(x)
add_kernel[(4,)](x, y, out, n, BLOCK_SIZE=256)
torch.cuda.synchronize()

# Search for the compiled kernel
cache_dir = os.path.expanduser("~/.triton/cache")
print(f"Triton cache: {cache_dir}")
for root, dirs, files in os.walk(cache_dir):
    for f in sorted(files):
        if f.endswith(".hsaco") or f.endswith(".amdgcn") or f.endswith(".cubin"):
            path = os.path.join(root, f)
            print(f"  {path} ({os.path.getsize(path)} bytes)")
            os.system(f"/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 {path} 2>&1 | head -100")
            print()
