"""
PyTorch training loop for aegisbit profiling.

A small transformer block (2-layer, 512-dim) trained on random data.
Exercises: GEMM (linear layers), softmax, layernorm, dropout, Adam optimizer,
loss backward pass — each producing distinct GPU kernels.

Usage (standalone):
    python3 pytorch_train.py

Usage (profiled):
    AEGISBIT_ENABLED=1 AEGISBIT_MODE=MEMORY_ONLY AEGISBIT_STRATEGY=on_gpu_reduce \
    AEGISBIT_LOG=1 LD_PRELOAD=build/src/libaegisbit.so python3 test/stress/pytorch_train.py
"""

import torch
import torch.nn as nn
import time
import sys

BATCH = 16
SEQ_LEN = 128
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 2
VOCAB = 1000
STEPS = 5


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.Sequential(*[TransformerBlock(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        return self.head(x)


def main():
    device = torch.device("cuda")
    model = TinyLM().to(device).to(torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Warmup (compiles kernels, loads code objects)
    dummy = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    print(f"Warmup done (model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # Training loop
    losses = []
    t0 = time.time()
    for step in range(STEPS):
        x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
        targets = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)

        logits = model(x)
        loss = loss_fn(logits.view(-1, VOCAB), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        losses.append(loss.item())
        print(f"  step {step}: loss={loss.item():.4f}")

    elapsed = time.time() - t0
    print(f"Training: {STEPS} steps in {elapsed:.1f}s ({elapsed/STEPS:.2f}s/step)")
    print(f"Final loss: {losses[-1]:.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
