#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Minimal PyTorch training test: fits sin(x)/x on the GPU with an MLP.

Passes only if the error on points it never trained on falls close to zero.
That covers device discovery, host-to-device copies, nn.Linear GEMMs via
rocBLAS, autograd, AdamW state and convergence. Data is TARGET at random
points.
"""

import argparse
import math
import sys
import time

# Training epochs, overridable via CLI
DEFAULT_EPOCHS = 200
DEFAULT_REPORT_EVERY = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.partition("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="passes over the training set; more of them buy a closer fit",
    )
    parser.add_argument(
        "-r",
        "--report-every",
        type=int,
        default=DEFAULT_REPORT_EVERY,
        metavar="N",
        help="print the training loss every N epochs",
    )
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.report_every < 1:
        parser.error("--report-every must be at least 1")
    return args


_CLI_ARGS = parse_args() if __name__ == "__main__" else None

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# ---------------------------------------------------------------------------


def sinc(x):
    # sin(x)/x with the x=0 limit (1) filled in. torch.where evaluates both
    # branches, so the denominator is patched to 1 before the divide, not after.
    ones = torch.ones_like(x)
    return torch.where(x == 0, ones, torch.sin(x) / torch.where(x == 0, ones, x))


TARGET = sinc
DOMAIN = (-10.0, 10.0)
ACTIVATION = nn.Tanh

# Capacity. More of either fits a wigglier TARGET, at more work per step.
HIDDEN_LAYERS = 3
HIDDEN_DIM = 256

# Training uses the 8192 points in batches of 1024, so the weights get updated
# 8 times per epoch; LEARNING_RATE scales how far each update moves them. The
# 4096 validation points are only scored, never trained on, so they show whether
# the fit generalizes rather than memorizes. SEED makes every run identical.
TRAIN_POINTS = 8192
VAL_POINTS = 4096
BATCH = 1024
LEARNING_RATE = 1e-3
SEED = 1234

# ASCII plot size, in characters
PLOT_WIDTH = 76
PLOT_HEIGHT = 21

# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Linear layers with ACTIVATION between them, one number in and one out.

    forward rescales the input to about [-1, 1], the scale Linear's default
    initialization assumes. The last Linear is bare on purpose: an activation
    there would clamp the output to its own range, and sin(x)/x goes negative.
    """

    def __init__(self):
        super().__init__()
        low, high = DOMAIN
        # Buffers, not parameters: never trained, but they follow .to(device).
        self.register_buffer("center", torch.tensor((low + high) / 2))
        self.register_buffer("half_width", torch.tensor((high - low) / 2))

        layers = []
        width = 1
        for _ in range(HIDDEN_LAYERS):
            layers += [nn.Linear(width, HIDDEN_DIM), ACTIVATION()]
            width = HIDDEN_DIM
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net((x - self.center) / self.half_width)


def sample(count, generator):
    """`count` points drawn uniformly from DOMAIN, paired with TARGET there.

    Random rather than gridded, so the held-out split lands on unseen points.
    """
    low, high = DOMAIN
    # (count, 1) to match the model's output shape, so the loss cannot broadcast.
    inputs = torch.rand(count, 1, generator=generator) * (high - low) + low
    return inputs, TARGET(inputs)


def train_one_epoch(model, optimizer, inputs, targets, device):
    """Train over every batch once and return the mean loss."""
    model.train()
    total = 0.0
    batches = 0
    for start in range(0, len(inputs), BATCH):
        x = inputs[start : start + BATCH].to(device)
        y = targets[start : start + BATCH].to(device)
        # Gradients accumulate into .grad, so they need clearing every step.
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        batches += 1
    return total / batches


@torch.no_grad()
def evaluate(model, inputs, targets, device):
    """Mean squared and worst-case error on points that were never trained on."""
    model.eval()
    predictions = model(inputs.to(device))
    targets = targets.to(device)
    mse = F.mse_loss(predictions, targets).item()
    return mse, (predictions - targets).abs().max().item()


@torch.no_grad()
def print_plot(model, device):
    """Draw TARGET and the model's fit over DOMAIN as ASCII.

    Both curves share one grid: '#' where they land in the same cell, 'o' and
    '+' where they differ. A good fit reads as one curve, a bad one as two.
    """
    model.eval()
    low, high = DOMAIN
    inputs = torch.linspace(low, high, PLOT_WIDTH).unsqueeze(1)
    targets = TARGET(inputs).flatten().tolist()
    predictions = model(inputs.to(device)).cpu().flatten().tolist()

    # Frame both curves; `or` survives a flat one that would give span 0.
    ceiling = max(max(targets), max(predictions))
    floor = min(min(targets), min(predictions))
    span = (ceiling - floor) or 1.0

    def row_of(value):
        """Grid row holding `value`, counting from 0 at the top."""
        return round((ceiling - value) / span * (PLOT_HEIGHT - 1))

    # One comprehension per row: [[" "] * W] * H would alias a single row.
    grid = [[" "] * PLOT_WIDTH for _ in range(PLOT_HEIGHT)]
    for column, (target, prediction) in enumerate(zip(targets, predictions)):
        target_row, prediction_row = row_of(target), row_of(prediction)
        if target_row == prediction_row:
            grid[target_row][column] = "#"
        else:
            grid[target_row][column] = "o"
            grid[prediction_row][column] = "+"

    print(f"plot    : {TARGET.__name__} 'o'  fit '+'  overlapping '#'")
    for index, row in enumerate(grid):
        value = ceiling - span * index / (PLOT_HEIGHT - 1)
        print(f"  {value:7.3f} |{''.join(row)}")
    # The 10 spaces put the axis corner under the '|' of the labels above.
    print(f"{'':10}+{'-' * PLOT_WIDTH}")
    left, right = f"{low:g}", f"{high:g}"
    print(f"{'':11}{left}{right:>{PLOT_WIDTH - len(left)}}")


def require_gpu():
    if not torch.cuda.is_available():
        sys.exit("FATAL: no GPU visible to PyTorch (torch.cuda.is_available() is False)")
    return torch.device("cuda")


def main(args=None):
    if args is None:
        args = parse_args()
    device = require_gpu()
    props = torch.cuda.get_device_properties(0)
    print(f"device : {props.name} ({props.gcnArchName})")
    print(f"torch  : {torch.__version__}  hip {torch.version.hip}")

    # Global seed for the weight init; a separate generator for the data, so
    # resizing the network cannot shift which points get sampled.
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    train_inputs, train_targets = sample(TRAIN_POINTS, generator)
    val_inputs, val_targets = sample(VAL_POINTS, generator)

    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"target : {TARGET.__name__}(x) on [{DOMAIN[0]:g}, {DOMAIN[1]:g}]")
    print(
        f"model  : {HIDDEN_LAYERS}x{HIDDEN_DIM} {ACTIVATION.__name__}, "
        f"{sum(p.numel() for p in model.parameters()) / 1e3:.1f}K params"
    )

    # Predicting the mean everywhere scores exactly the variance, so the pass bar
    # is a fraction of that rather than an absolute number: it scales with how
    # hard TARGET is. unbiased=False divides by N, as MSE does.
    baseline = val_targets.var(unbiased=False).item()
    max_mse = 0.01 * baseline
    print(f"baseline: mse {baseline:.6f} from a constant prediction")

    losses = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        losses.append(
            train_one_epoch(model, optimizer, train_inputs, train_targets, device)
        )
        # First and last always report, so the loss trend is bounded on both ends.
        if epoch in (1, args.epochs) or epoch % args.report_every == 0:
            print(f"epoch {epoch:5d}/{args.epochs}  train mse {losses[-1]:.6f}")
    # GPU work is asynchronous, so the queue has to drain before the clock stops.
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    val_mse, max_error = evaluate(model, val_inputs, val_targets, device)
    steps = args.epochs * math.ceil(TRAIN_POINTS / BATCH)
    print(f"elapsed : {elapsed:.2f}s ({elapsed / steps * 1e3:.2f} ms/step)")
    print(f"memory  : {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB peak")
    print(f"held out: mse {val_mse:.6f}  worst error {max_error:.4f}")
    print_plot(model, device)

    converged = val_mse < max_mse
    # NaN is the one value not equal to itself, so `x == x` is the NaN test.
    finite = all(loss == loss and abs(loss) != float("inf") for loss in losses)
    finite = finite and val_mse == val_mse
    status = "pass" if (converged and finite) else "fail"
    print(
        f"RESULT: status={status} epochs={args.epochs} "
        f"train_mse_last={losses[-1]:.6f} "
        f"val_mse={val_mse:.6f} baseline={baseline:.6f}"
    )

    # Diagnostics on stderr, keeping the RESULT line above clean for a grep.
    if not finite:
        print("FAIL: non-finite loss encountered", file=sys.stderr)
    elif not converged:
        print(
            f"FAIL: held-out mse {val_mse:.6f} did not fall below " f"{max_mse:.6f}",
            file=sys.stderr,
        )
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main(_CLI_ARGS))
