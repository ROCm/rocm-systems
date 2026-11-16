# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU-reference checks for training and tiled, fused matrix multiplication."""

import copy
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_bias_relu(
    a,
    b,
    bias,
    output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    reduction = tl.arange(0, BLOCK)
    accumulator = tl.zeros((BLOCK, BLOCK), tl.float32)
    for tile in range(tl.cdiv(K, BLOCK)):
        k = tile * BLOCK + reduction
        left = tl.load(
            a + rows[:, None] * AM + k[None, :] * AK,
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        right = tl.load(
            b + k[:, None] * BK + columns[None, :] * BN,
            (k[:, None] < K) & (columns[None, :] < N),
            other=0,
        )
        accumulator += tl.dot(left, right)
    values = tl.maximum(
        accumulator + tl.load(bias + columns, columns < N, other=0)[None, :], 0
    )
    tl.store(
        output + rows[:, None] * N + columns[None, :],
        values,
        (rows[:, None] < M) & (columns[None, :] < N),
    )


def _training(device):
    torch.manual_seed(19)
    reference = torch.nn.Sequential(
        torch.nn.Linear(37, 65), torch.nn.ReLU(), torch.nn.Linear(65, 13)
    )
    model = copy.deepcopy(reference).to(device)
    inputs = torch.randn(19, 37)
    targets = torch.randn(19, 13)
    gpu_inputs, gpu_targets = inputs.to(device), targets.to(device)
    cpu_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
    gpu_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    worst_gradient_error = 0.0
    for _ in range(3):
        cpu_optimizer.zero_grad()
        gpu_optimizer.zero_grad()
        expected = reference(inputs)
        actual = model(gpu_inputs)
        cpu_loss = (expected - targets).square().mean()
        gpu_loss = (actual - gpu_targets).square().mean()
        cpu_loss.backward()
        gpu_loss.backward()
        torch.testing.assert_close(
            actual.detach().cpu(), expected.detach(), rtol=3e-4, atol=1e-5
        )
        torch.testing.assert_close(
            gpu_loss.detach().cpu(), cpu_loss.detach(), rtol=3e-4, atol=1e-5
        )
        for gpu_parameter, cpu_parameter in zip(
            model.parameters(), reference.parameters()
        ):
            gradient = gpu_parameter.grad.cpu()
            torch.testing.assert_close(
                gradient, cpu_parameter.grad, rtol=3e-4, atol=1e-5
            )
            worst_gradient_error = max(
                worst_gradient_error, (gradient - cpu_parameter.grad).abs().max().item()
            )
        cpu_optimizer.step()
        gpu_optimizer.step()
    for gpu_parameter, cpu_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(
            gpu_parameter.detach().cpu(), cpu_parameter.detach(), rtol=3e-4, atol=1e-5
        )
    return {
        "workload": "MLP forward, backward, and SGD",
        "steps": 3,
        "max_gradient_error": worst_gradient_error,
        "output": actual.detach().cpu(),
    }


def _triton_matmul(device):
    generator = torch.Generator().manual_seed(23)
    shapes = ((47, 61, 73), (65, 33, 96))
    worst_error = 0.0
    for m, n, k in shapes:
        a_cpu = torch.randn((m, k), generator=generator, dtype=torch.float16)
        b_cpu = torch.randn((n, k), generator=generator, dtype=torch.float16).T
        bias_cpu = torch.randn(n, generator=generator)
        a, b, bias = a_cpu.to(device), b_cpu.to(device), bias_cpu.to(device)
        assert not b.is_contiguous()
        output = torch.empty((m, n), dtype=torch.float32, device=device)
        _matmul_bias_relu[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
            a,
            b,
            bias,
            output,
            m,
            n,
            k,
            *a.stride(),
            *b.stride(),
            BLOCK=32,
            num_warps=4,
        )
        actual = output.cpu()
        expected = (a_cpu.float() @ b_cpu.float() + bias_cpu).relu()
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)
        worst_error = max(worst_error, (actual - expected).abs().max().item())
    return {
        "workload": "tiled FP16 matmul with bias and ReLU",
        "shapes": shapes,
        "max_output_error": worst_error,
        "output": actual,
    }


def check(framework, device=0):
    with torch.cuda.device(device):
        torch.backends.cuda.matmul.allow_tf32 = False
        target = torch.device("cuda", device)
        if framework == "pytorch":
            result = _training(target)
        elif framework == "triton":
            result = _triton_matmul(target)
        else:
            raise ValueError(framework)
        properties = torch.cuda.get_device_properties(device)
        return {
            "framework": framework,
            "pid": os.getpid(),
            "name": properties.name,
            "arch": properties.gcnArchName,
            "simulated": "memfd:rocjitsu_" in Path("/proc/self/maps").read_text(),
            **result,
        }
