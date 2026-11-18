// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Compiler-backed evidence for wave-owned SGPR observation.
//
// The kernel descriptor describes the entry kernel's register use, while the
// separately callable device function deliberately uses s40. On gfx950, the
// resulting kernel descriptor grants s0..s39, but the callee contains
// `s_mov_b32 s40, 1`. This pins the instrumentation scenario in which a
// callable can name a register beyond the entry descriptor. It motivates
// keeping observation attribution inside the wave's simulator storage block;
// it does not claim that the out-of-descriptor access is valid on hardware.

extern "C" __device__ __attribute__((noinline, used, visibility("default"))) void
callable_sgpr_probe() {
  asm volatile("s_mov_b32 s40, 1" ::: "memory");
}

extern "C" __global__ void kernel_calls_s40() { callable_sgpr_probe(); }
