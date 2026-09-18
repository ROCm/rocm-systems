// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Compiler-backed evidence that argument 0 arrives in v0.
//
// The whole probe-argument feature rests on the AMDGPU calling convention
// placing explicit arguments in VGPRs from v0 in declaration order. Every other
// test asserts that against a hand-assembled body; this one asserts it against
// what amdclang++ actually emits. Consumed by tests/dbi/probe_fixture_test.cpp,
// which declares the probe with one argument (accepted, nothing left over) and
// with none (rejected, naming v0).
//
// See rj_nop_probe.hip for what the attributes and the keepalive pointer are
// for -- both are needed here for the same reasons.
//
// The argument is consumed by inline asm rather than stored somewhere, because a
// store to a __device__ global lowers to an s_getpc_b64 plus a relocated address
// add, and build_probe_callable() rejects a body carrying any relocation. The
// asm keeps the body self-contained without constraining where the compiler
// chooses to put the argument, which is the thing under test.
//
// Compiled for gfx90a this lowers to `s_waitcnt 0 ; v_mov_b32 v3, v0 ;
// s_setpc_b64 s[30:31]`.

extern "C" __device__ __attribute__((noinline, used, visibility("default"))) void
rj_arg_probe(int a) {
  asm volatile("v_mov_b32 v3, %0" ::"v"(a));
}

__device__ void (*rj_arg_probe_keepalive)(int) = rj_arg_probe;
