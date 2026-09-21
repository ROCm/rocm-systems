// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Minimal no-op device probe fixture for testing/debugging DBI.
//
// Currently consumed by tests/dbi/probe_fixture_test.cpp.
//
// This is a __device__ function, not a __global__ kernel. See
// rj_add_probe_object in cmake/rj_add_device_kernel.cmake for how to build it.
//
// The attributes keep the symbol alive and callable:
//   extern "C"             - unmangled "rj_nop_probe" symbol name
//   noinline               - keep a real, callable function body
//   used                   - do not eliminate the unreferenced function
//   visibility("default")  - keep the symbol exported
//
// Compiled for gfx90a, this lowers to `s_waitcnt 0` then
// `s_setpc_b64 s[30:31]` (a return through the s[30:31] link pair).
//
extern "C" __device__ __attribute__((noinline, used, visibility("default"))) void rj_nop_probe() {}

// Keeps rj_nop_probe in the emitted code object. A device-only compile of a
// translation unit with no kernel and no device global drops the function
// entirely: `used` survives IR-level DCE, but Clang's HIP device-internalize
// pass then marks the hidden symbol internal and the device linker discards it.
// Taking its address from a device global is what forces it out of line and
// keeps it in .text. -fgpu-rdc would also do this, but it is incompatible with
// the unbundle step -- see rj_add_probe_object in
// cmake/rj_add_device_kernel.cmake.
//
// The pointer is never read; only the probe's own body is copied out of the
// object by the instrumentor.
__device__ void (*rj_nop_probe_keepalive)() = rj_nop_probe;
