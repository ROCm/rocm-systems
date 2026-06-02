// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Sizes and index layout for mega_kernel global atomic scratch buffers.
// main.cpp allocates and initializes; test_atomic_operations_asm / test_atomic_*
// in mega_kernel.hip must stay within these bounds.

#pragma once

// Element counts (device buffers sized with these in main.cpp).
#define MEGA_KERNEL_GLOBAL_INT_ELEMENTS    256
#define MEGA_KERNEL_GLOBAL_FLOAT_ELEMENTS  256
#define MEGA_KERNEL_GLOBAL_DOUBLE_ELEMENTS 64

// test_atomic_operations_asm (tid = global linear thread index):
//   global_int_ptr:  [tid % 64]           — TEST 1 add / gfx12 asm add
//                    [(tid % 64) + 64]    — TEST 1 max / gfx12 asm max
//                    [(tid % 64) + 128]   — TEST 4 flat int add
//                    [(tid % 64) + 192]   — TEST 10 CAS (expects initial 0)
//   Max int index: 63 + 192 = 255  →  need >= 256 int elements.
//
//   global_f32_ptr:  [tid % 64]           — TEST 2 FP32 add
//                    [(tid % 64) + 64]    — TEST 8 packed FP16
//                    [(tid % 64) + 128]   — TEST 9 packed BF16
//   Max float index: 63 + 128 = 191  →  need >= 192 float elements (256 used).
//
//   global_f64_ptr:  [tid % 32], [(tid % 32) + 32]  — TEST 3 (CDNA only)
//   Max double index: 31 + 32 = 63  →  need >= 64 double elements.
//
// Earlier in the same kernel launch (same buffers):
//   test_atomic_add_f32: global_float[0]
//   test_atomic_add_f64: global_double[0]
//   test_atomic_int: global_int[0..6]
