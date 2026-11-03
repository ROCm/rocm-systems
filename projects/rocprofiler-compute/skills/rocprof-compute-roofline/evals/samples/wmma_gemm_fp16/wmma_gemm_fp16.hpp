// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "wmma_gemm_fp16_config.hpp"

// FP16 WMMA GEMM for gfx1250. Kernel adapted from hipNPIKernels
// WMMA_gemm/src/fp16/gfx1250/fp16_16x16x32/native16x16x32.cpp.
extern "C" __global__ void gemm_wmma_fp16(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          const _Float16* __restrict__ a,
                                          const _Float16* __restrict__ b,
                                          _Float16* __restrict__ d);
