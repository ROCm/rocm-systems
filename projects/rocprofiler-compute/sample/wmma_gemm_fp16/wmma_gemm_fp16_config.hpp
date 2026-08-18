// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace wmma_gemm_fp16 {

constexpr uint32_t kWmmaM = 16;
constexpr uint32_t kWmmaN = 16;
constexpr uint32_t kWmmaK = 32;

constexpr uint32_t kWaveSize = 32;
constexpr uint32_t kBlockM = 16;
constexpr uint32_t kBlockN = 16;
constexpr uint32_t kBlockK = 32;

constexpr uint32_t kThreadsX = 1 * kWaveSize;
constexpr uint32_t kThreadsY = 1;
constexpr uint32_t kFlatWorkgroupSize = kThreadsX * kThreadsY;

constexpr uint32_t kDefaultM = 4096;
constexpr uint32_t kDefaultN = 4096;
constexpr uint32_t kDefaultK = 1024;
constexpr uint32_t kDefaultIters = 50;
constexpr uint32_t kDefaultWarmup = 5;

constexpr int DivUp(int a, int b) { return (a + b - 1) / b; }

}  // namespace wmma_gemm_fp16
