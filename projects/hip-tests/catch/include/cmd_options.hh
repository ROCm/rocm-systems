/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <limits>

struct CmdOptions {
  int iterations = 5;
  int warmups = 5;
  int cg_iterations = 1;
  double cg_reduction_factor = 6.25;
  double warp_reduction_factor = 6.25;
  bool no_display = false;
  bool progress = false;
  uint64_t accuracy_iterations = std::numeric_limits<uint32_t>::max() + 1ull;
  uint64_t reduce_iterations = 1;
  uint64_t reduce_input_size = 50;
  int accuracy_max_memory = 80;
  uint64_t max_memory = 2147483648; // 2 GB
  double reduction_factor = 0.1;
};

extern CmdOptions cmd_options;
