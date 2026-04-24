/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

struct CmdOptions {
  bool no_display = false;
  bool progress = false;
  uint64_t reduce_iterations = 1;
  uint64_t reduce_input_size = 50;
};

extern CmdOptions cmd_options;
