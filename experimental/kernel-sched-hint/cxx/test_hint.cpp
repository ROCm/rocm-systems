/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "kernel_sched_hint.hpp"

#include <iostream>

// Fixed saxpy-shaped work. tests/test_hint.py compares these lines to Python.
int main() {
  kernel_sched_hint::Work work;
  work.valu_f32_flops = 128;
  work.global_bytes = 768;
  work.mem_ops = 3;
  work.waitcnts = 2;
  work.valu_issues = 2;
  work.vgprs = 4;
  work.sgprs = 8;
  work.wavefront_size = 64;
  const auto pred = kernel_sched_hint::predict(work, kernel_sched_hint::mi300x(), 4194304.0, 1.0);
  std::cout.precision(17);
  std::cout << pred.bound << "\n"
            << pred.resource << "\n"
            << pred.roofline_s << "\n"
            << pred.duration_s << "\n";
  return 0;
}
