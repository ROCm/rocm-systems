// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/rj_waitcheck.h"

int main(void) {
  rj_waitcheck_options_t options;
  rj_waitcheck_result_t result;
  rj_waitcheck_options_init(&options);
  if (rj_waitcheck_analyze(NULL, 0, &options, &result) != ROCJITSU_STATUS_INVALID_ARGUMENT)
    return 1;
  if (rj_waitcheck_analyze_kernel(NULL, 0, 0, &options, &result) !=
      ROCJITSU_STATUS_INVALID_ARGUMENT)
    return 2;
  return 0;
}
