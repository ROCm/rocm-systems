// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/analysis/rj_waitcheck.h"

int main(void) {
  rj_waitcheck_options_t options;
  rj_waitcheck_result_t result;
  if (rj_waitcheck_options_init(&options, sizeof(options)) != ROCJITSU_STATUS_SUCCESS)
    return 1;
  if (rj_waitcheck_result_init(&result, sizeof(result)) != ROCJITSU_STATUS_SUCCESS)
    return 2;
  if (rj_waitcheck_analyze(NULL, 0, &options, &result) != ROCJITSU_STATUS_INVALID_ARGUMENT)
    return 3;
  if (rj_waitcheck_analyze_kernel(NULL, 0, 0, &options, &result) !=
      ROCJITSU_STATUS_INVALID_ARGUMENT)
    return 4;
  if (rj_waitcheck_target_name(ROCJITSU_WAITCHECK_TARGET_GFX1250) == NULL)
    return 5;
  if (rj_waitcheck_diagnostic_code_name(ROCJITSU_WAITCHECK_DIAGNOSTIC_WAIT_COUNTER) == NULL)
    return 6;
  return 0;
}
