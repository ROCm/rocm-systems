/* Copyright (c) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT */

/* Compiled as strict C99: values.h must stay usable from C (HRX). */

#include "values_c.h"

#include <rocm/trace/values.h>

uint64_t rocm_trace_test_c_keywords(void) {
  return ROCM_TRACE_KW_API | ROCM_TRACE_KW_DISPATCH | ROCM_TRACE_KW_STATE |
         ROCM_TRACE_KW_SERIALIZE | ROCM_TRACE_KW_COPY_TIME |
         ROCM_TRACE_KW_CODE_BYTES | ROCM_TRACE_KW_ALLOC;
}

const char *rocm_trace_test_c_hip_name(void) {
  return ROCM_TRACE_PROVIDER_HIP_NAME;
}

const char *rocm_trace_test_c_hsa_guid(void) {
  return ROCM_TRACE_PROVIDER_HSA_GUID;
}

int rocm_trace_test_c_values_version(void) { return ROCM_TRACE_VALUES_VERSION; }

int rocm_trace_test_c_capture_full(void) { return ROCM_TRACE_CAPTURE_FULL; }

int rocm_trace_test_c_string_cap(void) { return ROCM_TRACE_STRING_CAP; }
