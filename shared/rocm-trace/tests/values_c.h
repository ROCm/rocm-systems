/* Copyright (c) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT */

/* Values seen by a C translation unit, compared with the C++ view in the tests.
 */

#ifndef ROCM_TRACE_TESTS_VALUES_C_H
#define ROCM_TRACE_TESTS_VALUES_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint64_t rocm_trace_test_c_keywords(void);
const char *rocm_trace_test_c_hip_name(void);
const char *rocm_trace_test_c_hsa_guid(void);
int rocm_trace_test_c_values_version(void);
int rocm_trace_test_c_capture_full(void);
int rocm_trace_test_c_string_cap(void);

#ifdef __cplusplus
}
#endif

#endif /* ROCM_TRACE_TESTS_VALUES_C_H */
