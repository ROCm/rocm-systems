/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Thin wrappers over librocprofiler-register's correlation API. The TLS
 * corr-id slot lives in librocprofiler-register/correlation.h so HIP and
 * HSA share it via the dynamic linker -- when HIP pushes its corr_id on
 * entry, HSA tracepoints fired on the same thread emit it as
 * `parent_corr_id`.
 */
#ifndef ROCM_TRACE_TID_H_
#define ROCM_TRACE_TID_H_

#include <stdint.h>
#include <rocprofiler-register/correlation.h>

static inline uint32_t rocm_trace_current_tid(void) {
    return rocp_reg_current_tid_();
}

static inline uint64_t rocm_trace_next_corr_id(void) {
    return rocp_reg_next_corr_id();
}

static inline uint64_t rocm_trace_push_corr_id(uint64_t v) {
    return rocp_reg_push_corr_id(v);
}

static inline void rocm_trace_pop_corr_id(uint64_t prev) {
    rocp_reg_pop_corr_id(prev);
}

static inline uint64_t rocm_trace_active_corr_id(void) {
    return rocp_reg_active_corr_id_get();
}

#endif /* ROCM_TRACE_TID_H_ */
