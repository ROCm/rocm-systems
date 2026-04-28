/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Thin wrappers over librocprofiler-register's correlation API. The TLS
 * corr-id slot lives in librocprofiler-register/correlation.h so HIP and
 * HSA share it via the dynamic linker -- when HIP pushes its corr_id on
 * entry, HSA tracepoints fired on the same thread emit it as
 * `parent_corr_id`.
 *
 * On platforms where <rocprofiler-register/correlation.h> isn't on the
 * include path (currently: Windows builds via TheRock), the wrappers
 * fall back to no-op stubs and correlation tracking is disabled. The
 * caller-side macros (in rocm_trace_emit.h) that invoke these wrappers
 * are themselves guarded by *_ENABLE_LTTNG_UST so on a no-LTTng build
 * the no-op stubs are never even called.
 */
#ifndef ROCM_TRACE_TID_H_
#define ROCM_TRACE_TID_H_

#include <stdint.h>

#if defined(__has_include)
#  if __has_include(<rocprofiler-register/correlation.h>)
#    include <rocprofiler-register/correlation.h>
#    define ROCM_TRACE_HAS_ROCPROFILER_REGISTER 1
#  endif
#endif

#ifdef ROCM_TRACE_HAS_ROCPROFILER_REGISTER

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

#else  /* no rocprofiler-register: correlation tracking disabled */

static inline uint32_t rocm_trace_current_tid(void)            { return 0; }
static inline uint64_t rocm_trace_next_corr_id(void)           { return 0; }
static inline uint64_t rocm_trace_push_corr_id(uint64_t)       { return 0; }
static inline void     rocm_trace_pop_corr_id(uint64_t)        { }
static inline uint64_t rocm_trace_active_corr_id(void)         { return 0; }

#endif /* ROCM_TRACE_HAS_ROCPROFILER_REGISTER */

#endif /* ROCM_TRACE_TID_H_ */
