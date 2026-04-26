// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Single shared definition of the ROCm active-correlation-id TLS slot.
// See source/include/rocprofiler-register/correlation.h for rationale.

#include <rocprofiler-register/correlation.h>

extern "C" {
// Default visibility so HIP and HSA (and any future runtime) all resolve
// load/store of this symbol to the same per-thread storage via the dynamic
// linker. The `extern __thread` declaration in correlation.h provides the
// matching declaration; this TU is the unique definition.
__attribute__((visibility("default"))) __thread uint64_t rocp_reg_active_corr_id = 0;

// Shared per-thread auto-stack of saved previous corr_id values, used by
// the emit_enter / emit_exit auto push/pop helpers. See correlation.h.
__attribute__((visibility("default"))) __thread uint64_t
    rocp_reg_auto_stack_[ROCP_REG_AUTO_STACK_CAP] = {0};
__attribute__((visibility("default"))) __thread int rocp_reg_auto_depth_ = 0;
}
