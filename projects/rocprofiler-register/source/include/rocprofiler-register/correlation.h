// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Process-wide active-correlation-id TLS slot for ROCm runtime tracing.
//
// Both the HIP runtime (libamdhip64) and the HSA runtime (libhsa-runtime64)
// already link librocprofiler-register. Defining the slot here gives both
// runtimes a single shared definition, which enables cross-runtime same-thread
// correlation propagation: when HIP enters an instrumented API and pushes its
// corr_id onto this slot, HSA's downstream tracepoints fired on the same
// thread can read the slot and emit the HIP corr_id as their `parent_corr_id`.
//
// Per-runtime TU-local TLS (the previous approach) cannot achieve this because
// every translation unit gets its own slot.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single shared TLS slot. Default visibility so the dynamic linker resolves
// every load/store to the same per-thread storage no matter which .so the
// access is generated from. Value of 0 means "no active correlation context
// on this thread".
extern __thread uint64_t rocp_reg_active_corr_id
    __attribute__((visibility("default")));

#ifdef __cplusplus
}
#endif

// --- inline implementations (header-only, header is C-compatible) ---

#if defined(__linux__)
#  include <sys/syscall.h>
#  include <unistd.h>
#  if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 30))
static inline uint32_t rocp_reg_current_tid_(void) {
    return (uint32_t)gettid();
}
#  else
static inline uint32_t rocp_reg_current_tid_(void) {
    return (uint32_t)syscall(SYS_gettid);
}
#  endif
#else
static inline uint32_t rocp_reg_current_tid_(void) { return 0; }
#endif

// Generate a process-globally-unique 64-bit corr_id.
//   high 32 bits = tid (cached per-thread)
//   low  32 bits = per-thread monotonic counter
// Cost: 1-2 TLS loads + 1 increment. No syscalls (after first call), no atomics.
static inline uint64_t rocp_reg_next_corr_id(void) {
    static __thread uint32_t counter    = 0;
    static __thread uint32_t cached_tid = 0;
    if (cached_tid == 0) cached_tid = rocp_reg_current_tid_();
    return ((uint64_t)cached_tid << 32) | (uint64_t)(++counter);
}

// Push a new corr_id onto the per-thread "active" slot. Returns the previous
// value so the caller can restore it on exit. The explicit-prev "stack" is
// implemented by the caller saving the return value and restoring it at API
// exit; the slot itself only holds the topmost value. Used for code paths
// that have an explicit save site for the previous value.
static inline uint64_t rocp_reg_push_corr_id(uint64_t new_id) {
    uint64_t prev             = rocp_reg_active_corr_id;
    rocp_reg_active_corr_id   = new_id;
    return prev;
}

// Restore a previously-pushed corr_id.
static inline void rocp_reg_pop_corr_id(uint64_t prev) {
    rocp_reg_active_corr_id = prev;
}

// Read the currently-active corr_id without modifying it.
static inline uint64_t rocp_reg_active_corr_id_get(void) {
    return rocp_reg_active_corr_id;
}

// --- TLS auto-stack used by tracepoint emit_enter/emit_exit ---
//
// The HIP and HSA tracepoint emit helpers push the entering API's corr_id
// onto the slot at enter and pop it at exit, so downstream code on the same
// thread (e.g., HSA tracepoints fired from within a HIP API body) sees the
// caller's corr_id on the slot and can record it as `parent_corr_id`.
//
// Because the per-wrapper bodies do not have a natural place to save the
// previous corr_id between enter and exit, we use a small per-thread fixed
// stack here. Depth is in practice 1-3 (HIP API -> HSA API -> nested HSA API);
// 32 is comfortably more than enough. When the stack is full we still update
// the slot but do not record the prev (so pop will not restore correctly if
// the depth ever exceeds the cap; this is a deliberate trade-off vs unbounded
// stack growth and is documented).

#define ROCP_REG_AUTO_STACK_CAP 32

#ifdef __cplusplus
extern "C" {
#endif

// Both the stack array and the depth counter live in librocprofiler-register
// with default visibility, so all runtimes share a single per-thread stack
// regardless of which TU (HIP TU vs HSA TU) called push/pop.
extern __thread uint64_t rocp_reg_auto_stack_[ROCP_REG_AUTO_STACK_CAP]
    __attribute__((visibility("default")));
extern __thread int rocp_reg_auto_depth_
    __attribute__((visibility("default")));

#ifdef __cplusplus
}
#endif

// Emit-enter helper: save current slot value and push the new corr_id.
// Returns the value that the slot held before the push, so callers that
// want to record `parent_corr_id` on the entering tracepoint can use the
// return value. Always safe to call.
static inline uint64_t rocp_reg_auto_push(uint64_t new_id) {
    uint64_t prev = rocp_reg_active_corr_id;
    if (rocp_reg_auto_depth_ < ROCP_REG_AUTO_STACK_CAP) {
        rocp_reg_auto_stack_[rocp_reg_auto_depth_] = prev;
    }
    rocp_reg_auto_depth_++;
    rocp_reg_active_corr_id = new_id;
    return prev;
}

// Emit-exit helper: pop the saved previous value back onto the slot.
// On overflow (depth was beyond the stack cap when we pushed) the slot
// is left holding `new_id` which is wrong for nested cases, but this
// only manifests beyond depth 32 nesting on a single thread.
static inline void rocp_reg_auto_pop(void) {
    if (rocp_reg_auto_depth_ > 0) {
        rocp_reg_auto_depth_--;
        if (rocp_reg_auto_depth_ < ROCP_REG_AUTO_STACK_CAP) {
            rocp_reg_active_corr_id = rocp_reg_auto_stack_[rocp_reg_auto_depth_];
        }
    }
}
