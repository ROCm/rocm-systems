/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 *
 * USDT probe shims for HSA runtime.
 *
 * Provider: hsa
 * Probes:
 *   hsa:api_entry(const char* func_name, uint64_t correlation_id)
 *   hsa:api_exit(const char* func_name, uint64_t correlation_id, uint64_t status_or_value)
 *
 * Function name is passed as a pointer to the .rodata literal produced by __func__.
 * BPF programs read it with bpf_probe_read_user_str() on demand.
 *
 * For hsa_status_t-returning APIs, status_or_value is the hsa_status_t cast to
 * uint64_t. For other return types it is the raw return value (handled via 64-bit
 * truncation; pointer/uint64/uint32 returns work; void returns are emitted from a
 * void-specialized guard with status_or_value == 0).
 */

#ifndef HSA_CORE_USDT_ROCM_HSA_USDT_H
#define HSA_CORE_USDT_ROCM_HSA_USDT_H

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <utility>

#if defined(ROCM_ENABLE_USDT) && ROCM_ENABLE_USDT
#  include <sys/sdt.h>
#  define ROCM_HSA_USDT_ENTRY(name, corr) \
        DTRACE_PROBE2(hsa, api_entry, (name), (uint64_t)(corr))
#  define ROCM_HSA_USDT_EXIT(name, corr, val) \
        DTRACE_PROBE3(hsa, api_exit, (name), (uint64_t)(corr), (uint64_t)(val))
#else
#  define ROCM_HSA_USDT_ENTRY(name, corr) ((void)0)
#  define ROCM_HSA_USDT_EXIT(name, corr, val) ((void)0)
#endif

namespace rocm { namespace hsa { namespace usdt {

inline uint64_t generate_correlation_id() noexcept {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

// EntryGuard fires the entry probe in its constructor. It is intended to be
// used with operator+ to wrap a lambda whose return value the guard captures
// and reports through the exit probe. Used by the redefined TRY/CATCH macros.
struct EntryGuard {
    const char* name;
    uint64_t corr;

    explicit EntryGuard(const char* n) noexcept
        : name(n), corr(generate_correlation_id()) {
        ROCM_HSA_USDT_ENTRY(name, corr);
    }

    // Wraps a callable (the function body lambda). Invokes it, captures the
    // return value, fires the exit probe, returns the value. Specialized below
    // for void return.
    template <typename F>
    auto operator+(F&& fn) -> typename std::enable_if<
        !std::is_void<decltype(fn())>::value,
        decltype(fn())>::type {
        auto result = std::forward<F>(fn)();
        ROCM_HSA_USDT_EXIT(name, corr, (uint64_t)result);
        return result;
    }
};

// Void-return specialization helper. Used by callers whose lambda returns void.
struct EntryGuardVoid {
    const char* name;
    uint64_t corr;

    explicit EntryGuardVoid(const char* n) noexcept
        : name(n), corr(generate_correlation_id()) {
        ROCM_HSA_USDT_ENTRY(name, corr);
    }

    template <typename F>
    void operator+(F&& fn) {
        std::forward<F>(fn)();
        ROCM_HSA_USDT_EXIT(name, corr, 0);
    }
};

}}}  // namespace rocm::hsa::usdt

#endif  // HSA_CORE_USDT_ROCM_HSA_USDT_H
