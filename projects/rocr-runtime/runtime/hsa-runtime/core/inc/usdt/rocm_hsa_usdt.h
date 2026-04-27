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
 * BPF programs read it with bpf_probe_read_user_str() (or str(arg0) in bpftrace).
 *
 * Cost model:
 *   Disabled-build:                 zero (macros expand to (void)0).
 *   Enabled-build, unattached:      one nop per probe site.
 *   Enabled-build, attached:        ~1-3 us per fire (kernel uprobe trap + BPF program).
 *
 * IMPORTANT: probe macros must be invoked from a context where their argument
 * expressions are simple value reads gcc can size-classify (function locals,
 * function parameters, register-resident temporaries). C++ struct member
 * accesses inside member methods trip a sys/sdt.h codegen path that drops the
 * argument size prefix in .note.stapsdt, which BPF tools cannot parse. The
 * helpers in rocm::hsa::usdt accept their arguments by value/parameter, which
 * keeps the operands size-classifiable; do not move the DTRACE_PROBE call
 * into a member method that reads `this->name` / `this->corr` instead.
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
        DTRACE_PROBE2(hsa, api_entry, (name), (corr))
#  define ROCM_HSA_USDT_EXIT(name, corr, val) \
        DTRACE_PROBE3(hsa, api_exit, (name), (corr), (val))
#else
#  define ROCM_HSA_USDT_ENTRY(name, corr) ((void)0)
#  define ROCM_HSA_USDT_EXIT(name, corr, val) ((void)0)
#endif

namespace rocm { namespace hsa { namespace usdt {

// Process-global monotonic correlation id. Reserves 0 as the "no correlation"
// sentinel, so the first id handed out is 1.
inline uint64_t generate_correlation_id() noexcept {
    static std::atomic<uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

// Convert an arbitrary integral / enum / pointer value to a uint64_t bit
// pattern for transport through the USDT exit probe slot. Signed integer
// returns (e.g. hsa_signal_value_t = int64_t) are bit-preserved, not
// sign-extended into a wider field — BPF consumers must reinterpret as the
// original signed type when needed.
template <typename T>
inline uint64_t to_u64(T v) noexcept {
    static_assert(std::is_integral<T>::value || std::is_enum<T>::value ||
                      std::is_pointer<T>::value,
                  "USDT exit value must be integral, enum, or pointer; "
                  "use fire_exit_void for void returns");
    if constexpr (std::is_pointer<T>::value) {
        return reinterpret_cast<uintptr_t>(v);
    } else {
        return static_cast<uint64_t>(v);
    }
}

// Free functions instead of an RAII helper because gcc / sys/sdt.h together
// drop the argument size prefix from .note.stapsdt when the DTRACE_PROBE
// invocation reads a C++ member access from inside an inlined member method
// (verified on systemtap-sdt-dev 4.6 with g++ 11). Keeping the probe call in
// a free function whose operands are function-local locals / parameters
// preserves the size prefix.

inline uint64_t fire_entry(const char* name) noexcept {
    uint64_t corr = generate_correlation_id();
    ROCM_HSA_USDT_ENTRY(name, corr);
    return corr;
}

template <typename T>
inline T fire_exit(const char* name, uint64_t corr, T result) noexcept {
    uint64_t val = to_u64(result);
    ROCM_HSA_USDT_EXIT(name, corr, val);
    return result;
}

inline void fire_exit_void(const char* name, uint64_t corr) noexcept {
    ROCM_HSA_USDT_EXIT(name, corr, static_cast<uint64_t>(0));
}

// Wrap a function body lambda. Fires entry probe, runs the body, fires exit
// probe with the returned value, returns the value to the caller. This is the
// shape the redefined TRY/CATCH macros use:
//
//   #define TRY  return ::rocm::hsa::usdt::run_traced(__func__,            \
//                    [&]() -> hsa_status_t { try {
//   #define CATCH  } catch(...) { return AMD::handleException(); } });
template <typename F>
inline auto run_traced(const char* name, F&& fn)
    -> typename std::enable_if<!std::is_void<decltype(fn())>::value,
                               decltype(fn())>::type {
    uint64_t corr = fire_entry(name);
    auto result = std::forward<F>(fn)();
    return fire_exit(name, corr, result);
}

// Void-return overload. Use for HSA APIs that return void.
template <typename F>
inline auto run_traced_void(const char* name, F&& fn)
    -> typename std::enable_if<std::is_void<decltype(fn())>::value, void>::type {
    uint64_t corr = fire_entry(name);
    std::forward<F>(fn)();
    fire_exit_void(name, corr);
}

}}}  // namespace rocm::hsa::usdt

#endif  // HSA_CORE_USDT_ROCM_HSA_USDT_H
