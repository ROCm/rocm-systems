/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * TraceLogging-backend lifecycle for the HIP CLR runtime.
 *
 * Backend swap of lttng/rocm_trace_init.cpp. The TraceLogging provider
 * (rocm_hip_tlg, defined in tracelogging/rocm_trace_emit_curated.cpp) must be
 * registered explicitly -- TraceLoggingRegister is not automatic.
 *
 * HIP init is effectively one-shot per process (the runtime's own init uses
 * std::call_once). The provider is therefore registered once from the library
 * constructor and unregistered once from the library destructor. The
 * refcounted register/unregister wrappers (see the generated .cpp) make this
 * exactly one register and one unregister; the refcount never exceeds 1 for
 * HIP. __rocm_hip_tp_init() is retained as an idempotent explicit "ready"
 * marker (register is idempotent under the refcount), matching the classic
 * backend's entry point.
 *
 * Defines `rocm_hip_trace_g_disabled` -- the runtime-wide kill switch. Per-DSO
 * name so ELF interposition cannot cross-bind the HSA/HIP flags.
 */
#if defined(HIP_ENABLE_LTTNG_UST) && HIP_ENABLE_LTTNG_UST

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <atomic>
#include <cstdlib>
#include <link.h>
#include <dlfcn.h>

std::atomic<bool> rocm_hip_trace_g_disabled
    __attribute__((visibility("default"))) {false};

extern "C" void rocm_hip_tlg_register(void);
extern "C" void rocm_hip_tlg_unregister(void);

namespace {
std::atomic<bool> g_permanently_disabled{false};
std::atomic<bool> g_registered{false};
}  // namespace

extern "C" void __rocm_hip_tp_init(void) {
    if (g_permanently_disabled.load(std::memory_order_relaxed)) return;
    /* Idempotent: register only once even if called repeatedly. */
    bool expected = false;
    if (g_registered.compare_exchange_strong(expected, true)) {
        rocm_hip_tlg_register();
    }
}

/* Library constructor: honor ROCM_LTTNG_UST_DISABLE, apply the
 * dlmopen(LM_ID_NEWLM) mitigation, then register the provider (HIP's one-shot
 * init model means the constructor is the natural single registration point). */
extern "C" __attribute__((constructor(101))) void __rocm_hip_tp_ctor(void) {
    const char* dis = getenv("ROCM_LTTNG_UST_DISABLE");
    if (dis && dis[0] == '1') {
        rocm_hip_trace_g_disabled.store(true, std::memory_order_relaxed);
        g_permanently_disabled.store(true, std::memory_order_relaxed);
        return;
    }

#if defined(__GLIBC__)
    Lmid_t ns_id = LM_ID_BASE;
    Dl_info info;
    void*   lm_handle = NULL;
    if (dladdr1(reinterpret_cast<void*>(&__rocm_hip_tp_ctor),
                &info, &lm_handle, RTLD_DL_LINKMAP) != 0
        && lm_handle != NULL
        && dlinfo(lm_handle, RTLD_DI_LMID, &ns_id) == 0
        && ns_id != LM_ID_BASE) {
        rocm_hip_trace_g_disabled.store(true, std::memory_order_relaxed);
        g_permanently_disabled.store(true, std::memory_order_relaxed);
        return;
    }
#endif

    __rocm_hip_tp_init();
}

/* Library destructor: unregister at DSO unload. Symmetric with the constructor
 * registration; safe if never registered (refcount stays 0, unregister is a
 * no-op). */
extern "C" __attribute__((destructor(101))) void __rocm_hip_tp_dtor(void) {
    if (g_registered.exchange(false)) {
        rocm_hip_tlg_unregister();
    }
}

#else

extern "C" void __rocm_hip_tp_init(void) { /* no-op when LTTng is disabled */ }

#endif
