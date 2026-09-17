/* TraceLogging-backend lifecycle for the HSA runtime.
 *
 * Backend swap of lttng/rocm_trace_init.cpp: instead of relying on the classic
 * LTTNG_UST provider's constructor-based static-init, the TraceLogging provider
 * (rocm_hsa_tlg, defined in tracelogging/rocm_trace_emit_curated.cpp) is
 * registered/unregistered explicitly. TraceLoggingRegister is NOT automatic.
 *
 * Defines `rocm_hsa_trace_g_disabled` -- the runtime-wide kill switch the emit
 * helpers short-circuit on. Per-DSO name so ELF interposition cannot cross-bind
 * the HSA/HIP flags when both runtimes are loaded.
 *
 * Registration lifecycle for HSA's repeatable hsa_init()/hsa_shut_down():
 *   - __rocm_hsa_tp_init() is called from Runtime::LoadTools() (first Acquire,
 *     ref_count 0->1) and calls rocm_hsa_tlg_register().
 *   - __rocm_hsa_tp_fini() is called from Runtime::UnloadTools() (last Release,
 *     ref_count 1->0) and calls rocm_hsa_tlg_unregister().
 * The register/unregister wrappers are refcounted + mutex-guarded (see the
 * generated .cpp), so repeated init/shutdown cycles register once per 0->1 and
 * unregister once per 1->0, never overlapping and never double-registering. If
 * HSA init fails partway after LoadTools() ran, Unload()/UnloadTools() still
 * runs on the error path and pairs the register with an unregister, so the
 * provider is never left half-registered.
 */
#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <atomic>
#include <cstdlib>
#include <link.h>
#include <dlfcn.h>

/* Kill switch. Referenced (extern) by the generated emit .cpp. */
std::atomic<bool> rocm_hsa_trace_g_disabled
    __attribute__((visibility("default"))) {false};

/* Provider register/unregister, defined in the generated emit .cpp (must live
 * in the same TU as the TraceLoggingWrite calls -- the provider symbol is a
 * linker-section token). */
extern "C" void rocm_hsa_tlg_register(void);
extern "C" void rocm_hsa_tlg_unregister(void);

namespace {
/* Set once when emission is permanently disabled (explicit env disable or a
 * non-default link namespace). Prevents registering the provider at all. */
std::atomic<bool> g_permanently_disabled{false};
}  // namespace

extern "C" void __rocm_hsa_tp_init(void) {
    if (g_permanently_disabled.load(std::memory_order_relaxed)) return;
    rocm_hsa_tlg_register();
}

extern "C" void __rocm_hsa_tp_fini(void) {
    if (g_permanently_disabled.load(std::memory_order_relaxed)) return;
    rocm_hsa_tlg_unregister();
}

/* Library constructor: honor ROCM_LTTNG_UST_DISABLE and skip registration in a
 * non-default link namespace (dlmopen(LM_ID_NEWLM) mitigation, same as the
 * classic backend). Runs before any runtime code, so __rocm_hsa_tp_init() sees
 * g_permanently_disabled already set. */
extern "C" __attribute__((constructor(101))) void __rocm_hsa_tp_ctor(void) {
    const char* dis = getenv("ROCM_LTTNG_UST_DISABLE");
    if (dis && dis[0] == '1') {
        rocm_hsa_trace_g_disabled.store(true, std::memory_order_relaxed);
        g_permanently_disabled.store(true, std::memory_order_relaxed);
        return;
    }

#if defined(__GLIBC__)
    Lmid_t ns_id = LM_ID_BASE;
    Dl_info info;
    void*   lm_handle = NULL;
    if (dladdr1(reinterpret_cast<void*>(&__rocm_hsa_tp_ctor),
                &info, &lm_handle, RTLD_DL_LINKMAP) != 0
        && lm_handle != NULL
        && dlinfo(lm_handle, RTLD_DI_LMID, &ns_id) == 0
        && ns_id != LM_ID_BASE) {
        rocm_hsa_trace_g_disabled.store(true, std::memory_order_relaxed);
        g_permanently_disabled.store(true, std::memory_order_relaxed);
        return;
    }
#endif
}

#else

extern "C" void __rocm_hsa_tp_init(void) { /* no-op when LTTng is disabled */ }
extern "C" void __rocm_hsa_tp_fini(void) { /* no-op when LTTng is disabled */ }

#endif
