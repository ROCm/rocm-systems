/* One-shot LTTng-UST initialization for the HSA runtime, plus the
 * dlmopen(LM_ID_NEWLM) SEGFAULT mitigation.
 *
 * Defines `rocm_hsa_trace_g_disabled` -- the runtime-wide kill switch the
 * emit helpers in rocm_trace_emit.h short-circuit on. Per-DSO name
 * (`rocm_hsa_...` vs HIP's `rocm_hip_...`) so ELF symbol interposition
 * cannot cross-bind the two runtimes' flags when both libraries are loaded
 * into the same process.
 */
#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST

/* dladdr1 + RTLD_DL_LINKMAP are GNU extensions; _GNU_SOURCE makes them
 * visible from <dlfcn.h>. Defined before any system header is included. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <atomic>
#include <cstdlib>
#include <link.h>
#include <dlfcn.h>
#include "rocm_hsa_tp.h"

std::atomic<bool> rocm_hsa_trace_g_disabled
    __attribute__((visibility("default"))) {false};

namespace {
std::atomic<bool> g_initialized{false};
}

extern "C" void __rocm_hsa_tp_init(void) {
    /* Idempotent. The LTTng tracepoint provider has already been registered
     * with the daemon by static-init; this function is the explicit
     * "we are ready" marker called from the runtime's own init. */
    bool expected = false;
    if (!g_initialized.compare_exchange_strong(expected, true)) return;

    /* Channel buffer config is set session-side via `lttng enable-channel`;
     * the per-application channel-attr API in liblttng-ust 2.13 is not
     * stable. */
}

/* Library constructor — runs at dlopen time, before any user code in the
 * runtime is reached. Honors ROCM_LTTNG_UST_DISABLE (the same env var
 * recognized by the HIP provider; one switch silences both) and skips
 * registration when loaded into a non-default link namespace.
 *
 * dlmopen mitigation: when this DSO is loaded via dlmopen(LM_ID_NEWLM, ...)
 * (HPCToolkit / Score-P style isolation), the LTTng tracepoint provider's
 * static-init internally calls dlopen("liblttng-ust-tracepoint.so.1"),
 * which triggers a glibc bug in add_to_global_resize() and SEGFAULTs the
 * process. We detect non-LM_ID_BASE namespace and disable emission for the
 * lifetime of the process.
 */
extern "C" __attribute__((constructor(101))) void __rocm_hsa_tp_ctor(void) {
    const char* dis = getenv("ROCM_LTTNG_UST_DISABLE");
    if (dis && dis[0] == '1') {
        rocm_hsa_trace_g_disabled.store(true, std::memory_order_relaxed);
        g_initialized.store(true);
        return;
    }

#if defined(__GLIBC__)
    /* dladdr1 returns this DSO's link_map; dlopen(NULL) would return the
     * main program's, which is the wrong namespace. */
    Lmid_t ns_id = LM_ID_BASE;
    Dl_info info;
    void*   lm_handle = NULL;
    if (dladdr1(reinterpret_cast<void*>(&__rocm_hsa_tp_ctor),
                &info, &lm_handle, RTLD_DL_LINKMAP) != 0
        && lm_handle != NULL
        && dlinfo(lm_handle, RTLD_DI_LMID, &ns_id) == 0
        && ns_id != LM_ID_BASE) {
        rocm_hsa_trace_g_disabled.store(true, std::memory_order_relaxed);
        g_initialized.store(true);
        return;
    }
#endif
}

#else

extern "C" void __rocm_hsa_tp_init(void) { /* no-op when LTTng is disabled */ }

#endif
