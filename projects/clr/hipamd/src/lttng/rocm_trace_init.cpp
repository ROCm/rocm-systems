/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * One-shot LTTng-UST initialization for the HIP runtime, plus the
 * dlmopen(LM_ID_NEWLM) SEGFAULT mitigation.
 *
 * Defines `rocm_hip_trace_g_disabled` -- the runtime-wide kill switch the
 * emit helpers in rocm_trace_emit.h short-circuit on. Per-DSO name
 * (`rocm_hip_...` vs HSA's `rocm_hsa_...`) so ELF symbol interposition
 * cannot cross-bind the two runtimes' flags when both libraries are loaded
 * into the same process.
 */
#if defined(HIP_ENABLE_LTTNG_UST) && HIP_ENABLE_LTTNG_UST

/* dladdr1 + RTLD_DL_LINKMAP are GNU extensions; _GNU_SOURCE makes them
 * visible from <dlfcn.h>. Defined before any system header is included. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <atomic>
#include <cstdlib>
#include <link.h>     /* Lmid_t, LM_ID_BASE */
#include <dlfcn.h>    /* dladdr1, RTLD_DL_LINKMAP, dlinfo, RTLD_DI_LMID */
#include "rocm_hip_tp.h"

std::atomic<bool> rocm_hip_trace_g_disabled
    __attribute__((visibility("default"))) {false};

namespace {
std::atomic<bool> g_initialized{false};
}  // namespace

extern "C" void __rocm_hip_tp_init(void) {
    /* Idempotent. The LTTng tracepoint provider has already been registered
     * with the daemon by static-init (TRACEPOINT_DEFINE expands a
     * __attribute__((constructor))); this function is the explicit
     * "we are ready" marker called from the runtime's own init. */
    bool expected = false;
    if (!g_initialized.compare_exchange_strong(expected, true)) return;

    /* Channel buffer config (mode, sub-buffer size, num sub-buffers) is
     * set session-side via `lttng enable-channel --discard ...`; the
     * per-application channel-attr API in liblttng-ust 2.13 is not stable. */
}

/* Library constructor — runs at dlopen time, before any user code in the
 * runtime is reached. Honors ROCM_LTTNG_UST_DISABLE and skips registration
 * when loaded into a non-default link namespace.
 *
 * dlmopen mitigation: when this DSO is loaded via dlmopen(LM_ID_NEWLM, ...)
 * (HPCToolkit / Score-P style isolation), the LTTng tracepoint provider's
 * static-init internally calls dlopen("liblttng-ust-tracepoint.so.1"),
 * which triggers a glibc bug in add_to_global_resize() and SEGFAULTs the
 * process. We detect non-LM_ID_BASE namespace and disable emission for the
 * lifetime of the process.
 */
extern "C" __attribute__((constructor(101))) void __rocm_hip_tp_ctor(void) {
    /* Honor explicit disable. */
    const char* dis = getenv("ROCM_LTTNG_UST_DISABLE");
    if (dis && dis[0] == '1') {
        rocm_hip_trace_g_disabled.store(true, std::memory_order_relaxed);
        g_initialized.store(true);  /* permanently disabled */
        return;
    }

#if defined(__GLIBC__)
    /* dladdr1 returns this DSO's link_map; dlopen(NULL) would return the
     * main program's, which is the wrong namespace. */
    Lmid_t ns_id = LM_ID_BASE;
    Dl_info info;
    void*   lm_handle = NULL;
    if (dladdr1(reinterpret_cast<void*>(&__rocm_hip_tp_ctor),
                &info, &lm_handle, RTLD_DL_LINKMAP) != 0
        && lm_handle != NULL
        && dlinfo(lm_handle, RTLD_DI_LMID, &ns_id) == 0
        && ns_id != LM_ID_BASE) {
        rocm_hip_trace_g_disabled.store(true, std::memory_order_relaxed);
        g_initialized.store(true);  /* permanently disabled */
        return;
    }
#endif
}

#else  /* !HIP_ENABLE_LTTNG_UST */

extern "C" void __rocm_hip_tp_init(void) { /* no-op when LTTng is disabled */ }

#endif
