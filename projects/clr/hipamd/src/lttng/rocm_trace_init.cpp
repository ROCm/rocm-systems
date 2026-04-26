/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * One-shot LTTng-UST initialization for the HIP runtime, plus the
 * dlmopen(LM_ID_NEWLM) SEGFAULT mitigation discovered during Phase 0.
 *
 * Defines `rocm_hip_trace_g_disabled` -- the runtime-wide kill switch that
 * the emit helpers in rocm_trace_emit.h short-circuit on. The library
 * constructor sets it to `true` when:
 *   - getenv("ROCM_LTTNG_UST_DISABLE") returns "1", OR
 *   - dladdr1(&__rocm_hip_tp_ctor, RTLD_DL_LINKMAP) reports a non-LM_ID_BASE
 *     namespace for THIS DSO (the dlmopen-into-non-default-namespace
 *     SEGFAULT mitigation).
 *
 * The symbol is named `rocm_hip_trace_g_disabled` (not `rocm_trace_g_disabled`)
 * so that ELF symbol interposition cannot bind references to HSA's
 * default-visible `rocm_hsa_trace_g_disabled`. Both DSOs would otherwise
 * collide on a shared name and one runtime's flag could disable the other.
 *
 * Once set the flag is never cleared. Each emit helper does a single
 * relaxed atomic load on the hot path; the cost is ~1 ns and dominated
 * by the existing tracepoint_enabled() check.
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

/* Visible to the emit helpers in rocm_trace_emit.h via an extern declaration.
 * Defined with default visibility so a single definition lives in the
 * libamdhip64.so PLT/GOT and every TU links to the same flag.
 *
 * Per-DSO name (`rocm_hip_...` vs HSA's `rocm_hsa_...`) prevents ELF
 * symbol interposition from binding HIP's references to HSA's flag (or
 * vice versa) when both libraries are loaded into the same process.
 *
 * Note: NOT `extern "C"` — std::atomic<bool> is a C++ type, and combining
 * `extern` with an initializer is flagged by GCC -Werror=extern-initialized.
 * The header declares it with matching C++ linkage. The symbol is name-mangled
 * but unique within the .so (only one TU defines it). */
std::atomic<bool> rocm_hip_trace_g_disabled
    __attribute__((visibility("default"))) {false};

namespace {
/* Tri-state:
 *   false initially -> __rocm_hip_tp_init has not run
 *   true after init  -> registered (or permanently disabled, see ctor below)
 */
std::atomic<bool> g_initialized{false};
}  // namespace

extern "C" void __rocm_hip_tp_init(void) {
    /* Recursion-safe: the LTTng tracepoint provider object's static-init has
     * already registered the provider with the daemon by the time we land
     * here (the provider is auto-loaded via __attribute__((constructor)) in
     * lttng-ust's TRACEPOINT_DEFINE expansion). This function is the
     * explicit "we are ready" marker the runtime calls from inside its own
     * init, so callers don't have to think about static-init ordering.
     * Idempotent. */
    bool expected = false;
    if (!g_initialized.compare_exchange_strong(expected, true)) return;

    /* No-op for buffer config: liblttng-ust 2.13's per-application
     * channel-attr API is not stable. The committed channel mode and
     * discard policy from the architecture spec are enforced session-side
     * via `lttng enable-channel --discard --subbuf-size=N --num-subbuf=M`. */
}

/* Library constructor - runs at dlopen time, BEFORE any user code in the
 * runtime is reached. This is where we honor the disable knobs, including
 * the link-namespace SEGFAULT mitigation discovered by Phase 0:
 *
 *   When the runtime is loaded via dlmopen(LM_ID_NEWLM, ..., RTLD_NOW)
 *   into a non-default link namespace (HPCToolkit / Score-P style isolation),
 *   the LTTng tracepoint provider's static-init internally calls
 *   dlopen("liblttng-ust-tracepoint.so.1") which triggers a glibc bug in
 *   add_to_global_resize() and SEGFAULTs the process.
 *
 *   The fix: detect we are NOT in the base namespace (LM_ID_BASE) and
 *   no-op forever. The LTTng provider's static-init may have already run
 *   (its constructor priority is unspecified vs. ours), but we set
 *   g_initialized to a "permanently disabled" marker so __rocm_hip_tp_init
 *   never proceeds, AND we set rocm_hip_trace_g_disabled so individual emit
 *   helpers short-circuit even if a tracepoint was inadvertently enabled
 *   by the LTTng constructor before we got here.
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
    /* Detect non-default link namespace.
     *
     * IMPORTANT: do NOT use `dlopen(NULL, RTLD_NOLOAD)` + `dlinfo()` here.
     * Per dlopen(3): "If filename is NULL, then the returned handle is for
     * the main program." That handle reflects the MAIN PROGRAM's namespace,
     * not the namespace this DSO was loaded into via dlmopen. In the very
     * dlmopen-into-non-default-namespace scenario this mitigation targets,
     * dlinfo(RTLD_DI_LMID) on the main-program handle silently returns
     * LM_ID_BASE and bypasses the mitigation. (Verified empirically: in a
     * dlmopen(LM_ID_NEWLM, ...) DSO the old pattern reports ns_id=0 while
     * the correct namespace is 1.)
     *
     * Use `dladdr1(<symbol-in-this-DSO>, ..., RTLD_DL_LINKMAP)` to obtain
     * the `struct link_map*` for THIS DSO (the one containing
     * __rocm_hip_tp_ctor), then bridge to `dlinfo(handle, RTLD_DI_LMID, ...)`
     * to read its namespace id. The link_map handle is exactly what dlopen
     * returns under the hood, so dlinfo accepts it directly. We avoid
     * touching `lm->l_lmid` directly because that field is not exposed in
     * the public glibc <link.h>. */
    Lmid_t ns_id = LM_ID_BASE;
    Dl_info info;
    void*   lm_handle = NULL;
    if (dladdr1(reinterpret_cast<void*>(&__rocm_hip_tp_ctor),
                &info, &lm_handle, RTLD_DL_LINKMAP) != 0
        && lm_handle != NULL
        && dlinfo(lm_handle, RTLD_DI_LMID, &ns_id) == 0
        && ns_id != LM_ID_BASE) {
        /* In a non-default namespace. Skip registration to avoid the
         * glibc dlmopen + LTTng-internal-dlopen SEGFAULT. */
        rocm_hip_trace_g_disabled.store(true, std::memory_order_relaxed);
        g_initialized.store(true);  /* permanently disabled */
        return;
    }
#endif
    /* Otherwise fall through; __rocm_hip_tp_init will be called normally
     * from GetDispatchTableImpl. */
}

#else  /* !HIP_ENABLE_LTTNG_UST */

extern "C" void __rocm_hip_tp_init(void) { /* no-op when LTTng is disabled */ }

#endif
