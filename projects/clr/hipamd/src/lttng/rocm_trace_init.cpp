/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * One-shot LTTng-UST initialization for the HIP runtime, plus the
 * dlmopen(LM_ID_NEWLM) SEGFAULT mitigation discovered during Phase 0.
 */
#if defined(HIP_ENABLE_LTTNG_UST) && HIP_ENABLE_LTTNG_UST

#include <atomic>
#include <cstdlib>
#include <link.h>     /* Lmid_t, LM_ID_BASE */
#include <dlfcn.h>    /* dlopen, dlinfo, RTLD_DI_LMID */
#include "rocm_hip_tp.h"

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
 *   never proceeds.
 */
extern "C" __attribute__((constructor(101))) void __rocm_hip_tp_ctor(void) {
    /* Honor explicit disable. */
    const char* dis = getenv("ROCM_LTTNG_UST_DISABLE");
    if (dis && dis[0] == '1') {
        g_initialized.store(true);  /* permanently disabled */
        return;
    }

#if defined(__GLIBC__)
    /* Detect non-default link namespace. dlopen(NULL, RTLD_NOLOAD) returns
     * a handle to the namespace the current load-unit is in; dlinfo with
     * RTLD_DI_LMID then reads its namespace id. */
    Lmid_t ns_id = LM_ID_BASE;
    void* self = dlopen(NULL, RTLD_LAZY | RTLD_NOLOAD);
    if (self != NULL) {
        if (dlinfo(self, RTLD_DI_LMID, &ns_id) == 0 && ns_id != LM_ID_BASE) {
            /* In a non-default namespace. Skip registration to avoid the
             * glibc dlmopen + LTTng-internal-dlopen SEGFAULT. */
            g_initialized.store(true);  /* permanently disabled */
            dlclose(self);
            return;
        }
        dlclose(self);
    }
#endif
    /* Otherwise fall through; __rocm_hip_tp_init will be called normally
     * from GetDispatchTableImpl. */
}

#else  /* !HIP_ENABLE_LTTNG_UST */

extern "C" void __rocm_hip_tp_init(void) { /* no-op when LTTng is disabled */ }

#endif
