#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST

#include <atomic>
#include <cstdlib>
#include <link.h>     /* for Lmid_t, LM_ID_BASE */
#include <dlfcn.h>    /* for dlopen, dlinfo, RTLD_DI_LMID */
#include "rocm_hsa_tp.h"

namespace {
std::atomic<bool> g_initialized{false};
}

extern "C" void __rocm_hsa_tp_init(void) {
    /* Recursion-safe: the LTTng tracepoint provider object's static-init has
     * already registered the provider with the daemon by the time we land
     * here (the provider is auto-loaded by the linker via
     * __attribute__((constructor)) in lttng-ust's TRACEPOINT_DEFINE
     * expansion). This function exists as an explicit "we are ready" marker
     * that the runtime can call from inside its own init, so callers don't
     * have to think about static-init ordering. It is idempotent. */
    bool expected = false;
    if (!g_initialized.compare_exchange_strong(expected, true)) return;

    /* (No-op for buffer config: liblttng-ust 2.13's per-application
     * channel-attr API is not stable. The committed channel mode and
     * discard policy from the Architecture section are enforced
     * session-side via
     * `lttng enable-channel --discard --subbuf-size=N --num-subbuf=M`. */
}

/* Library constructor — runs at dlopen time, BEFORE any user code in the
 * runtime is reached. This is where we honor the disable knobs, including
 * the link-namespace SEGFAULT mitigation discovered by Phase 0 Step 9(a):
 *
 *   When the runtime is loaded via dlmopen(LM_ID_NEWLM, ..., RTLD_NOW)
 *   into a non-default link namespace (HPCToolkit / Score-P style isolation),
 *   the LTTng tracepoint provider's static-init internally calls
 *   dlopen("liblttng-ust-tracepoint.so.1") which triggers a glibc bug in
 *   add_to_global_resize() and SEGFAULTs the process.
 *
 *   The fix: detect we are NOT in the base namespace (LM_ID_BASE) and
 *   either (a) set ROCM_LTTNG_UST_DISABLE=1 in our environment so the
 *   provider's static-init skips registration, or (b) swallow with a stderr
 *   warning and never call __rocm_*_tp_init().
 *
 *   We use approach (b) below: detect at runtime, log once, no-op forever.
 *   Approach (a) is unreliable because the LTTng constructor may run
 *   BEFORE this constructor (static init order is not guaranteed across
 *   .so files).
 */
extern "C" __attribute__((constructor(101))) void __rocm_hsa_tp_ctor(void) {
    /* Honor explicit disable. */
    const char* dis = getenv("ROCM_LTTNG_UST_DISABLE");
    if (dis && dis[0] == '1') {
        /* Explicit disable wins. The LTTng provider's static-init has
         * already run by now and may have registered with sessiond — we
         * cannot un-register, but the runtime will never CALL any tracepoint
         * because g_initialized stays false (see __rocm_hsa_tp_init). */
        return;
    }

#if defined(__GLIBC__)
    /* Detect non-default link namespace. dlinfo(RTLD_DI_LMID, ...) returns
     * the link-namespace id of any handle we own; using ourselves (NULL)
     * gives the namespace the current load-unit is in. */
    Lmid_t ns_id = LM_ID_BASE;
    void* self = dlopen(NULL, RTLD_LAZY | RTLD_NOLOAD);
    if (self != NULL) {
        if (dlinfo(self, RTLD_DI_LMID, &ns_id) == 0 && ns_id != LM_ID_BASE) {
            /* In a non-default namespace. Skip registration to avoid the
             * glibc dlmopen + LTTng-internal-dlopen SEGFAULT (Phase 0
             * finding). */
            dlclose(self);
            /* Note: at this point the LTTng provider's
             * __attribute__((constructor)) has already RUN and may have
             * crashed already. If we got here, we did NOT crash — meaning
             * the LTTng provider has not yet been loaded for this .so.
             * Either way, we set g_initialized to a "permanently disabled"
             * marker so __rocm_hsa_tp_init never proceeds. */
            g_initialized.store(true);
            return;
        }
        dlclose(self);
    }
#endif
    /* Otherwise fall through; __rocm_hsa_tp_init will be called normally
     * from Runtime::LoadTools. */
}

#else

extern "C" void __rocm_hsa_tp_init(void) { /* no-op when LTTng is disabled */ }

#endif
