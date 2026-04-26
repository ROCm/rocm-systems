#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST

/* dladdr1 + RTLD_DL_LINKMAP are GNU extensions; _GNU_SOURCE makes them
 * visible from <dlfcn.h>. Defined before any system header is included. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <atomic>
#include <cstdlib>
#include <link.h>     /* for Lmid_t, LM_ID_BASE */
#include <dlfcn.h>    /* for dladdr1, RTLD_DL_LINKMAP, dlinfo, RTLD_DI_LMID */
#include "rocm_hsa_tp.h"

/* Runtime-wide kill switch -- definition. The emit helpers in
 * rocm_trace_emit.h short-circuit on this flag (single relaxed atomic load
 * per emit). Set true when ROCM_LTTNG_UST_DISABLE=1 or when running in a
 * non-default link namespace (dlmopen mitigation). Default visibility so
 * the dynamic linker resolves all loads from the HSA TUs to this single
 * definition.
 *
 * The symbol is named `rocm_hsa_trace_g_disabled` (not the shared
 * `rocm_trace_g_disabled` it used to be) so ELF symbol interposition
 * cannot bind references to HIP's `rocm_hip_trace_g_disabled` (or vice
 * versa) when both libamdhip64 and libhsa-runtime64 are loaded into the
 * same process. */
/* Note: NOT `extern "C"` — std::atomic<bool> is a C++ type and combining
 * `extern` with initializer is flagged by GCC -Werror=extern-initialized.
 * The header declares with matching C++ linkage. */
std::atomic<bool> rocm_hsa_trace_g_disabled
    __attribute__((visibility("default"))) {false};

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
 *   Mitigation: set rocm_hsa_trace_g_disabled so the per-call emit helpers
 *   short-circuit before touching any LTTng state, AND set g_initialized
 *   so __rocm_hsa_tp_init never proceeds.
 */
extern "C" __attribute__((constructor(101))) void __rocm_hsa_tp_ctor(void) {
    /* Honor explicit disable. */
    const char* dis = getenv("ROCM_LTTNG_UST_DISABLE");
    if (dis && dis[0] == '1') {
        /* Explicit disable wins. The LTTng provider's static-init has
         * already run by now and may have registered with sessiond — we
         * cannot un-register, but the runtime will never CALL any tracepoint
         * because rocm_hsa_trace_g_disabled is now true. g_initialized is also
         * marked so __rocm_hsa_tp_init bails out. */
        rocm_hsa_trace_g_disabled.store(true, std::memory_order_relaxed);
        g_initialized.store(true);
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
     * __rocm_hsa_tp_ctor), then bridge to `dlinfo(handle, RTLD_DI_LMID, ...)`
     * to read its namespace id. The link_map handle is exactly what dlopen
     * returns under the hood, so dlinfo accepts it directly. We avoid
     * touching `lm->l_lmid` directly because that field is not exposed in
     * the public glibc <link.h>. */
    Lmid_t ns_id = LM_ID_BASE;
    Dl_info info;
    void*   lm_handle = NULL;
    if (dladdr1(reinterpret_cast<void*>(&__rocm_hsa_tp_ctor),
                &info, &lm_handle, RTLD_DL_LINKMAP) != 0
        && lm_handle != NULL
        && dlinfo(lm_handle, RTLD_DI_LMID, &ns_id) == 0
        && ns_id != LM_ID_BASE) {
        /* In a non-default namespace. Skip registration to avoid the
         * glibc dlmopen + LTTng-internal-dlopen SEGFAULT (Phase 0
         * finding). */
        rocm_hsa_trace_g_disabled.store(true, std::memory_order_relaxed);
        g_initialized.store(true);
        return;
    }
#endif
    /* Otherwise fall through; __rocm_hsa_tp_init will be called normally
     * from Runtime::LoadTools. */
}

#else

extern "C" void __rocm_hsa_tp_init(void) { /* no-op when LTTng is disabled */ }

#endif
