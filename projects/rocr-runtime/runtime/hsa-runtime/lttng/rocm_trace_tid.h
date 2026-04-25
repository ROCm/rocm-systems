#ifndef ROCM_TRACE_TID_H_
#define ROCM_TRACE_TID_H_

#include <stdint.h>

#if defined(__linux__)
  #include <sys/syscall.h>     /* SYS_gettid is the portable, arch-correct macro */
  #include <unistd.h>          /* syscall() prototype */
  /* glibc 2.30+ exposes gettid() directly. Older bases (Ubuntu 18.04, RHEL 8)
   * lack it; fall through to the SYS_gettid syscall path. */
  #if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 30))
    static inline uint32_t rocm_trace_current_tid(void) {
        return (uint32_t)gettid();   /* glibc inlines or caches in TLS */
    }
  #else
    static inline uint32_t rocm_trace_current_tid(void) {
        /* SYS_gettid is defined in <sys/syscall.h> per-arch:
         *   x86_64 -> 186, aarch64 -> 178, riscv64 -> 178, etc.
         * glibc caches the result per-thread starting in 2.18 if the syscall
         * was used at thread-init; expect 5-15 ns/call overall. */
        return (uint32_t)syscall(SYS_gettid);
    }
  #endif
#else
  static inline uint32_t rocm_trace_current_tid(void) { return 0; }
#endif

/* Compose a process-globally-unique 64-bit correlation id:
 *   high 32 bits = tid (truncated to 32 bits)
 *   low  32 bits = per-thread monotonic counter
 * Two threads cannot collide because their tids differ. A single thread
 * cannot collide with itself within 2^32 events (~4 billion API calls).
 * Cost: 1 TLS load + 1 increment + 1 (cached) tid read. */
static inline uint64_t rocm_trace_next_corr_id(void) {
    static __thread uint32_t counter = 0;
    static __thread uint32_t cached_tid = 0;
    if (cached_tid == 0) cached_tid = rocm_trace_current_tid();
    return ((uint64_t)cached_tid << 32) | (uint64_t)(++counter);
}

/* Push a corr id into the TLS slot read by downstream tracepoints
 * (e.g., HSA's StoreRelaxed when called on the same thread that ran
 * the HIP API). Returns the previous value so the caller can restore
 * it on exit, supporting nested tracepoints.
 *
 * NOTE: this slot is `static __thread`, which makes it TU-local —
 * each translation unit including this header gets its own slot.
 * Cross-runtime corr id propagation (HIP -> HSA on the same thread)
 * therefore does NOT work through this slot today; it would require
 * the slot to live in a single shared .so (e.g., librocprofiler-register).
 * The HSA-side tracepoints will report active corr id 0 when called
 * from outside the HSA runtime; consumers should fall back to
 * (tid_high_32, timestamp) joins for cross-runtime correlation. */
static __thread uint64_t g_rocm_active_corr_id;
static inline uint64_t rocm_trace_push_corr_id(uint64_t v) {
    uint64_t prev = g_rocm_active_corr_id;
    g_rocm_active_corr_id = v;
    return prev;
}
static inline void rocm_trace_pop_corr_id(uint64_t prev) {
    g_rocm_active_corr_id = prev;
}
static inline uint64_t rocm_trace_active_corr_id(void) {
    return g_rocm_active_corr_id;
}

#endif /* ROCM_TRACE_TID_H_ */
