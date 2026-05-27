/*
 * Standalone test: hipStreamBatchMemOp compatibility with different memory types.
 *
 * Reproduces the scenario from rma_proxy_launch.cc where CU_STREAM_MEM_OP_WAIT_VALUE_64
 * is used on a signalsDev buffer.  The test checks which allocation strategies produce
 * an address that hipStreamBatchMemOp accepts for WAIT/WRITE_VALUE_64 operations.
 *
 * Build:
 *   hipcc -o hip_stream_batch_memop_test hip_stream_batch_memop_test.cpp -lpthread
 *
 * Run:
 *   ./hip_stream_batch_memop_test
 */

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Forward declarations
static bool probe_same_stream(const char* label, void* ptr, hipStream_t stream);
static bool probe_cross_thread(const char* label, void* ptr, hipStream_t stream);
static bool probe_nonbatch(const char* label, void* ptr, hipStream_t stream);

#define HIP_CHECK(call)                                                         \
  do {                                                                          \
    hipError_t _e = (call);                                                     \
    if (_e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error '%s' (%d) at %s:%d\n",                        \
              hipGetErrorString(_e), (int)_e, __FILE__, __LINE__);              \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

static const char* pass_fail(bool ok) { return ok ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m"; }

// ---------------------------------------------------------------------------
// Core probe: submit WRITE_VALUE_64(1) then WAIT_VALUE_64 GEQ(1) on the same
// stream, then synchronize.  This exercises address validation without needing
// a second thread — if the address type is rejected, hipStreamBatchMemOp
// itself returns an error immediately.
// ---------------------------------------------------------------------------

static bool probe_same_stream(const char* label, void* ptr, hipStream_t stream)
{
    printf("  [same-stream] %-52s ", label);
    fflush(stdout);

    // Zero the slot from the CPU (works for CPU-accessible memory types; for
    // pure device memory the memset on stream below covers initialization).
    hipStreamBatchMemOpParams ops[2];
    memset(ops, 0, sizeof(ops));

    ops[0].operation              = hipStreamMemOpWriteValue64;
    ops[0].writeValue.operation   = hipStreamMemOpWriteValue64;
    ops[0].writeValue.address     = (hipDeviceptr_t)ptr;
    ops[0].writeValue.value       = 1ULL;
    ops[0].writeValue.flags       = 0;

    ops[1].operation              = hipStreamMemOpWaitValue64;
    ops[1].waitValue.operation    = hipStreamMemOpWaitValue64;
    ops[1].waitValue.address      = (hipDeviceptr_t)ptr;
    ops[1].waitValue.value        = 1ULL;
    ops[1].waitValue.flags        = hipStreamWaitValueGte;

    hipError_t e = hipStreamBatchMemOp(stream, 2, ops, 0);
    if (e != hipSuccess) {
        printf("%s  (hipStreamBatchMemOp: %s)\n", pass_fail(false), hipGetErrorString(e));
        return false;
    }

    e = hipStreamSynchronize(stream);
    if (e != hipSuccess) {
        printf("%s  (sync: %s)\n", pass_fail(false), hipGetErrorString(e));
        return false;
    }

    printf("%s\n", pass_fail(true));
    return true;
}

// ---------------------------------------------------------------------------
// Cross-thread probe: stream issues WAIT_VALUE_64 GEQ(1); a CPU thread writes
// 1 after a short delay.  Tests that the wait actually blocks and unblocks.
// Only run this when the same-stream probe already passed to avoid a hang.
// ---------------------------------------------------------------------------

struct ThreadArg { volatile uint64_t* ptr; unsigned delay_us; };

static void* writer_thread(void* arg)
{
    struct ThreadArg* a = (struct ThreadArg*)arg;
    usleep(a->delay_us);
    __atomic_store_n(a->ptr, 1ULL, __ATOMIC_SEQ_CST);
    return nullptr;
}

static bool probe_cross_thread(const char* label, void* ptr, hipStream_t stream)
{
    printf("  [cross-thread] %-51s ", label);
    fflush(stdout);

    volatile uint64_t* addr = (volatile uint64_t*)ptr;
    *addr = 0;

    // CPU thread will write 1 after 50 ms.
    struct ThreadArg arg = { addr, 50000 };
    pthread_t tid;
    if (pthread_create(&tid, nullptr, writer_thread, &arg) != 0) {
        printf("%s  (pthread_create failed)\n", pass_fail(false));
        return false;
    }

    hipStreamBatchMemOpParams op;
    memset(&op, 0, sizeof(op));
    op.operation           = hipStreamMemOpWaitValue64;
    op.waitValue.operation = hipStreamMemOpWaitValue64;
    op.waitValue.address   = (hipDeviceptr_t)ptr;
    op.waitValue.value     = 1ULL;
    op.waitValue.flags     = hipStreamWaitValueGte;

    hipError_t e = hipStreamBatchMemOp(stream, 1, &op, 0);
    if (e != hipSuccess) {
        pthread_join(tid, nullptr);
        printf("%s  (hipStreamBatchMemOp: %s)\n", pass_fail(false), hipGetErrorString(e));
        return false;
    }

    e = hipStreamSynchronize(stream);
    pthread_join(tid, nullptr);
    if (e != hipSuccess) {
        printf("%s  (sync: %s)\n", pass_fail(false), hipGetErrorString(e));
        return false;
    }

    // Verify the value was written correctly.
    uint64_t val = __atomic_load_n(addr, __ATOMIC_SEQ_CST);
    bool ok = (val >= 1ULL);
    printf("%s%s\n", pass_fail(ok), ok ? "" : "  (wrong value after sync)");
    return ok;
}

// ---------------------------------------------------------------------------
// Per-device test matrix
// ---------------------------------------------------------------------------

static void run_device(int dev)
{
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("\n=== Device %d: %s ===\n", dev, prop.name);
    HIP_CHECK(hipSetDevice(dev));

    // Check device capability.
    int canWait = 0;
    hipDeviceGetAttribute(&canWait, hipDeviceAttributeCanUseStreamWaitValue, dev);
    printf("    hipDeviceAttributeCanUseStreamWaitValue = %d\n\n", canWait);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    const size_t SZ = 16 * sizeof(uint64_t); // 128 bytes — plenty of room

    // ---- Allocation variants ----

    // 1. Fine-grain device memory (the failing case in RCCL rma_proxy.cc)
    void* fg = nullptr;
    hipError_t fg_err = hipExtMallocWithFlags(&fg, SZ, hipDeviceMallocFinegrained);
    if (fg_err != hipSuccess) {
        printf("  hipExtMallocWithFlags(Finegrained): alloc failed (%s) — skipping\n",
               hipGetErrorString(fg_err));
    }

    // 2. Host pinned, coherent (CPU-accessible, expected to work with stream wait)
    void* pinned_coh = nullptr;
    hipHostMalloc(&pinned_coh, SZ, hipHostMallocCoherent);

    // 3. Host pinned, default flags
    void* pinned_def = nullptr;
    hipHostMalloc(&pinned_def, SZ, 0);

    // 4. Host pinned, mapped (device can access via device pointer)
    void* pinned_map = nullptr;
    hipHostMalloc(&pinned_map, SZ, hipHostMallocMapped);
    void* pinned_map_devptr = nullptr;
    if (pinned_map)
        hipHostGetDevicePointer(&pinned_map_devptr, pinned_map, 0);

    // 5. Managed memory
    void* managed = nullptr;
    hipMallocManaged(&managed, SZ, hipMemAttachGlobal);

    // 6. Regular (non-coherent) device memory — expected to fail
    void* devmem = nullptr;
    hipMalloc(&devmem, SZ);

    printf("  Allocation addresses:\n");
    printf("    fine-grain device (hipDeviceMallocFinegrained) : %p\n", fg);
    printf("    host pinned, coherent                          : %p\n", pinned_coh);
    printf("    host pinned, default flags                     : %p\n", pinned_def);
    printf("    host pinned, mapped (host ptr)                 : %p\n", pinned_map);
    printf("    host pinned, mapped (device ptr)               : %p\n", pinned_map_devptr);
    printf("    managed                                        : %p\n", managed);
    printf("    regular device memory                          : %p\n", devmem);
    printf("\n");

    // ---- Same-stream probes (address acceptance test) ----
    printf("  --- same-stream WRITE_VALUE_64 + WAIT_VALUE_64 ---\n");

    bool fg_pass      = fg             && probe_same_stream("hipDeviceMallocFinegrained",   fg,              stream);
    bool coh_pass     = pinned_coh     && probe_same_stream("hipHostMalloc(Coherent)",       pinned_coh,      stream);
    bool def_pass     = pinned_def     && probe_same_stream("hipHostMalloc(default)",        pinned_def,      stream);
    bool map_h_pass   = pinned_map     && probe_same_stream("hipHostMalloc(Mapped, host ptr)",  pinned_map,   stream);
    bool map_d_pass   = pinned_map_devptr
                                       && probe_same_stream("hipHostMalloc(Mapped, dev ptr)",   pinned_map_devptr, stream);
    bool man_pass     = managed        && probe_same_stream("hipMallocManaged",              managed,         stream);
    bool dev_pass     = devmem         && probe_same_stream("hipMalloc (device)",            devmem,          stream);

    // ---- Cross-thread probes (only for CPU-accessible memory that passed) ----
    // Fine-grain device memory is CPU-accessible, so include it if same-stream passed.
    printf("\n  --- cross-thread CPU write -> stream WAIT_VALUE_64 ---\n");

    if (fg_pass)    probe_cross_thread("hipDeviceMallocFinegrained",  fg,         stream);
    if (coh_pass)   probe_cross_thread("hipHostMalloc(Coherent)",      pinned_coh, stream);
    if (def_pass)   probe_cross_thread("hipHostMalloc(default)",       pinned_def, stream);
    if (map_h_pass) probe_cross_thread("hipHostMalloc(Mapped, host ptr)", pinned_map, stream);
    if (man_pass)   probe_cross_thread("hipMallocManaged",             managed,    stream);

    // ---- Non-batch baseline (scalar API) ----
    printf("\n  --- non-batch hipStreamWriteValue64/WaitValue64 (baseline) ---\n");
    bool fg_nb    = fg             && probe_nonbatch("hipDeviceMallocFinegrained",      fg,              stream);
    bool coh_nb   = pinned_coh     && probe_nonbatch("hipHostMalloc(Coherent)",          pinned_coh,      stream);
    bool def_nb   = pinned_def     && probe_nonbatch("hipHostMalloc(default)",           pinned_def,      stream);
    bool maph_nb  = pinned_map     && probe_nonbatch("hipHostMalloc(Mapped, host ptr)",  pinned_map,      stream);
    bool mapd_nb  = pinned_map_devptr
                                   && probe_nonbatch("hipHostMalloc(Mapped, dev ptr)",   pinned_map_devptr, stream);
    bool man_nb   = managed        && probe_nonbatch("hipMallocManaged",                 managed,         stream);
    bool dev_nb   = devmem         && probe_nonbatch("hipMalloc (device)",               devmem,          stream);

    printf("\n  --- Summary for device %d ---\n", dev);
    printf("    %-52s  batch=%-6s  scalar=%s\n", "hipDeviceMallocFinegrained",
           pass_fail(fg_pass),    pass_fail(fg_nb));
    printf("    %-52s  batch=%-6s  scalar=%s\n", "hipHostMalloc(Coherent)",
           pass_fail(coh_pass),   pass_fail(coh_nb));
    printf("    %-52s  batch=%-6s  scalar=%s\n", "hipHostMalloc(default)",
           pass_fail(def_pass),   pass_fail(def_nb));
    printf("    %-52s  batch=%-6s  scalar=%s\n", "hipHostMalloc(Mapped, host ptr)",
           pass_fail(map_h_pass), pass_fail(maph_nb));
    printf("    %-52s  batch=%-6s  scalar=%s\n", "hipHostMalloc(Mapped, dev ptr)",
           pass_fail(map_d_pass), pass_fail(mapd_nb));
    printf("    %-52s  batch=%-6s  scalar=%s\n", "hipMallocManaged",
           pass_fail(man_pass),   pass_fail(man_nb));
    printf("    %-52s  batch=%-6s  scalar=%s\n", "hipMalloc (device, non-coherent)",
           pass_fail(dev_pass),   pass_fail(dev_nb));

    // Cleanup
    if (fg)              hipFree(fg);
    if (pinned_coh)      hipHostFree(pinned_coh);
    if (pinned_def)      hipHostFree(pinned_def);
    if (pinned_map)      hipHostFree(pinned_map);
    if (managed)         hipFree(managed);
    if (devmem)          hipFree(devmem);
    hipStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// Non-batch baseline: hipStreamWriteValue64 / hipStreamWaitValue64.
// These are the scalar equivalents and should work wherever the batch API
// works (and often more broadly).  If these fail too, the issue is with
// the memory type itself; if they pass while batch fails, the issue is
// specific to the batch API.
// ---------------------------------------------------------------------------

// Same-stream: validates address acceptance only (write always precedes wait).
static bool probe_nonbatch_same_stream(const char* label, void* ptr, hipStream_t stream)
{
    printf("  [non-batch same-stream]   %-40s ", label);
    fflush(stdout);

    hipError_t e = hipStreamWriteValue64(stream, (hipDeviceptr_t)ptr, 1ULL, 0);
    if (e != hipSuccess) {
        printf("%s  (WriteValue64: %s)\n", pass_fail(false), hipGetErrorString(e));
        return false;
    }

    e = hipStreamWaitValue64(stream, (hipDeviceptr_t)ptr, 1ULL, hipStreamWaitValueGte);
    if (e != hipSuccess) {
        printf("%s  (WaitValue64: %s)\n", pass_fail(false), hipGetErrorString(e));
        return false;
    }

    e = hipStreamSynchronize(stream);
    if (e != hipSuccess) {
        printf("%s  (sync: %s)\n", pass_fail(false), hipGetErrorString(e));
        return false;
    }

    printf("%s\n", pass_fail(true));
    return true;
}

// Cross-thread: stream enqueues WaitValue64 BEFORE the CPU writes.
// This is the real RCCL use case — stream must actually block.
static bool probe_nonbatch_cross_thread(const char* label, void* ptr, hipStream_t stream)
{
    printf("  [non-batch cross-thread]  %-40s ", label);
    fflush(stdout);

    volatile uint64_t* addr = (volatile uint64_t*)ptr;
    *addr = 0;

    // Enqueue the wait first — stream will stall until *addr >= 1.
    hipError_t e = hipStreamWaitValue64(stream, (hipDeviceptr_t)ptr, 1ULL, hipStreamWaitValueGte);
    if (e != hipSuccess) {
        printf("%s  (WaitValue64 enqueue: %s)\n", pass_fail(false), hipGetErrorString(e));
        return false;
    }

    // CPU thread writes 1 after 50 ms — stream should unblock.
    struct ThreadArg arg = { addr, 50000 };
    pthread_t tid;
    if (pthread_create(&tid, nullptr, writer_thread, &arg) != 0) {
        printf("%s  (pthread_create failed)\n", pass_fail(false));
        return false;
    }

    e = hipStreamSynchronize(stream);
    pthread_join(tid, nullptr);
    if (e != hipSuccess) {
        printf("%s  (sync: %s)\n", pass_fail(false), hipGetErrorString(e));
        return false;
    }

    uint64_t val = __atomic_load_n(addr, __ATOMIC_SEQ_CST);
    bool ok = (val >= 1ULL);
    printf("%s%s\n", pass_fail(ok), ok ? "" : "  (value not written — stream didn't block)");
    return ok;
}

// Convenience: run both same-stream and cross-thread probes.
static bool probe_nonbatch(const char* label, void* ptr, hipStream_t stream)
{
    bool a = probe_nonbatch_same_stream(label, ptr, stream);
    bool b = a && probe_nonbatch_cross_thread(label, ptr, stream);
    return a && b;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    printf("hipStreamBatchMemOp memory-type compatibility test\n");
    printf("==================================================\n");
    printf("Scenario: RCCL rma_proxy_launch.cc uses WAIT_VALUE_64 on signalsDev.\n");
    printf("          This test checks which allocation types are accepted.\n");

    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    printf("Found %d HIP device(s)\n", ndev);

    int start = 0, end = ndev;
    if (argc > 1) {
        start = atoi(argv[1]);
        end   = start + 1;
    }

    for (int d = start; d < end; d++)
        run_device(d);

    printf("\nDone.\n");
    return 0;
}
