// a2-gpu-side: does a running (or newly launched) kernel see PCIe-originated writes to memory it
// has already cached? Standalone HIP probe, no hipFile.
//
// Build: amdclang++ -x hip --offload-arch=gfx1030 -O2 -std=c++20 stale-probe.hip -o stale-probe
//
// Per iteration:
//   1. init: buffer := pattern OLD (hipMemcpy H2D from pinned staging + hipDeviceSynchronize)
//   2. "during": launch checker on non-blocking stream sK; each WG primes (reads OLD), sets
//      ready[wg] (release, system), spins on host flag (relaxed system load + s_sleep).
//      "after": launch primer kernel on sK, sync it (or keep it resident, --after-resident).
//   3. host writes pattern NEW through --writer
//   4. host --intervene
//   5. "during": publish flag (seq_cst store), kernel does --acq then reads with --load and counts
//      words still OLD (stale) / neither (unexpected). With --poll-ms it keeps re-reading (each
//      round = acq + load) until clean or timeout, recording wall_clock64 latency.
//      "after": launch a new checker on sK (same stream) or sC (other stream).
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <x86intrin.h>
#ifdef WITH_HIPFILE
#include <fcntl.h>
#include <hipfile.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));    \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

constexpr unsigned THREADS = 256;

enum Acq : int {
    ACQ_NONE,
    ACQ_ATOMIC_SYS,  // __hip_atomic_load(flag, ACQUIRE, SYSTEM)  (John's kernel)
    ACQ_FENCE_SYS,   // __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "")
    ACQ_FENCE_AGENT, // __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")
    ACQ_GL0,         // buffer_gl0_inv
    ACQ_GL01,        // buffer_gl0_inv; buffer_gl1_inv
    ACQ_COUNT
};
static const char *acq_names[] = {"none", "atomic_sys", "fence_sys", "fence_agent", "gl0inv", "gl01inv"};

enum Load : int {
    LD_PLAIN,
    LD_RELAXED_SYS, // __hip_atomic_load(p, RELAXED, SYSTEM)
    LD_NT,          // __builtin_nontemporal_load
    LD_VOLATILE,
    LD_GLC,         // asm global_load_dword glc
    LD_GLC_DLC,     // asm glc dlc
    LD_GLC_SLC_DLC, // asm glc slc dlc
    LD_SLC,         // asm slc
    LD_BUF_ALL,     // __builtin_amdgcn_raw_buffer_load_b32 aux = glc|slc|dlc
    LD_COUNT
};
static const char *load_names[] = {"plain", "relaxed_sys", "nontemporal", "volatile", "glc",
                                   "glc_dlc", "glc_slc_dlc", "slc", "buf_glc_slc_dlc"};

struct Result {
    uint32_t prev_mismatch; // words != OLD during prime
    uint32_t stale_first;   // words == OLD on first check after flag
    uint32_t unexp_first;   // words neither OLD nor NEW on first check
    uint32_t timed_out;     // flag never arrived
    uint32_t rounds;        // extra polling rounds until clean (0 = clean first time)
    uint32_t resolved;      // 1 if clean at the end
    uint64_t t_flag;        // wall clock when WG saw flag
    uint64_t t_clean;       // wall clock when WG first saw all NEW (0 if never)
};

struct Args {
    const uint32_t *buf;
    size_t words;
    uint32_t old_base, new_base;
    int prime;           // read & check OLD before waiting
    uint32_t *flag;      // host flag (nullptr: no wait)
    uint32_t *acqp;      // location the acquire variants load from (always valid)
    uint32_t *ready;     // per WG
    uint64_t wait_ticks; // timeout waiting for flag
    uint64_t poll_ticks; // keep re-reading for this long if stale
    Result *res;
};

template <int L> __device__ __forceinline__ uint32_t ld(const uint32_t *p) {
    if constexpr (L == LD_PLAIN) {
        return *p;
    } else if constexpr (L == LD_RELAXED_SYS) {
        return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    } else if constexpr (L == LD_NT) {
        return __builtin_nontemporal_load(p);
    } else if constexpr (L == LD_VOLATILE) {
        return *(const volatile uint32_t *)p;
    } else if constexpr (L == LD_GLC) {
        uint32_t r;
        asm volatile("global_load_dword %0, %1, off glc\n\ts_waitcnt vmcnt(0)" : "=v"(r) : "v"(p) : "memory");
        return r;
    } else if constexpr (L == LD_GLC_DLC) {
        uint32_t r;
        asm volatile("global_load_dword %0, %1, off glc dlc\n\ts_waitcnt vmcnt(0)" : "=v"(r) : "v"(p) : "memory");
        return r;
    } else if constexpr (L == LD_GLC_SLC_DLC) {
        uint32_t r;
        asm volatile("global_load_dword %0, %1, off glc slc dlc\n\ts_waitcnt vmcnt(0)" : "=v"(r) : "v"(p) : "memory");
        return r;
    } else if constexpr (L == LD_SLC) {
        uint32_t r;
        asm volatile("global_load_dword %0, %1, off slc\n\ts_waitcnt vmcnt(0)" : "=v"(r) : "v"(p) : "memory");
        return r;
    } else {
        // Whole buffer as one raw buffer resource; caller passes byte offset via p - base
        return 0;
    }
}

template <int A> __device__ __forceinline__ void acquire(uint32_t *flag) {
    if constexpr (A == ACQ_ATOMIC_SYS) {
        (void)__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    } else if constexpr (A == ACQ_FENCE_SYS) {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
    } else if constexpr (A == ACQ_FENCE_AGENT) {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    } else if constexpr (A == ACQ_GL0) {
        asm volatile("buffer_gl0_inv" ::: "memory");
    } else if constexpr (A == ACQ_GL01) {
        asm volatile("buffer_gl0_inv\n\tbuffer_gl1_inv" ::: "memory");
    }
}

template <int L>
__device__ __forceinline__ void check_range(const Args &a, size_t b, size_t e, uint32_t *stale, uint32_t *unexp) {
    uint32_t s = 0, u = 0;
    if constexpr (L == LD_BUF_ALL) {
        __amdgpu_buffer_rsrc_t rsrc = __builtin_amdgcn_make_buffer_rsrc((void *)a.buf, 0, 0x7fffffff, 0x31014000);
        for (size_t i = b + threadIdx.x; i < e; i += blockDim.x) {
            uint32_t v = __builtin_amdgcn_raw_buffer_load_b32(rsrc, (int)(i * 4), 0, 1 | 2 | 4);
            if (v == a.new_base + (uint32_t)i) continue;
            if (v == a.old_base + (uint32_t)i) s++; else u++;
        }
    } else {
        for (size_t i = b + threadIdx.x; i < e; i += blockDim.x) {
            uint32_t v = ld<L>(a.buf + i);
            if (v == a.new_base + (uint32_t)i) continue;
            if (v == a.old_base + (uint32_t)i) s++; else u++;
        }
    }
    if (s) atomicAdd(stale, s);
    if (u) atomicAdd(unexp, u);
}

template <int A, int L> __global__ void checker(Args a) {
    __shared__ uint32_t prev, stale, unexp, seen;
    __shared__ uint64_t t_flag;
    if (threadIdx.x == 0) { prev = 0; stale = 0; unexp = 0; seen = 1; t_flag = 0; }
    __syncthreads();
    const size_t b = blockIdx.x * a.words / gridDim.x, e = (blockIdx.x + 1) * a.words / gridDim.x;
    if (a.prime) {
        uint32_t m = 0;
        for (size_t i = b + threadIdx.x; i < e; i += blockDim.x)
            if (a.buf[i] != a.old_base + (uint32_t)i) m++;
        if (m) atomicAdd(&prev, m);
    }
    if (a.flag) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __hip_atomic_store(&a.ready[blockIdx.x], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            const uint64_t dl = wall_clock64() + a.wait_ticks;
            for (;;) {
                if (__hip_atomic_load(a.flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != 0) break;
                if ((uint64_t)wall_clock64() >= dl) { seen = 0; break; }
                __builtin_amdgcn_s_sleep(127);
            }
            t_flag = wall_clock64();
        }
        __syncthreads();
    } else if (threadIdx.x == 0) {
        t_flag = wall_clock64();
    }
    acquire<A>(a.acqp);
    check_range<L>(a, b, e, &stale, &unexp);
    __syncthreads();
    const uint32_t s0 = stale, u0 = unexp;
    uint32_t rounds = 0;
    uint64_t t_clean = (s0 == 0 && u0 == 0) ? t_flag : 0;
    if ((s0 || u0) && a.poll_ticks) {
        const uint64_t dl = t_flag + a.poll_ticks;
        for (;;) {
            __syncthreads();
            if (threadIdx.x == 0) { stale = 0; unexp = 0; }
            __syncthreads();
            acquire<A>(a.acqp);
            check_range<L>(a, b, e, &stale, &unexp);
            __syncthreads();
            rounds++;
            if (stale == 0 && unexp == 0) { t_clean = wall_clock64(); break; }
            if ((uint64_t)wall_clock64() >= dl) break;
            if (threadIdx.x == 0) __builtin_amdgcn_s_sleep(2);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        a.res[blockIdx.x] = Result{prev, s0, u0, seen ? 0u : 1u, rounds, t_clean ? 1u : 0u, t_flag, t_clean};
    }
}

// Primer for --mode after: reads buffer (brings it into caches); optionally stays resident until
// *hold != 0.
__global__ void primer(const uint32_t *buf, size_t words, uint32_t old_base, uint32_t *bad, uint32_t *hold,
                       uint32_t *ready) {
    const size_t b = blockIdx.x * words / gridDim.x, e = (blockIdx.x + 1) * words / gridDim.x;
    uint32_t m = 0;
    for (size_t i = b + threadIdx.x; i < e; i += blockDim.x)
        if (buf[i] != old_base + (uint32_t)i) m++;
    if (m) atomicAdd(bad, m);
    if (hold) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __hip_atomic_store(&ready[blockIdx.x], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            const uint64_t dl = wall_clock64() + 500000000ull;
            while (__hip_atomic_load(hold, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0 &&
                   (uint64_t)wall_clock64() < dl)
                __builtin_amdgcn_s_sleep(127);
        }
    }
}

__global__ void empty_kernel(uint32_t *p) {
    if (p && threadIdx.x == 9999) *p = 1;
}

// Signals completion through fine-grained host memory, so the host can wait without any HIP sync
// (which would add a system-scope barrier packet).
__global__ void signal_kernel(uint32_t *done) {
    if (threadIdx.x == 0) __hip_atomic_store(done, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

static void list_pools() {
    hsa_init();
    hsa_iterate_agents(
        [](hsa_agent_t a, void *) {
            hsa_device_type_t t;
            hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
            hsa_amd_agent_iterate_memory_pools(
                a,
                [](hsa_amd_memory_pool_t p, void *d) {
                    hsa_amd_segment_t seg;
                    uint32_t fl = 0;
                    size_t sz = 0;
                    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
                    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
                    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &fl);
                    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SIZE, &sz);
                    printf("POOL agent=%s global_flags=0x%x [%s%s%s%s] size=%zu MiB\n", (const char *)d, fl,
                           fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT ? "KERNARG " : "",
                           fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED ? "FINE " : "",
                           fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED ? "COARSE " : "",
                           fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED ? "EXT_FINE " : "",
                           sz >> 20);
                    return HSA_STATUS_SUCCESS;
                },
                (void *)(t == HSA_DEVICE_TYPE_GPU ? "GPU" : "CPU"));
            return HSA_STATUS_SUCCESS;
        },
        nullptr);
}


// Raw HSA barrier-AND packet on a private queue, with chosen acquire/release fence scopes, to see
// which CP-side cache operation makes the running kernel see the CPU's writes.
struct HsaKick {
    hsa_agent_t gpu{};
    hsa_queue_t *q = nullptr;
    hsa_signal_t sig{};
    void init() {
        if (q) return;
        if (hsa_init() != HSA_STATUS_SUCCESS) { fprintf(stderr, "hsa_init failed\n"); exit(2); }
        hsa_iterate_agents(
            [](hsa_agent_t a, void *d) {
                hsa_device_type_t t;
                hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
                if (t == HSA_DEVICE_TYPE_GPU) { *(hsa_agent_t *)d = a; return HSA_STATUS_INFO_BREAK; }
                return HSA_STATUS_SUCCESS;
            },
            &gpu);
        if (hsa_queue_create(gpu, 64, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &q) !=
            HSA_STATUS_SUCCESS) { fprintf(stderr, "hsa_queue_create failed\n"); exit(2); }
        hsa_signal_create(1, 0, nullptr, &sig);
    }
    void barrier(int acq, int rel) { // scopes: 0 none, 1 agent, 2 system
        init();
        hsa_signal_store_relaxed(sig, 1);
        const uint64_t idx = hsa_queue_add_write_index_relaxed(q, 1);
        auto *pk = (hsa_barrier_and_packet_t *)q->base_address + (idx % q->size);
        memset((char *)pk + 4, 0, sizeof(*pk) - 4);
        pk->completion_signal = sig;
        const uint16_t hdr = (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
                             (acq << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                             (rel << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
        __atomic_store_n((uint16_t *)pk, hdr, __ATOMIC_RELEASE);
        hsa_signal_store_screlease(q->doorbell_signal, idx);
        while (hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0) {}
    }
};
static HsaKick hsakick;

typedef void (*KFn)(Args);
template <int A, int L> struct Reg {
    static void fill(KFn (*t)[LD_COUNT]) {
        t[A][L] = checker<A, L>;
        if constexpr (L + 1 < LD_COUNT) Reg<A, L + 1>::fill(t);
        else if constexpr (A + 1 < ACQ_COUNT) Reg<A + 1, 0>::fill(t);
    }
};

static int find(const char *const *names, int n, const std::string &s) {
    for (int i = 0; i < n; i++)
        if (s == names[i]) return i;
    fprintf(stderr, "unknown value %s\n", s.c_str());
    exit(2);
}

int main(int argc, char **argv) {
    std::string mem = "fine", writer = "cpu", mode = "during", intervene = "none", acqs = "atomic_sys",
                loads = "plain", after_stream = "same", init = "memcpy";
    int iters = 200, prime = 1, wgs = 64, after_resident = 0;
    size_t kib = 1024;
    double poll_ms = 0, kick_delay_ms = 0, host_sleep_ms = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--mem") mem = v;
        else if (k == "--writer") writer = v;
        else if (k == "--mode") mode = v;
        else if (k == "--intervene") intervene = v;
        else if (k == "--acq") acqs = v;
        else if (k == "--load") loads = v;
        else if (k == "--iters") iters = std::stoi(v);
        else if (k == "--prime") prime = std::stoi(v);
        else if (k == "--wgs") wgs = std::stoi(v);
        else if (k == "--kib") kib = std::stoul(v);
        else if (k == "--poll-ms") poll_ms = std::stod(v);
        else if (k == "--kick-delay-ms") kick_delay_ms = std::stod(v);
        else if (k == "--host-sleep-ms") host_sleep_ms = std::stod(v);
        else if (k == "--after-stream") after_stream = v;
        else if (k == "--after-resident") after_resident = std::stoi(v);
        else if (k == "--init") init = v;
        else if (k == "--list-pools") { list_pools(); return 0; }
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    const int acq = find(acq_names, ACQ_COUNT, acqs), load = find(load_names, LD_COUNT, loads);
    static KFn table[ACQ_COUNT][LD_COUNT];
    Reg<0, 0>::fill(table);

    const size_t bytes = kib * 1024, words = bytes / 4;
    int clk_khz = 0;
    CHECK(hipDeviceGetAttribute(&clk_khz, hipDeviceAttributeWallClockRate, 0));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));

    uint32_t *buf = nullptr;
    bool host_mem = false;
    if (mem == "coarse") CHECK(hipMalloc(&buf, bytes));
    else if (mem == "fine") CHECK(hipExtMallocWithFlags((void **)&buf, bytes, hipDeviceMallocFinegrained));
    else if (mem == "uncached") CHECK(hipExtMallocWithFlags((void **)&buf, bytes, hipDeviceMallocUncached));
    else if (mem == "contiguous") CHECK(hipExtMallocWithFlags((void **)&buf, bytes, hipDeviceMallocContiguous));
    else if (mem == "managed") { CHECK(hipMallocManaged(&buf, bytes)); host_mem = true; }
    else if (mem == "hostcoh") { CHECK(hipHostMalloc(&buf, bytes, hipHostMallocCoherent)); host_mem = true; }
    else if (mem == "hostnc") { CHECK(hipHostMalloc(&buf, bytes, hipHostMallocNonCoherent)); host_mem = true; }
    else if (mem == "hostdefault") { CHECK(hipHostMalloc(&buf, bytes, 0)); host_mem = true; }
    else { fprintf(stderr, "bad --mem\n"); return 2; }

    uint32_t *staging_old, *staging_new, *flag, *ready, *bad, *hold;
    Result *res;
    CHECK(hipHostMalloc(&staging_old, bytes, 0));
    CHECK(hipHostMalloc(&staging_new, bytes, 0));
    CHECK(hipHostMalloc(&flag, 4, hipHostMallocCoherent));
    CHECK(hipHostMalloc(&hold, 4, hipHostMallocCoherent));
    CHECK(hipHostMalloc(&ready, wgs * 4, hipHostMallocCoherent));
    CHECK(hipHostMalloc(&bad, 4, hipHostMallocCoherent));
    uint32_t *kdone;
    CHECK(hipHostMalloc(&kdone, 4, hipHostMallocCoherent));
    hipEvent_t evstop;
    CHECK(hipEventCreate(&evstop));
    CHECK(hipHostMalloc(&res, wgs * sizeof(Result), hipHostMallocCoherent));
    std::vector<uint32_t> pageable(words);
#ifdef WITH_HIPFILE
    // File holds the 128 NEW patterns: slot k = iteration k mod 128, new_base = (2k+1) << 24
    hipFileHandle_t hf{};
    if (writer == "hipfile") {
        const char *path = getenv("HF_PATH");
        if (!path) { fprintf(stderr, "set HF_PATH\n"); return 2; }
        int wfd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
        std::vector<uint32_t> tmp(words);
        for (int k = 0; k < 128; k++) {
            for (size_t i = 0; i < words; i++) tmp[i] = ((uint32_t)(2 * k + 1) << 24) + (uint32_t)i;
            if (pwrite(wfd, tmp.data(), bytes, (off_t)k * bytes) != (ssize_t)bytes) { perror("pwrite"); return 2; }
        }
        fsync(wfd);
        close(wfd);
        int fd = open(path, O_RDONLY | O_DIRECT);
        if (fd < 0) { perror("open O_DIRECT"); return 2; }
        hipFileDescr_t d{};
        d.type = hipFileHandleTypeOpaqueFD;
        d.handle.fd = fd;
        if (hipFileDriverOpen().err != hipFileSuccess) { fprintf(stderr, "hipFileDriverOpen failed\n"); return 2; }
        if (hipFileHandleRegister(&hf, &d).err != hipFileSuccess) { fprintf(stderr, "HandleRegister failed\n"); return 2; }
    }
#endif
    uint32_t *other;
    CHECK(hipMalloc(&other, 4096));
    hipStream_t sK, sC, sX;
    CHECK(hipStreamCreateWithFlags(&sK, hipStreamNonBlocking));
    CHECK(hipStreamCreateWithFlags(&sC, hipStreamNonBlocking));
    CHECK(hipStreamCreateWithFlags(&sX, hipStreamNonBlocking));
    hipEvent_t evsys;
    CHECK(hipEventCreateWithFlags(&evsys, hipEventDisableTiming | hipEventReleaseToSystem));
    volatile uint32_t *hdp = (volatile uint32_t *)prop.hdpMemFlushCntl;

    hipPointerAttribute_t pa{};
    (void)hipPointerGetAttributes(&pa, buf);
    {
        hsa_amd_pointer_info_t pi{};
        pi.size = sizeof(pi);
        const hsa_status_t st = hsa_amd_pointer_info(buf, &pi, nullptr, nullptr, nullptr);
        const uint32_t fl = pi.global_flags;
        printf("POINTER mem=%s hsa_status=%d type=%d global_flags=0x%x [%s%s%s%s] size=%zu\n", mem.c_str(), (int)st,
               (int)pi.type, fl, fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT ? "KERNARG " : "",
               fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED ? "FINE " : "",
               fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED ? "COARSE " : "",
               fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED ? "EXT_FINE " : "", pi.sizeInBytes);
    }
    printf("# mem=%s ptr=%p type=%d isManaged=%d hdpMemFlushCntl=%p wallclock_khz=%d writer=%s mode=%s "
           "intervene=%s acq=%s load=%s prime=%d kib=%zu wgs=%d poll_ms=%g kick_delay_ms=%g init=%s "
           "after_stream=%s after_resident=%d\n",
           mem.c_str(), (void *)buf, (int)pa.type, pa.isManaged, (void *)hdp, clk_khz, writer.c_str(),
           mode.c_str(), intervene.c_str(), acqs.c_str(), loads.c_str(), prime, kib, wgs, poll_ms, kick_delay_ms,
           init.c_str(), after_stream.c_str(), after_resident);
    fflush(stdout);

    const double tick_us = 1000.0 / clk_khz;
    int it_stale = 0, it_unexp = 0, it_prev = 0, it_timeout = 0, it_unresolved = 0;
    uint64_t sum_stale = 0;
    std::vector<double> lat_us, spread_us;
    uint32_t max_rounds = 0;

    auto do_intervene = [&](const std::string &what) {
        if (what == "none") return;
        if (what == "kick_kernel") { empty_kernel<<<1, 64, 0, sX>>>(nullptr); CHECK(hipStreamSynchronize(sX)); }
        else if (what == "kick_kernel_nosync") { empty_kernel<<<1, 64, 0, sX>>>(nullptr); }
        else if (what == "kick_kernel_spinwait" || what == "kick_ext_stopevent_spinwait") {
            *kdone = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (what == "kick_kernel_spinwait") signal_kernel<<<1, 64, 0, sX>>>(kdone);
            else {
                void *kargs[] = {&kdone};
                CHECK(hipExtLaunchKernel((const void *)signal_kernel, dim3(1), dim3(64), kargs, 0, sX, nullptr, evstop, 0));
            }
            while (__atomic_load_n(kdone, __ATOMIC_ACQUIRE) == 0) _mm_pause();
        }
        else if (what == "kick_memset") { CHECK(hipMemsetAsync(other, 0, 4, sX)); CHECK(hipStreamSynchronize(sX)); }
        else if (what == "kick_memcpy_h2d") {
            CHECK(hipMemcpyAsync(other, staging_new, 4, hipMemcpyHostToDevice, sX));
            CHECK(hipStreamSynchronize(sX));
        } else if (what == "event_sys") { CHECK(hipEventRecord(evsys, sX)); CHECK(hipEventSynchronize(evsys)); }
        else if (what == "streamsync") { CHECK(hipStreamSynchronize(sX)); }
        else if (what.rfind("hsa_barrier_", 0) == 0) { // hsa_barrier_<acq><rel>, digits 0/1/2
            hsakick.barrier(what[12] - '0', what[13] - '0');
        } else if (what == "hdpflush") { *hdp = 1u; _mm_mfence(); } // no read-back (upstream a5cb344033c7)
        else if (what == "sfence") _mm_sfence();
        else if (what == "mfence") _mm_mfence();
        else if (what == "clflush") {
            for (size_t o = 0; o < bytes; o += 64) _mm_clflush((char *)buf + o);
            _mm_mfence();
        } else if (what == "sleep") {
            std::this_thread::sleep_for(std::chrono::microseconds((long)(host_sleep_ms * 1000)));
        } else { fprintf(stderr, "bad --intervene %s\n", what.c_str()); exit(2); }
    };

    if (intervene.rfind("hsa_barrier_", 0) == 0) hsakick.init(); // queue creation outside the loop
    for (int it = 0; it < iters; it++) {
        const uint32_t old_base = (uint32_t)(2 * it % 256) << 24, new_base = (uint32_t)((2 * it + 1) % 256) << 24;
        for (size_t i = 0; i < words; i++) { staging_old[i] = old_base + (uint32_t)i; staging_new[i] = new_base + (uint32_t)i; }
        // 1. init OLD
        if (init == "memcpy") CHECK(hipMemcpy(buf, staging_old, bytes, hipMemcpyDefault));
        else if (init == "cpu") { memcpy(buf, staging_old, bytes); _mm_mfence(); }
        CHECK(hipDeviceSynchronize());
        *flag = 0; *hold = 0; *bad = 0;
        memset(ready, 0, wgs * 4);
        memset(res, 0, wgs * sizeof(Result));
        std::atomic_thread_fence(std::memory_order_seq_cst);

        Args a{buf, words, old_base, new_base, prime, nullptr, flag, ready, (uint64_t)clk_khz * 1000 * 10,
               (uint64_t)(poll_ms * clk_khz), res};
        auto wait_ready = [&] {
            for (int w = 0; w < wgs; w++)
                while (__atomic_load_n(&ready[w], __ATOMIC_ACQUIRE) == 0) _mm_pause();
        };
        if (mode == "during") {
            a.flag = flag;
            // node bundle: a successful fallback hipFileRead leaves HIP's sticky last error set
            // to invalid argument, so clear it before checking the launch
            (void)hipGetLastError();
            table[acq][load]<<<wgs, THREADS, 0, sK>>>(a);
            CHECK(hipGetLastError());
            wait_ready();
        } else {
            a.prime = 0; // only the primer checks OLD; prev_mismatch then counts primer mismatches only
            (void)hipGetLastError();  // node bundle: see the comment in the "during" branch
            primer<<<wgs, THREADS, 0, sK>>>(buf, words, old_base, bad, after_resident ? hold : nullptr, ready);
            CHECK(hipGetLastError());
            if (after_resident) wait_ready();
            else CHECK(hipStreamSynchronize(sK));
        }
        // 3. write NEW
        if (writer == "cpu") { memcpy(buf, staging_new, bytes); std::atomic_thread_fence(std::memory_order_seq_cst); }
        else if (writer == "cpu_nt") {
            for (size_t i = 0; i < words; i += 4)
                _mm_stream_si128((__m128i *)(buf + i), _mm_load_si128((const __m128i *)(staging_new + i)));
            _mm_sfence();
        } else if (writer == "memcpy") CHECK(hipMemcpy(buf, staging_new, bytes, hipMemcpyDefault));
        else if (writer == "memcpy_pageable") {
            memcpy(pageable.data(), staging_new, bytes);
            CHECK(hipMemcpy(buf, pageable.data(), bytes, hipMemcpyHostToDevice));
        } else if (writer == "memcpyasync") {
            CHECK(hipMemcpyAsync(buf, staging_new, bytes, hipMemcpyDefault, sC));
            CHECK(hipStreamSynchronize(sC));
        }
#ifdef WITH_HIPFILE
        else if (writer == "hipfile") {
            const ssize_t n = hipFileRead(hf, buf, bytes, (hoff_t)(it % 128) * bytes, 0);
            if (n != (ssize_t)bytes) { fprintf(stderr, "hipFileRead returned %zd\n", n); return 2; }
        }
#endif
        else { fprintf(stderr, "bad --writer\n"); return 2; }
        // 4. intervene
        do_intervene(intervene);
        // 5. publish / launch checker
        if (mode == "during") {
            __atomic_store_n(flag, 1u, __ATOMIC_SEQ_CST);
            if (kick_delay_ms > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds((long)(kick_delay_ms * 1000)));
                empty_kernel<<<1, 64, 0, sX>>>(nullptr);
                CHECK(hipStreamSynchronize(sX));
            }
            CHECK(hipStreamSynchronize(sK));
        } else {
            // resident + same stream: let the primer finish, then the checker runs behind it on sK
            // with no host-side HIP sync in between. resident + other: checker runs on sC while
            // the primer is still resident.
            hipStream_t s = after_stream == "same" ? sK : sC;
            if (after_resident && after_stream == "same") __atomic_store_n(hold, 1u, __ATOMIC_SEQ_CST);
            (void)hipGetLastError();  // node bundle: see the comment in the "during" branch
            table[acq][load]<<<wgs, THREADS, 0, s>>>(a);
            CHECK(hipGetLastError());
            CHECK(hipStreamSynchronize(s));
            if (after_resident) { __atomic_store_n(hold, 1u, __ATOMIC_SEQ_CST); }
            CHECK(hipDeviceSynchronize());
        }
        // tally
        uint64_t s = 0, u = 0, p = 0, tmo = 0, unres = 0, tf_min = UINT64_MAX, tc_min = UINT64_MAX, tc_max = 0;
        for (int w = 0; w < wgs; w++) {
            s += res[w].stale_first; u += res[w].unexp_first; p += res[w].prev_mismatch; tmo += res[w].timed_out;
            max_rounds = std::max(max_rounds, res[w].rounds);
            if (res[w].stale_first || res[w].unexp_first) {
                if (!res[w].resolved) unres++;
                else { tc_min = std::min(tc_min, res[w].t_clean); tc_max = std::max(tc_max, res[w].t_clean); }
                tf_min = std::min(tf_min, res[w].t_flag);
            }
        }
        if (mode == "after") p += *bad;
        sum_stale += s;
        if (s) it_stale++;
        if (u) it_unexp++;
        if (p) it_prev++;
        if (tmo) it_timeout++;
        if ((s || u) && poll_ms > 0) {
            if (unres) it_unresolved++;
            else { lat_us.push_back((tc_max - tf_min) * tick_us); spread_us.push_back((tc_max - tc_min) * tick_us); }
        }
    }
    auto pct = [](std::vector<double> v, double q) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, (size_t)(q * v.size()))];
    };
    printf("RESULT mem=%s writer=%s mode=%s acq=%s load=%s intervene=%s prime=%d after_stream=%s resident=%d "
           "iters=%d stale_iters=%d unexpected_iters=%d prev_mismatch_iters=%d timeouts=%d avg_stale_words=%.0f/%zu",
           mem.c_str(), writer.c_str(), mode.c_str(), acqs.c_str(), loads.c_str(), intervene.c_str(), prime,
           after_stream.c_str(), after_resident, iters, it_stale, it_unexp, it_prev, it_timeout,
           it_stale ? (double)sum_stale / it_stale : 0.0, words);
    if (poll_ms > 0)
        printf(" poll_ms=%g unresolved_iters=%d resolved_iters=%zu lat_us[min/med/max]=%.1f/%.1f/%.1f "
               "wg_spread_us[med/max]=%.1f/%.1f max_rounds=%u",
               poll_ms, it_unresolved, lat_us.size(), pct(lat_us, 0), pct(lat_us, 0.5), pct(lat_us, 1.0),
               pct(spread_us, 0.5), pct(spread_us, 1.0), max_rounds);
    printf("\n");
    return 0;
}
