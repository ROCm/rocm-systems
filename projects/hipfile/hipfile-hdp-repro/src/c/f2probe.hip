// c-hypotheses Failure-2 discriminator (no hipFile).
// A running kernel primes (reads) a VRAM buffer, signals ready, spins on a host flag. The host
// overwrites the buffer through the large BAR (memcpy + sfence), optionally flushes HDP
// (HDP_MEM_COHERENCY_FLUSH_CNTL write + readback), then releases the flag. The kernel re-reads the
// buffer with the selected load flavour and counts stale words; one lane also measures how long a
// stale word stays stale while it keeps re-reading it (up to ~20 ms, s_memrealtime @ 100 MHz).
//
// env: MEM = fine | uncached | coarse   PRIME = 1|0   FLUSH = none | hdp | hdpreg
//      LOAD = plain | atomic_sys | nt    ITERS (20)   WORDS (16384 = 64 KiB)
// Build: amdclang++ -x hip --offload-arch=gfx1030 -O2 -std=c++20 f2probe.hip -o f2probe
#include <hip/hip_runtime.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Ctl {
    unsigned ready;
    unsigned go;
    unsigned stale_words;
    unsigned long long fresh_after_ticks;  // ~0 = never within budget
    unsigned long long sink;
};

template <int LOAD>
__device__ inline uint32_t dload(const uint32_t *p) {
    if constexpr (LOAD == 1) return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    else if constexpr (LOAD == 2) return __builtin_nontemporal_load(p);
    else return *p;
}

template <int LOAD>
__global__ void k(const uint32_t *buf, size_t words, Ctl *ctl, uint32_t expect, int prime) {
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    unsigned long long s = 0;
    if (prime)
        for (size_t i = gid; i < words; i += stride) s += buf[i];
    __syncthreads();
    if (threadIdx.x == 0) __hip_atomic_fetch_add(&ctl->ready, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    while (__hip_atomic_load(&ctl->go, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0) __builtin_amdgcn_s_sleep(1);
    unsigned stale = 0;
    for (size_t i = gid; i < words; i += stride) stale += dload<LOAD>(buf + i) != expect;
    if (stale) __hip_atomic_fetch_add(&ctl->stale_words, stale, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (gid == 0) {
        const unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
        unsigned long long dt = ~0ull;
        while (__builtin_amdgcn_s_memrealtime() - t0 < 2000000ull) {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);  // buffer_gl0_inv/gl1_inv at agent+ scope
            if (dload<LOAD>(buf) == expect) { dt = __builtin_amdgcn_s_memrealtime() - t0; break; }
        }
        ctl->fresh_after_ticks = dt;
    }
    if (s == 0x12345) ctl->sink = s;
}

static std::string env(const char *k, const char *d) { const char *v = std::getenv(k); return v ? v : d; }

int main() {
    const std::string mem = env("MEM", "fine"), flush = env("FLUSH", "none"), load = env("LOAD", "plain");
    const int prime = std::stoi(env("PRIME", "1")), iters = std::stoi(env("ITERS", "20"));
    const size_t words = std::stoul(env("WORDS", "16384"));
    hipDeviceProp_t prop{};
    (void)hipGetDeviceProperties(&prop, 0);
    volatile uint32_t *hdpMem = reinterpret_cast<volatile uint32_t *>(prop.hdpMemFlushCntl);
    volatile uint32_t *hdpReg = reinterpret_cast<volatile uint32_t *>(prop.hdpRegFlushCntl);
    uint32_t *buf{};
    if (mem == "fine") (void)hipExtMallocWithFlags((void **)&buf, words * 4, hipDeviceMallocFinegrained);
    else if (mem == "uncached") (void)hipExtMallocWithFlags((void **)&buf, words * 4, hipDeviceMallocUncached);
    else (void)hipMalloc(&buf, words * 4);
    Ctl *ctl{};
    (void)hipHostMalloc((void **)&ctl, sizeof(Ctl), hipHostMallocCoherent);
    hipStream_t st;
    (void)hipStreamCreateWithFlags(&st, hipStreamNonBlocking);
    std::vector<uint32_t> host(words);
    const int blocks = 72, threads = 256;
    int stale_iters = 0, never = 0;
    unsigned long long total_stale = 0, worst_ticks = 0;
    for (int it = 1; it <= iters; it++) {
        const uint32_t oldv = 0xA0000000u + it, newv = 0xB0000000u + it;
        std::fill(host.begin(), host.end(), oldv);
        (void)hipMemcpy(buf, host.data(), words * 4, hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
        std::memset(ctl, 0, sizeof(Ctl));
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (load == "atomic_sys") hipLaunchKernelGGL(k<1>, blocks, threads, 0, st, buf, words, ctl, newv, prime);
        else if (load == "nt") hipLaunchKernelGGL(k<2>, blocks, threads, 0, st, buf, words, ctl, newv, prime);
        else hipLaunchKernelGGL(k<0>, blocks, threads, 0, st, buf, words, ctl, newv, prime);
        while (__atomic_load_n(&ctl->ready, __ATOMIC_ACQUIRE) < unsigned(blocks)) _mm_pause();
        std::fill(host.begin(), host.end(), newv);
        std::memcpy(buf, host.data(), words * 4);  // CPU writes through the large BAR
        _mm_sfence();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (flush == "hdp") { *hdpMem = 1u; }  // node bundle: no read-back
        else if (flush == "hdpreg") { *hdpReg = 1u; }  // node bundle: no read-back
        __atomic_store_n(&ctl->go, 1u, __ATOMIC_RELEASE);
        (void)hipStreamSynchronize(st);
        if (ctl->stale_words) stale_iters++;
        total_stale += ctl->stale_words;
        if (ctl->fresh_after_ticks == ~0ull) never++;
        else if (ctl->fresh_after_ticks > worst_ticks) worst_ticks = ctl->fresh_after_ticks;
    }
    printf("MEM=%s PRIME=%d FLUSH=%s LOAD=%s WORDS=%zu: iters_with_stale=%d/%d avg_stale_words=%.0f "
           "word0_never_fresh_in_20ms=%d worst_fresh_after=%.1f us\n",
           mem.c_str(), prime, flush.c_str(), load.c_str(), words, stale_iters, iters,
           double(total_stale) / iters, never, worst_ticks / 100.0);
}
