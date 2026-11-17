// c-hypotheses Failure-1 discriminator (no hipFile).
// READERS CPU threads spin-read one 64B line each of a VRAM buffer through the large BAR while the
// main thread does hipMemcpy H2D (pageable) + hipDeviceSynchronize, then an optional "KICK" action,
// then reads every reader line once and counts which ones are stale. Lateness of the first stale
// line is measured by spinning until it flips (or 2 s timeout).
//
// env: SIZE bytes (65536)  READERS (7)  ITERS (200)
//      KICK = none | memset | emptykernel | barwrite | barwrite_rb | barread | hdpflush | hdpreg |
//             sleep_us:<N>
//      READ = plain | clflush | ntload    (how readers AND the checker load)
//      SAMELINE=1 : the KICK barwrite targets the first reader line of the *test* buffer (rewrites
//                   the value it already should have) instead of an unrelated buffer
//      STOPREAD=1 : readers stop during the copy (pause before, resume after the sync)
// Build: amdclang++ -x hip --offload-arch=gfx1030 -O2 -std=c++20 -msse4.1 f1probe.hip -o f1probe
#include <hip/hip_runtime.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

__global__ void empty_kernel() {}

static std::string env(const char *k, const char *d) {
    const char *v = std::getenv(k);
    return v ? v : d;
}
enum class Rd { plain, clflush, ntload };
static Rd rdmode;
static inline uint32_t ld(volatile uint32_t *p) {
    switch (rdmode) {
    case Rd::clflush:
        _mm_clflush((const void *)p);
        _mm_mfence();
        return *p;
    case Rd::ntload: {
        __m128i v = _mm_stream_load_si128((__m128i *)(uintptr_t(p) & ~uintptr_t(15)));
        uint32_t out[4];
        _mm_storeu_si128((__m128i *)out, v);
        return out[(uintptr_t(p) & 15) / 4];
    }
    default:
        return *p;
    }
}

int main() {
    const size_t size = std::stoul(env("SIZE", "65536"));
    const int R = std::stoi(env("READERS", "7"));
    const int iters = std::stoi(env("ITERS", "200"));
    const std::string kick = env("KICK", "none");
    const std::string rd = env("READ", "plain");
    const bool sameline = env("SAMELINE", "0") == "1";
    const bool stopread = env("STOPREAD", "0") == "1";
    rdmode = rd == "clflush" ? Rd::clflush : rd == "ntload" ? Rd::ntload : Rd::plain;

    hipDeviceProp_t prop{};
    (void)hipGetDeviceProperties(&prop, 0);
    volatile uint32_t *hdpMem = reinterpret_cast<volatile uint32_t *>(prop.hdpMemFlushCntl);
    volatile uint32_t *hdpReg = reinterpret_cast<volatile uint32_t *>(prop.hdpRegFlushCntl);

    void *other{};
    (void)hipMalloc(&other, 1 << 20);
    (void)hipMemset(other, 0, 1 << 20);
    (void)hipDeviceSynchronize();
    void *dev{};
    (void)hipMalloc(&dev, size);
    (void)hipMemset(dev, 0, size);
    (void)hipDeviceSynchronize();
    auto *host = static_cast<uint32_t *>(
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    auto *dv = static_cast<volatile uint32_t *>(dev);
    auto *ov = static_cast<volatile uint32_t *>(other);
    const size_t n = size / 4;
    std::vector<size_t> line(R);
    for (int r = 0; r < R; r++) line[r] = size_t(r) * n / size_t(R) / 16 * 16;

    std::atomic<bool> stop{false};
    std::atomic<int> pause{0};
    std::vector<std::atomic<uint64_t>> nreads(R);
    std::vector<std::thread> th;
    auto t_start = std::chrono::steady_clock::now();
    for (int r = 0; r < R; r++) {
        th.emplace_back([&, r] {
            uint64_t c = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                if (pause.load(std::memory_order_relaxed)) continue;
                (void)ld(dv + line[r]);
                c++;
            }
            nreads[r] = c;
        });
    }
    std::vector<int> stale_by_line(R, 0);
    int iters_with_stale = 0;
    std::vector<long> late_us;
    for (uint32_t it = 1; it <= uint32_t(iters); it++) {
        for (size_t i = 0; i < n; i++) host[i] = it;
        if (stopread) {
            pause = 1;
            usleep(200);
        }
        (void)hipMemcpy(dev, host, size, hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
        if (stopread) pause = 0;
        if (kick == "memset") {
            (void)hipMemset(other, 0, 4);
            (void)hipDeviceSynchronize();
        } else if (kick == "emptykernel") {
            empty_kernel<<<1, 1>>>();
            (void)hipDeviceSynchronize();
        } else if (kick == "barwrite") {
            if (sameline) dv[line[0]] = it; else ov[64] = it;
            _mm_sfence();
        } else if (kick == "barwrite_rb") {
            if (sameline) { dv[line[0]] = it; _mm_mfence(); (void)dv[line[0]]; }
            else { ov[64] = it; _mm_mfence(); (void)ov[64]; }
        } else if (kick == "barread") {
            (void)ov[1024];
        } else if (kick == "hdpflush") {
            *hdpMem = 1u;  // node bundle: read-back removed (upstream amdgpu a5cb344033c7)
        } else if (kick == "hdpflush_nr") {
            // write only, no read-back of the flush register (upstream a5cb344033c7 removed the
            // read-back after AER storms); PCIe ordering puts the checker's later reads behind it
            *hdpMem = 1u;
            _mm_mfence();
        } else if (kick == "eventrecord") {
            static hipEvent_t ev = [] { hipEvent_t e; (void)hipEventCreate(&e); return e; }();
            (void)hipEventRecord(ev, nullptr);
            (void)hipEventSynchronize(ev);
        } else if (kick == "sdma") {
            static void *stg = [&] { void *p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); return p; }();
            (void)hipMemcpy(static_cast<char *>(other) + (512 << 10), stg, size, hipMemcpyHostToDevice);
            (void)hipDeviceSynchronize();
        } else if (kick == "hdpreg") {
            *hdpReg = 1u;  // node bundle: read-back removed (upstream amdgpu a5cb344033c7)
        } else if (kick.rfind("sleep_us:", 0) == 0) {
            usleep(std::stoi(kick.substr(9)));
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int first = -1;
        for (int r = 0; r < R; r++) {
            if (ld(dv + line[r]) != it) {
                stale_by_line[r]++;
                if (first < 0) first = r;
            }
        }
        if (first >= 0) {
            iters_with_stale++;
            auto t0 = std::chrono::steady_clock::now();
            while (ld(dv + line[first]) != it) {
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) break;
            }
            late_us.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count());
        }
    }
    stop = true;
    for (auto &t : th) t.join();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    uint64_t tot = 0;
    for (auto &c : nreads) tot += c;
    printf("SIZE=%zu READERS=%d ITERS=%d KICK=%s READ=%s SAMELINE=%d STOPREAD=%d hdpMem=%p\n", size,
           R, iters, kick.c_str(), rd.c_str(), sameline, stopread, (void *)hdpMem);
    printf("  iters_with_stale=%d/%d  avg BAR read latency per reader=%.0f ns\n", iters_with_stale,
           iters, secs * 1e9 * R / double(tot ? tot : 1));
    printf("  stale_by_line(offset KiB:count):");
    for (int r = 0; r < R; r++) printf(" %zu:%d", line[r] * 4 / 1024, stale_by_line[r]);
    printf("\n");
    if (!late_us.empty()) {
        std::sort(late_us.begin(), late_us.end());
        printf("  lateness us: min=%ld p50=%ld p90=%ld max=%ld\n", late_us.front(),
               late_us[late_us.size() / 2], late_us[late_us.size() * 9 / 10], late_us.back());
    }
}
