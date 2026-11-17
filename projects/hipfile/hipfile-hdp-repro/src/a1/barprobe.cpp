// barprobe: CPU reads of VRAM through the large BAR after a GPU-engine write (Failure 1 probe).
// Derived from hip-bar-read-stale.cpp. Build on rowlet:
//   amdclang++ -x hip --offload-arch=gfx1030 -O2 -std=c++20 -DA1_WITH_HSA barprobe.cpp -o barprobe \
//     -L/opt/rocm/lib -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib -lpthread
// Env:
//   SIZE=65536 ITERS=300 READERS=7 MEM=coarse|fine|uncached
//   WRITER=memcpy (pageable mmap src, hipMemcpy H2D; default) | pinned (hipHostMalloc src, hipMemcpy)
//          | async (pageable, hipMemcpyAsync null stream + hipStreamSynchronize(0))
//          | asyncpinned | cpubar (CPU memcpy into BAR + sfence) | memsetd32 (hipMemsetD32)
//          | kernel (fill kernel on null stream + hipStreamSynchronize(0))
//          | hsasdma (pinned src, hsa_amd_memory_async_copy_on_engine SDMA0, wait signal)
//   PRE=tok,... POST=tok,...   (see interventions.h)
//   CHECK=volatile|clflush|clflushopt|nt|mfence|lfence   reader-side load for the check & lateness spin
//   RLOAD=volatile|nt|clflush   load used by the reader threads
//   PREREAD=1 main thread reads the check line before the write (as the original repro)
//   WAITALL=1 after the check, wait (<=1 s) until every reader has seen the new value
//   DUMP=path  per-iteration CSV (it,done_ns,stale,main_visible_ns,reader seen ns...)
//   MAIN_CPU=0 READER_CPU0=1 (reader r pinned to READER_CPU0+r)
//   LABEL=text
#include "interventions.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

static long envl(const char *n, long d) {
    const char *s = std::getenv(n);
    if (!s) return d;
    char *e{};
    long v = std::strtol(s, &e, 0);
    if (*e == 'K' || *e == 'k') v <<= 10;
    if (*e == 'M' || *e == 'm') v <<= 20;
    return v;
}
static std::string envs(const char *n, const char *d) { const char *s = std::getenv(n); return s ? s : d; }

static void pin(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    if (pthread_setaffinity_np(pthread_self(), sizeof s, &s)) fprintf(stderr, "pin %d failed\n", cpu);
}

__global__ void fill_kernel(uint32_t *p, size_t n, uint32_t v) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) p[i] = v;
}

enum class Load { Volatile, Clflush, Clflushopt, Nt, Mfence, Lfence };
static Load parse_load(const std::string &s) {
    if (s == "clflush") return Load::Clflush;
    if (s == "clflushopt") return Load::Clflushopt;
    if (s == "nt") return Load::Nt;
    if (s == "mfence") return Load::Mfence;
    if (s == "lfence") return Load::Lfence;
    return Load::Volatile;
}
__attribute__((target("sse4.1,clflushopt"))) static inline uint32_t load(volatile uint32_t *p, Load m) {
    switch (m) {
    case Load::Clflush: _mm_clflush((const void *)p); _mm_mfence(); return *p;
    case Load::Clflushopt: _mm_clflushopt((void *)p); _mm_mfence(); return *p;
    case Load::Mfence: _mm_mfence(); return *p;
    case Load::Lfence: _mm_lfence(); return *p;
    case Load::Nt: {
        auto a = reinterpret_cast<uintptr_t>(p);
        __m128i v = _mm_stream_load_si128(reinterpret_cast<__m128i *>(a & ~uintptr_t(15)));
        uint32_t w[4];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(w), v);
        return w[(a & 15) / 4];
    }
    default: return *p;
    }
}

int main() {
    const size_t size = envl("SIZE", 65536);
    const long iters = envl("ITERS", 300);
    const int nreaders = int(envl("READERS", 7));
    const std::string writer = envs("WRITER", "memcpy");
    const std::string mem = envs("MEM", "coarse");
    const auto pre = a1::split(std::getenv("PRE"));
    const auto post = a1::split(std::getenv("POST"));
    const Load check = parse_load(envs("CHECK", "volatile"));
    const Load rload = parse_load(envs("RLOAD", "volatile"));
    const bool preread = envl("PREREAD", 1) != 0;
    const bool waitall = envl("WAITALL", 0) != 0;
    const int main_cpu = int(envl("MAIN_CPU", 0));
    const int rcpu0 = int(envl("READER_CPU0", 1));
    const std::string dump = envs("DUMP", "");
    const std::string label = envs("LABEL", "");
    pin(main_cpu);

    a1::Ctx c;
    a1::init(c, envl("VERBOSE", 0) != 0);

    void *dev{};
    if (mem == "fine") (void)hipExtMallocWithFlags(&dev, size, hipDeviceMallocFinegrained);
    else if (mem == "uncached") (void)hipExtMallocWithFlags(&dev, size, hipDeviceMallocUncached);
    else (void)hipMalloc(&dev, size);
    (void)hipMemset(dev, 0, size);
    (void)hipDeviceSynchronize();
    if (envl("VERBOSE", 0)) {
        std::ifstream maps("/proc/self/maps");
        std::string line;
        auto d = reinterpret_cast<uintptr_t>(dev);
        while (std::getline(maps, line)) {
            uintptr_t lo = std::stoull(line.substr(0, line.find('-')), nullptr, 16);
            uintptr_t hi = std::stoull(line.substr(line.find('-') + 1), nullptr, 16);
            if (d >= lo && d < hi) fprintf(stderr, "# dev %p in map: %s\n", dev, line.c_str());
        }
    }
    const size_t n = size / 4;
    uint32_t *host{};
    const bool pinned_src = writer == "pinned" || writer == "asyncpinned" || writer == "hsasdma";
    if (pinned_src) (void)hipHostMalloc(reinterpret_cast<void **>(&host), size, hipHostMallocDefault);
    else host = static_cast<uint32_t *>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    auto *dv = static_cast<volatile uint32_t *>(dev);

    // LINE_STRIDE (bytes, optional): reader r reads LINE_BASE + r*LINE_STRIDE instead of spreading
    // the readers evenly across the buffer. LINE_BASE (bytes) may be negative (from the end).
    const long line_stride = envl("LINE_STRIDE", 0), line_base = envl("LINE_BASE", 0);
    auto line_of = [&](int r) {
        if (line_stride) return size_t((line_base < 0 ? long(size) + line_base : line_base) + r * line_stride) / 4 % n / 16 * 16;
        return size_t(r) * n / size_t(nreaders) / 16 * 16;
    };
    const long slots = iters + 2;
    std::vector<std::atomic<uint64_t>> seen(size_t(std::max(nreaders, 1)) * slots);
    for (auto &s : seen) s.store(0, std::memory_order_relaxed);
    std::vector<uint64_t> done_ns(slots, 0), vis_ns(slots, 0);
    std::vector<char> stale_v(slots, 0);

    std::atomic<bool> stop{false};
    std::atomic<int> ready{0};
    std::vector<std::thread> readers;
    for (int r = 0; r < nreaders; r++) {
        readers.emplace_back([&, r] {
            pin(rcpu0 + r);
            volatile uint32_t *p = dv + line_of(r);
            uint32_t last = 0xffffffffu;
            ready.fetch_add(1);
            while (!stop.load(std::memory_order_relaxed)) {
                uint32_t v = load(p, rload);
                if (v != last) {
                    last = v;
                    if (v >= 1 && v < uint32_t(slots)) {
                        auto &s = seen[size_t(r) * slots + v];
                        if (s.load(std::memory_order_relaxed) == 0) s.store(a1::now_ns(), std::memory_order_release);
                    }
                }
            }
        });
    }
    while (ready.load() < nreaders) {}

    int stale = 0, weird = 0, never = 0;
    long worst = 0;
    std::vector<long> lates;
    const uint64_t t_start = a1::now_ns();
    for (uint32_t it = 1; it <= uint32_t(iters); it++) {
        // CHECK_OFFSET (bytes): check a line next to a reader's line instead of the reader's own line
        const size_t idx = ((nreaders ? line_of(int(it % nreaders)) : (size_t(it) * 7919 * 16) % n / 16 * 16) +
                            size_t(envl("CHECK_OFFSET", 0)) / 4) % n;
        if (preread) (void)dv[idx];
        for (size_t i = 0; i < n; i++) host[i] = it;
        a1::apply_all(c, pre);
        if (writer == "memcpy" || writer == "pinned") (void)hipMemcpy(dev, host, size, hipMemcpyHostToDevice);
        else if (writer == "async" || writer == "asyncpinned") { (void)hipMemcpyAsync(dev, host, size, hipMemcpyHostToDevice, 0); (void)hipStreamSynchronize(0); }
        else if (writer == "cpubar") { std::memcpy(dev, host, size); _mm_sfence(); }
        else if (writer == "memsetd32") (void)hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(dev), int(it), n);
        else if (writer == "kernel") { fill_kernel<<<std::min<size_t>((n + 255) / 256, 1024), 256>>>(static_cast<uint32_t *>(dev), n, it); (void)hipStreamSynchronize(0); }
        else if (writer == "hsasdma") a1::hsa_sdma_copy(c, dev, host, size, true);
        else { fprintf(stderr, "bad WRITER\n"); return 2; }
        a1::apply_all(c, post);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const uint64_t t_done = a1::now_ns();
        done_ns[it] = t_done;
        uint32_t v = load(dv + idx, check);
        if (v != it) {
            if (v != it - 1 && !(it == 1 && v == 0)) weird++;
            stale++;
            stale_v[it] = 1;
            while ((v = load(dv + idx, check)) != it && a1::now_ns() - t_done < 1000000000ull)
                std::atomic_thread_fence(std::memory_order_seq_cst);
            const uint64_t t1 = a1::now_ns();
            vis_ns[it] = t1;
            if (v != it) never++;
            long late = long(t1 - t_done);
            lates.push_back(late);
            worst = std::max(worst, late);
        } else vis_ns[it] = t_done;
        if (waitall) {
            for (int r = 0; r < nreaders; r++)
                while (seen[size_t(r) * slots + it].load(std::memory_order_acquire) == 0 &&
                       a1::now_ns() - t_done < 1000000000ull) {}
        }
    }
    const double secs = double(a1::now_ns() - t_start) / 1e9;
    stop = true;
    for (auto &t : readers) t.join();

    // Reader-side stats (meaningful with WAITALL=1): a reader is "late" if it first saw value `it`
    // more than 20 us after the write+POST returned
    long any_late = 0, max_spread = 0;
    for (long it = 1; it <= iters && waitall; it++) {
        bool late = false;
        uint64_t mn = UINT64_MAX, mx = 0;
        for (int r = 0; r < nreaders; r++) {
            uint64_t s = seen[size_t(r) * slots + it].load();
            if (s == 0 || s > done_ns[it] + 20000) {
                late = true;
                if (s) { mn = std::min(mn, s); mx = std::max(mx, s); }
            }
        }
        if (late) { any_late++; if (mx >= mn && mn != UINT64_MAX) max_spread = std::max(max_spread, long(mx - mn)); }
    }
    std::sort(lates.begin(), lates.end());
    auto pct = [&](double p) { return lates.empty() ? 0L : lates[size_t(p * double(lates.size() - 1))] / 1000; };
    printf("%-28s size=%zu R=%d W=%s PRE=%s POST=%s CHECK=%s: stale %d/%ld, worst %ld us, p50 %ld us, p90 %ld us",
           label.c_str(), size, nreaders, writer.c_str(), envs("PRE", "-").c_str(), envs("POST", "-").c_str(),
           envs("CHECK", "volatile").c_str(), stale, iters, worst / 1000, pct(0.5), pct(0.9));
    if (weird) printf(", weird %d", weird);
    if (never) printf(", NEVER %d", never);
    if (waitall) printf(", readers-late-iters %ld, max late-reader spread %ld us", any_late, max_spread / 1000);
    printf(" (%.1fs)\n", secs);
    if (!dump.empty()) {
        FILE *f = fopen(dump.c_str(), "w");
        for (long it = 1; it <= iters; it++) {
            fprintf(f, "%ld,%lu,%d,%lu", it, (unsigned long)done_ns[it], stale_v[it], (unsigned long)vis_ns[it]);
            for (int r = 0; r < nreaders; r++) fprintf(f, ",%lu", (unsigned long)seen[size_t(r) * slots + it].load());
            fprintf(f, "\n");
        }
        fclose(f);
    }
    return 0;
}
