// HIP-only repro: CPU reads a VRAM line through the large BAR, hipMemcpy H2D
// overwrites it, hipDeviceSynchronize, CPU reads the line again.
#include <hip/hip_runtime_api.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <sys/mman.h>
#include <thread>
#include <vector>
int main() {
    const bool kick = std::getenv("KICK") != nullptr;
    void *other{};
    (void)hipMalloc(&other, 4096);
    for (size_t size : {size_t{64} << 10, size_t{1} << 20, size_t{4} << 20}) {
        void *dev{};
        // MEM=fine: fine-grained device memory; MEM=uncached; default: hipMalloc (coarse-grained)
        const char *mem = std::getenv("MEM");
        if (mem && std::string_view{mem} == "fine") (void)hipExtMallocWithFlags(&dev, size, hipDeviceMallocFinegrained);
        else if (mem && std::string_view{mem} == "uncached") (void)hipExtMallocWithFlags(&dev, size, hipDeviceMallocUncached);
        else (void)hipMalloc(&dev, size);
        auto *host = static_cast<uint32_t *>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        auto *dv = static_cast<volatile uint32_t *>(dev);
        const size_t n = size / 4;
        int stale = 0; long worst = 0;
        // Reader threads that keep reading lines spread across the buffer through the BAR, like the
        // PR's watchers, including while hipMemcpy runs
        std::atomic<bool> stop{false};
        std::vector<std::thread> readers;
        const int nreaders = std::getenv("READERS") ? std::atoi(std::getenv("READERS")) : 0;
        for (int r = 0; r < nreaders; r++) {
            readers.emplace_back([&, r] {
                const size_t line = size_t(r) * n / size_t(nreaders) / 16 * 16;
                while (!stop.load(std::memory_order_relaxed)) { (void)dv[line]; }
            });
        }
        for (uint32_t it = 1; it <= 300; it++) {
            const size_t idx = nreaders ? size_t(it % nreaders) * n / size_t(nreaders) / 16 * 16
                                        : (size_t(it) * 7919 * 16) % n / 16 * 16;
            (void)dv[idx];  // CPU read through the BAR before the copy
            for (size_t i = 0; i < n; i++) host[i] = it;
            (void)hipMemcpy(dev, host, size, hipMemcpyHostToDevice);
            (void)hipDeviceSynchronize();
            if (kick) { (void)hipMemset(other, 0, 4); (void)hipDeviceSynchronize(); }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (dv[idx] != it) {
                stale++;
                auto t0 = std::chrono::steady_clock::now();
                while (dv[idx] != it) std::atomic_thread_fence(std::memory_order_seq_cst);
                long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
                worst = ns > worst ? ns : worst;
            }
        }
        stop = true;
        for (auto &t : readers) t.join();
        printf("%8zu bytes: %3d/300 stale after hipMemcpy + hipDeviceSynchronize, worst %ld us\n", size, stale, worst / 1000);
        munmap(host, size); (void)hipFree(dev);
    }
}
