#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <string>
#include <thread>
#include <vector>

namespace
{
struct witness_state
{
    unsigned count;
    unsigned started;
    unsigned release;
    unsigned finished;
};

__global__ void
tiny_dispatch(witness_state* witness)
{
    atomicAdd_system(&witness->count, 1u);
}

__global__ void
unfinished_dispatch(witness_state* witness, uint64_t limit)
{
    atomicExch_system(&witness->started, 1u);
    const auto begin = clock64();
    while(atomicAdd_system(&witness->release, 0u) == 0 && clock64() - begin < limit)
        __builtin_amdgcn_s_sleep(1);
    atomicExch_system(&witness->finished, 1u);
}

void
hip_check(hipError_t status)
{
    if(status == hipSuccess) return;
    std::fprintf(stderr, "[flush-contract] FAIL HIP: %s\n", hipGetErrorString(status));
    std::exit(1);
}

template <typename Function>
Function
load(const char* name)
{
    auto function = reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
    if(function) return function;
    std::fprintf(stderr, "[flush-contract] FAIL missing client function: %s\n", name);
    std::exit(1);
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc != 3) return 2;
    int devices = 0;
    if(hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
    {
        std::puts("[flush-contract] SKIP no usable GPU");
        return 77;
    }
    hip_check(hipSetDevice(0));
    hipDeviceProp_t properties = {};
    hip_check(hipGetDeviceProperties(&properties, 0));
    // clockRate is deprecated and reports 0 on some runtime/device combinations. A zero limit
    // would make the "unfinished" kernel exit immediately and fail the case for the wrong reason,
    // so fall back to a nominal 1 GHz. The witness, not this bound, is what the test asserts on;
    // the bound only stops a wedged kernel from outliving the process.
    const auto clock_khz    = properties.clockRate > 0 ? static_cast<uint64_t>(properties.clockRate)
                                                       : static_cast<uint64_t>(1000 * 1000);
    const auto limit        = clock_khz * 1000 * 20;
    const bool stop_first   = std::string{argv[1]} == "stop-first";
    const auto scenario     = std::string{argv[2]};
    const bool unfinished   = scenario == "unfinished";
    const bool concurrent   = scenario == "concurrent";
    const int  windows      = scenario == "repeat" ? 3 : 1;
    auto       begin_window = load<int (*)(int, int)>("flush_contract_begin");
    auto       end_window   = load<int (*)(int, uint64_t)>("flush_contract_end");
    auto       poke_flush   = load<int (*)()>("flush_contract_poke");
    auto       flush_late   = load<int (*)(uint64_t)>("flush_contract_late");

    witness_state* host = nullptr;
    hip_check(hipHostMalloc(reinterpret_cast<void**>(&host),
                            sizeof(witness_state),
                            hipHostMallocMapped | hipHostMallocCoherent));
    void* device_pointer = nullptr;
    hip_check(hipHostGetDevicePointer(&device_pointer, host, 0));
    auto*       device = static_cast<witness_state*>(device_pointer);
    hipStream_t stream = nullptr;
    hipStream_t other  = nullptr;
    hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hip_check(hipStreamCreateWithFlags(&other, hipStreamNonBlocking));
    std::memset(host, 0, sizeof(*host));
    host->release = 1;
    tiny_dispatch<<<1, 1, 0, stream>>>(device);
    unfinished_dispatch<<<1, 1, 0, other>>>(device, limit);
    hip_check(hipDeviceSynchronize());

    int result = 0;
    for(int window = 0; window < windows; ++window)
    {
        std::memset(host, 0, sizeof(*host));
        if(begin_window(scenario == "delayed", scenario == "reentry") != 0)
        {
            result = 1;
            break;
        }
        auto watchdog_finished = std::promise<void>{};
        auto watchdog_ready    = watchdog_finished.get_future();
        auto watchdog_fired    = std::atomic<bool>{false};
        auto watchdog          = std::thread{};
        if(unfinished)
        {
            watchdog = std::thread{[&]() {
                if(watchdog_ready.wait_for(std::chrono::seconds{5}) == std::future_status::timeout)
                {
                    watchdog_fired.store(true);
                    __atomic_store_n(&host->release, 1u, __ATOMIC_RELEASE);
                }
            }};
            unfinished_dispatch<<<1, 1, 0, other>>>(device, limit);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
            while(__atomic_load_n(&host->started, __ATOMIC_ACQUIRE) == 0 &&
                  std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
        }
        // Background flushers, so the fence queue holds several requests at once while dispatches
        // are still completing. They are retired before the window's own flush so the record
        // count stays deterministic; what they exercise is the concurrent path, not the tail.
        auto pokers      = std::vector<std::thread>{};
        auto poking      = std::atomic<bool>{concurrent};
        auto poke_failed = std::atomic<bool>{false};
        for(int poker = 0; concurrent && poker < 4; ++poker)
            pokers.emplace_back([&]() {
                while(poking.load(std::memory_order_acquire))
                {
                    if(poke_flush() != 0) poke_failed.store(true, std::memory_order_release);
                    std::this_thread::sleep_for(std::chrono::milliseconds{1});
                }
            });

        const auto started = std::chrono::steady_clock::now();
        for(int burst = 0; burst < 2; ++burst)
        {
            std::this_thread::sleep_until(started + std::chrono::seconds{burst});
            for(int dispatch = 0; dispatch < 42; ++dispatch)
                tiny_dispatch<<<1, 1, 0, stream>>>(device);
            hip_check(hipStreamSynchronize(stream));
        }

        poking.store(false, std::memory_order_release);
        for(auto& poker : pokers)
            poker.join();
        if(poke_failed.load(std::memory_order_acquire))
        {
            std::puts("[flush-contract] FAIL concurrent flush reported an error");
            result = 1;
            break;
        }
        result = end_window(stop_first, 84);
        if(__atomic_load_n(&host->count, __ATOMIC_ACQUIRE) != 84) result = 1;
        if(unfinished)
        {
            if(__atomic_load_n(&host->started, __ATOMIC_ACQUIRE) == 0 ||
               __atomic_load_n(&host->finished, __ATOMIC_ACQUIRE) != 0 || watchdog_fired.load())
            {
                std::puts("[flush-contract] FAIL flush waited for unrelated unfinished work");
                result = 1;
            }
            __atomic_store_n(&host->release, 1u, __ATOMIC_RELEASE);
            watchdog_finished.set_value();
            watchdog.join();
            hip_check(hipStreamSynchronize(other));
            if(flush_late(85) != 0) result = 1;
        }
        if(result != 0) break;
    }
    hip_check(hipStreamDestroy(other));
    hip_check(hipStreamDestroy(stream));
    hip_check(hipHostFree(host));
    if(result == 0)
        std::printf("[flush-contract] PASS order=%s case=%s windows=%d expected=%d\n",
                    argv[1],
                    argv[2],
                    windows,
                    unfinished ? 85 : 84 * windows);
    return result;
}
