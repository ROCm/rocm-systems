// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Host replay's fence timeout, on a GPU.
//
// A kernel on a second stream spins until the host releases it through fine-grained host memory.
// The first range opens while it spins. The fence at begin cannot get past that kernel, so with a
// short ROCPROFILER_RANGE_REPLAY_HOST_FENCE_TIMEOUT_MS the range declines with CONCURRENT_DISPATCH
// after about that long, runs once, and the application carries on. The range body releases the
// spinner before it launches anything, so none of its own work waits behind the spinner, even when
// HIP puts both streams on one HSA queue. The second range opens after the spinner returned and
// must replay as usual, every pass starting from the restored entry state.
//
// The spinner gives up by itself after kSpinLimit, so a fence that does not time out fails the test
// rather than hanging it.

#include "fence_timeout.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <setjmp.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (call);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[app] FAIL: HIP error '%s' at %s:%d\n",                                       \
                    hipGetErrorString(_err),                                                       \
                    __FILE__,                                                                      \
                    __LINE__);                                                                     \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while(0)

namespace
{
constexpr int kCount = 4096;
constexpr int kBlock = 256;

// What the spinner reports through spin_flags_t::state.
constexpr int kSpinIdle     = 0;
constexpr int kSpinStarted  = 1;
constexpr int kSpinReleased = 2;
constexpr int kSpinTimedOut = 3;

constexpr auto kSpinLimit  = std::chrono::seconds{20};
constexpr auto kStartLimit = std::chrono::seconds{10};

// In fine-grained host memory, accessed by the host and the spinner at system scope.
struct spin_flags_t
{
    int release;
    int state;
};

__global__ void
rr_fence_spin(spin_flags_t* flags, long long limit_ticks)
{
    __hip_atomic_store(&flags->state, kSpinStarted, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    const long long start = wall_clock64();
    while(__hip_atomic_load(&flags->release, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0)
    {
        if(wall_clock64() - start > limit_ticks)
        {
            __hip_atomic_store(
                &flags->state, kSpinTimedOut, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            return;
        }
        __builtin_amdgcn_s_sleep(127);
    }
    __hip_atomic_store(&flags->state, kSpinReleased, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void
rr_fence_accumulate(float* acc, const float* in)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < kCount) acc[i] += in[i];
}

struct resources_t
{
    hipStream_t                                       work        = nullptr;
    hipStream_t                                       spin        = nullptr;
    float*                                            in          = nullptr;
    float*                                            acc         = nullptr;
    float*                                            warm        = nullptr;
    float*                                            h_in        = nullptr;  // pinned
    float*                                            h_out       = nullptr;  // pinned
    spin_flags_t*                                     flags       = nullptr;  // fine-grained
    long long                                         limit_ticks = 0;
    decltype(&rocprofiler_range_replay_get_host_pass) get_pass    = nullptr;
};

// What each run of a range body saw. Not registered with the SDK, so it survives the rewinds.
int    g_blocked_runs       = 0;
int    g_blocked_spin_state = kSpinIdle;
bool   g_blocked_result_ok  = false;
double g_blocked_begin_secs = 0.0;
int    g_replayed_runs      = 0;
bool   g_pass_ok[kPasses]   = {};

int
spin_state(const resources_t& r)
{
    return __atomic_load_n(&r.flags->state, __ATOMIC_ACQUIRE);
}

void
set_spin_flags(const resources_t& r, int release, int state)
{
    __atomic_store_n(&r.flags->state, state, __ATOMIC_RELEASE);
    __atomic_store_n(&r.flags->release, release, __ATOMIC_RELEASE);
}

void
release_spinner(const resources_t& r)
{
    __atomic_store_n(&r.flags->release, 1, __ATOMIC_RELEASE);
}

bool
wait_for_spinner_start(const resources_t& r)
{
    const auto give_up = std::chrono::steady_clock::now() + kStartLimit;
    while(spin_state(r) == kSpinIdle)
    {
        if(std::chrono::steady_clock::now() > give_up) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return true;
}

// Whether the last read-back holds `factor` times the input.
bool
read_back_is(const resources_t& r, float factor)
{
    for(int i = 0; i < kCount; ++i)
        if(r.h_out[i] != factor * r.h_in[i]) return false;
    return true;
}

int
accumulate_and_read_back(const resources_t& r)
{
    rr_fence_accumulate<<<kCount / kBlock, kBlock, 0, r.work>>>(r.acc, r.in);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(
        hipMemcpyAsync(r.h_out, r.acc, kCount * sizeof(float), hipMemcpyDeviceToHost, r.work));
    HIP_CHECK(hipStreamSynchronize(r.work));
    return EXIT_SUCCESS;
}

// Until the spinner returns, work on its stream, and on any stream HIP placed on the same HSA
// queue, waits behind it.
int
blocked_range_body(const resources_t& r)
{
    release_spinner(r);
    HIP_CHECK(hipStreamSynchronize(r.spin));
    g_blocked_spin_state = spin_state(r);
    if(accumulate_and_read_back(r) != EXIT_SUCCESS) return EXIT_FAILURE;
    g_blocked_result_ok = read_back_is(r, 1.0f);
    ++g_blocked_runs;
    return EXIT_SUCCESS;
}

// Every pass must find the accumulator as the first range left it.
int
replayed_range_body(const resources_t& r)
{
    if(accumulate_and_read_back(r) != EXIT_SUCCESS) return EXIT_FAILURE;
    auto pass = uint64_t{0};
    r.get_pass(&pass);
    if(pass < static_cast<uint64_t>(kPasses)) g_pass_ok[pass] = read_back_is(r, 2.0f);
    ++g_replayed_runs;
    return EXIT_SUCCESS;
}

int
env_int(const char* name, int fallback)
{
    const char* value = getenv(name);
    return (value != nullptr && *value != '\0') ? atoi(value) : fallback;
}
}  // namespace

int
main()
{
    auto* begin_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_begin)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_begin"));
    auto* end_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_end)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_end"));
    auto* jmpbuf_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_host_jmpbuf)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_host_jmpbuf"));

    auto r     = resources_t{};
    r.get_pass = reinterpret_cast<decltype(&rocprofiler_range_replay_get_host_pass)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_get_host_pass"));

    if(begin_fn == nullptr || end_fn == nullptr || jmpbuf_fn == nullptr || r.get_pass == nullptr)
    {
        fprintf(stderr, "[app] FAIL: the range replay host API is not in this process\n");
        return EXIT_FAILURE;
    }

    const double fence_timeout_secs =
        env_int("ROCPROFILER_RANGE_REPLAY_HOST_FENCE_TIMEOUT_MS", 0) / 1000.0;
    if(fence_timeout_secs <= 0.0)
    {
        fprintf(stderr, "[app] FAIL: ROCPROFILER_RANGE_REPLAY_HOST_FENCE_TIMEOUT_MS is not set\n");
        return EXIT_FAILURE;
    }

    auto device    = 0;
    auto clock_khz = 0;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeWallClockRate, device));
    if(clock_khz <= 0)
    {
        fprintf(stderr, "[app] FAIL: the device reports no wall clock rate\n");
        return EXIT_FAILURE;
    }
    r.limit_ticks = static_cast<long long>(clock_khz) * 1000 * kSpinLimit.count();

    constexpr auto bytes = kCount * sizeof(float);
    HIP_CHECK(hipStreamCreateWithFlags(&r.work, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&r.spin, hipStreamNonBlocking));
    HIP_CHECK(hipMalloc(&r.in, bytes));
    HIP_CHECK(hipMalloc(&r.acc, bytes));
    HIP_CHECK(hipMalloc(&r.warm, bytes));
    HIP_CHECK(hipHostMalloc(&r.h_in, bytes, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&r.h_out, bytes, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&r.flags, sizeof(spin_flags_t), hipHostMallocCoherent));
    for(int i = 0; i < kCount; ++i)
        r.h_in[i] = static_cast<float>(1 + i % 17);
    HIP_CHECK(hipMemcpyAsync(r.in, r.h_in, bytes, hipMemcpyHostToDevice, r.work));
    HIP_CHECK(hipMemsetAsync(r.acc, 0, bytes, r.work));
    HIP_CHECK(hipMemsetAsync(r.warm, 0, bytes, r.work));

    // Warm-up outside the ranges: HIP loads the module on the first launch of one of its kernels
    // and gives a stream its HSA queue on first use. Released in advance, this spinner returns at
    // once.
    rr_fence_accumulate<<<kCount / kBlock, kBlock, 0, r.work>>>(r.warm, r.in);
    set_spin_flags(r, 1, kSpinIdle);
    rr_fence_spin<<<1, 1, 0, r.spin>>>(r.flags, r.limit_ticks);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    if(spin_state(r) != kSpinReleased)
    {
        fprintf(stderr, "[app] FAIL: the warm-up spinner did not run (state %d)\n", spin_state(r));
        return EXIT_FAILURE;
    }

    // From here until the first range's body releases it, nothing may wait for the spinner.
    set_spin_flags(r, 0, kSpinIdle);
    rr_fence_spin<<<1, 1, 0, r.spin>>>(r.flags, r.limit_ticks);
    HIP_CHECK(hipGetLastError());
    if(!wait_for_spinner_start(r))
    {
        release_spinner(r);
        fprintf(stderr,
                "[app] FAIL: the spinner did not start within %lld s\n",
                static_cast<long long>(kStartLimit.count()));
        return EXIT_FAILURE;
    }

    // ROCPROFILER_RANGE_REPLAY_HOST_BEGIN, spelled out because the SDK is resolved via dlsym. This
    // range should decline at begin and never jump back; the checks below catch it if it does.
    const auto    blocked_begin  = std::chrono::steady_clock::now();
    volatile auto blocked_status = ROCPROFILER_STATUS_SUCCESS;
    if(sigsetjmp(*static_cast<sigjmp_buf*>(jmpbuf_fn()), 1) == 0)
        blocked_status = begin_fn(kBlockedRangeId);
    g_blocked_begin_secs =
        std::chrono::duration<double>{std::chrono::steady_clock::now() - blocked_begin}.count();
    if(blocked_status != ROCPROFILER_STATUS_SUCCESS)
    {
        release_spinner(r);
        fprintf(stderr,
                "[app] FAIL: begin of the blocked range returned status %d\n",
                static_cast<int>(blocked_status));
        return EXIT_FAILURE;
    }

    if(blocked_range_body(r) != EXIT_SUCCESS) return EXIT_FAILURE;

    if(const auto status = end_fn(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: end of the blocked range returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    HIP_CHECK(hipDeviceSynchronize());

    volatile auto replayed_status = ROCPROFILER_STATUS_SUCCESS;
    if(sigsetjmp(*static_cast<sigjmp_buf*>(jmpbuf_fn()), 1) == 0)
        replayed_status = begin_fn(kReplayedRangeId);
    if(replayed_status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: begin of the replayed range returned status %d\n",
                static_cast<int>(replayed_status));
        return EXIT_FAILURE;
    }

    if(replayed_range_body(r) != EXIT_SUCCESS) return EXIT_FAILURE;

    if(const auto status = end_fn(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: end of the replayed range returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    // Closed, each range has the effect of one run: one accumulation each.
    HIP_CHECK(hipMemcpy(r.h_out, r.acc, bytes, hipMemcpyDeviceToHost));
    const bool final_ok = read_back_is(r, 2.0f);

    printf("[app] blocked range: runs=%d begin=%.3f s spinner=%d result_ok=%d; replayed range: "
           "runs=%d final_ok=%d\n",
           g_blocked_runs,
           g_blocked_begin_secs,
           g_blocked_spin_state,
           static_cast<int>(g_blocked_result_ok),
           g_replayed_runs,
           static_cast<int>(final_ok));

    auto ok = true;
    if(g_blocked_runs != 1)
    {
        fprintf(
            stderr, "[app] FAIL: the blocked range ran %d times, expected once\n", g_blocked_runs);
        ok = false;
    }
    if(g_blocked_spin_state != kSpinReleased)
    {
        fprintf(stderr,
                "[app] FAIL: the spinner was not released by the blocked range (state %d, %d = "
                "gave up after %lld s)\n",
                g_blocked_spin_state,
                kSpinTimedOut,
                static_cast<long long>(kSpinLimit.count()));
        ok = false;
    }
    if(!g_blocked_result_ok)
    {
        fprintf(stderr, "[app] FAIL: the blocked range read back a wrong accumulator\n");
        ok = false;
    }
    // At least most of the timeout: the fence waited for the spinner rather than declining for some
    // other reason. Well under the spinner's own limit: the fence gave up, not the spinner.
    if(g_blocked_begin_secs < 0.5 * fence_timeout_secs ||
       g_blocked_begin_secs > fence_timeout_secs + 5.0)
    {
        fprintf(stderr,
                "[app] FAIL: begin of the blocked range took %.3f s, expected about the %.3f s "
                "fence timeout\n",
                g_blocked_begin_secs,
                fence_timeout_secs);
        ok = false;
    }
    if(g_replayed_runs != kPasses)
    {
        fprintf(stderr,
                "[app] FAIL: the replayed range ran %d times, expected %d\n",
                g_replayed_runs,
                kPasses);
        ok = false;
    }
    for(int pass = 0; pass < kPasses; ++pass)
    {
        if(g_pass_ok[pass]) continue;
        fprintf(stderr,
                "[app] FAIL: pass %d of the replayed range read back a wrong accumulator\n",
                pass);
        ok = false;
    }
    if(!final_ok)
    {
        fprintf(stderr,
                "[app] FAIL: after both ranges the accumulator does not hold one run of each\n");
        ok = false;
    }

    HIP_CHECK(hipHostFree(r.flags));
    HIP_CHECK(hipHostFree(r.h_out));
    HIP_CHECK(hipHostFree(r.h_in));
    HIP_CHECK(hipFree(r.warm));
    HIP_CHECK(hipFree(r.acc));
    HIP_CHECK(hipFree(r.in));
    HIP_CHECK(hipStreamDestroy(r.spin));
    HIP_CHECK(hipStreamDestroy(r.work));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
