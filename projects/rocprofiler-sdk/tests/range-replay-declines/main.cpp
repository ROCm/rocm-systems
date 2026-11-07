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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// An application whose range writes device memory with a copy between its two kernels, which the
// SDK must decline, and which checks afterwards that its own result is intact.
//
// RR_DECLINE_MODE selects the copy:
//   hip-memcpy-async  hipMemcpyAsync host-to-device on the range's stream
//   hsa-batch-copy    hsa_amd_memory_async_batch_copy host-to-device, waited for on the host;
//   skipped
//                     where the HSA runtime does not provide it
//
// The range is bound by its first kernel before the copy runs: a copy can only decline a range that
// already belongs to an agent.

#include "range.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

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

__global__ void
rr_decline_first(int* acc)
{
    if(threadIdx.x == 0) *acc = (*acc * 3) + 1;
}

// Reads both ends of the copied buffer, so a run in which the copy had not landed computes a
// different number.
__global__ void
rr_decline_second(int* acc, const int* copied, size_t count)
{
    if(threadIdx.x == 0) *acc = (*acc * 3) + copied[0] + copied[count - 1];
}

namespace
{
constexpr size_t kCopyBytes = kCopyElements * sizeof(int);

int
skip(const char* reason)
{
    setenv(kSkipEnv, reason, 1);
    printf("[app] SKIP: %s\n", reason);
    return EXIT_SUCCESS;
}

#if defined(HSA_AMD_MEMORY_COPY_OP_VERSION)
using batch_copy_fn_t = hsa_status_t (*)(const hsa_amd_memory_copy_op_t*,
                                         uint32_t,
                                         uint32_t,
                                         const hsa_signal_t*);

hsa_status_t
find_cpu_agent(hsa_agent_t agent, void* data)
{
    auto type = hsa_device_type_t{};
    if(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS &&
       type == HSA_DEVICE_TYPE_CPU)
    {
        *static_cast<hsa_agent_t*>(data) = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

// One linear host-to-device copy through hsa_amd_memory_async_batch_copy, waited for on the host.
struct batch_copier
{
    batch_copy_fn_t copy   = nullptr;
    hsa_agent_t     cpu    = {};
    hsa_agent_t     gpu    = {};
    hsa_signal_t    signal = {};

    hsa_status_t operator()(void* dst, const void* src, size_t bytes) const
    {
        // The batch's signal must start at the number of operations in it.
        hsa_signal_store_screlease(signal, 1);

        // Every field the operation does not use, reserved ones included, must be zero.
        auto op = hsa_amd_memory_copy_op_t{};
        std::memset(&op, 0, sizeof(op));
        op.version           = HSA_AMD_MEMORY_COPY_OP_VERSION;
        op.type              = HSA_AMD_MEMORY_COPY_OP_LINEAR;
        op.completion_signal = signal;
        op.src               = const_cast<void*>(src);
        op.src_agent         = cpu;
        op.dst               = dst;
        op.dst_agent         = gpu;
        op.size              = bytes;

        if(auto status = copy(&op, 1, 0, nullptr); status != HSA_STATUS_SUCCESS) return status;

        hsa_signal_wait_scacquire(
            signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
        return HSA_STATUS_SUCCESS;
    }
};

// Resolves the batch copy entry point and the agents it needs, then performs one small copy outside
// any range, so a runtime that rejects the operation is found before the range opens. Returns a
// reason to skip, or nullptr.
const char*
prepare_batch_copy(batch_copier& copier, const void* device_buffer)
{
    copier.copy =
        reinterpret_cast<batch_copy_fn_t>(dlsym(RTLD_DEFAULT, "hsa_amd_memory_async_batch_copy"));
    if(copier.copy == nullptr) return "this HSA runtime has no hsa_amd_memory_async_batch_copy";

    auto info = hsa_amd_pointer_info_t{};
    info.size = sizeof(info);
    if(hsa_amd_pointer_info(device_buffer, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS)
        return "hsa_amd_pointer_info failed on a device allocation";
    copier.gpu = info.agentOwner;

    if(hsa_iterate_agents(find_cpu_agent, &copier.cpu) != HSA_STATUS_INFO_BREAK)
        return "no CPU agent";

    if(hsa_signal_create(1, 0, nullptr, &copier.signal) != HSA_STATUS_SUCCESS)
        return "hsa_signal_create failed";

    constexpr size_t kProbeBytes = 64;
    void*            probe_src   = nullptr;
    void*            probe_dst   = nullptr;
    if(hipHostMalloc(&probe_src, kProbeBytes) != hipSuccess ||
       hipMalloc(&probe_dst, kProbeBytes) != hipSuccess)
        return "could not allocate the probe buffers";

    const auto probe = copier(probe_dst, probe_src, kProbeBytes);
    (void) hipHostFree(probe_src);
    (void) hipFree(probe_dst);
    if(probe != HSA_STATUS_SUCCESS) return "hsa_amd_memory_async_batch_copy rejected a linear copy";

    return nullptr;
}
#endif
}  // namespace

int
main()
{
    auto device_count = 0;
    if(hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        return skip("no GPU device");

    const char* mode_env = std::getenv("RR_DECLINE_MODE");
    const auto  mode     = std::string_view{mode_env != nullptr ? mode_env : ""};
    if(mode != "hip-memcpy-async" && mode != "hsa-batch-copy")
    {
        fprintf(stderr, "[app] FAIL: unknown RR_DECLINE_MODE '%s'\n", mode_env ? mode_env : "");
        return EXIT_FAILURE;
    }

    auto* begin_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_begin)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_begin"));
    auto* end_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_end)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_end"));
    if(begin_fn == nullptr || end_fn == nullptr)
    {
        fprintf(stderr, "[app] FAIL: the range replay API is not in this process\n");
        return EXIT_FAILURE;
    }

    // A device allocation inside a range is itself a decline reason, so everything the range
    // touches is allocated and settled before it opens.
    int* acc     = nullptr;
    int* copied  = nullptr;
    int* pattern = nullptr;
    HIP_CHECK(hipMalloc(&acc, sizeof(int)));
    HIP_CHECK(hipMalloc(&copied, kCopyBytes));
    HIP_CHECK(hipHostMalloc(&pattern, kCopyBytes));
    for(size_t i = 0; i < kCopyElements; ++i)
        pattern[i] = copy_pattern(i);
    HIP_CHECK(hipMemset(acc, 0, sizeof(int)));
    HIP_CHECK(hipMemset(copied, 0, kCopyBytes));

#if defined(HSA_AMD_MEMORY_COPY_OP_VERSION)
    auto batch_copy = batch_copier{};
    if(mode == "hsa-batch-copy")
    {
        if(const char* reason = prepare_batch_copy(batch_copy, copied); reason != nullptr)
            return skip(reason);
    }
#else
    if(mode == "hsa-batch-copy")
        return skip("these HSA headers do not declare hsa_amd_memory_async_batch_copy");
#endif

    HIP_CHECK(hipDeviceSynchronize());

    if(const auto status = begin_fn(kRangeId); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: rocprofiler_range_replay_begin returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    rr_decline_first<<<1, 64>>>(acc);
    HIP_CHECK(hipGetLastError());

    if(mode == "hip-memcpy-async")
    {
        HIP_CHECK(hipMemcpyAsync(copied, pattern, kCopyBytes, hipMemcpyHostToDevice, nullptr));
    }
#if defined(HSA_AMD_MEMORY_COPY_OP_VERSION)
    else if(const auto status = batch_copy(copied, pattern, kCopyBytes);
            status != HSA_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: hsa_amd_memory_async_batch_copy returned status %d inside the range\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }
#endif

    rr_decline_second<<<1, 64>>>(acc, copied, kCopyElements);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if(const auto status = end_fn(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: rocprofiler_range_replay_end returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    // A decline only skips the replayed passes: the application must see exactly what its own
    // execution of the range produced, the copied buffer included.
    auto acc_h    = 0;
    auto copied_h = std::vector<int>(kCopyElements);
    HIP_CHECK(hipMemcpy(&acc_h, acc, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(copied_h.data(), copied, kCopyBytes, hipMemcpyDeviceToHost));

#if defined(HSA_AMD_MEMORY_COPY_OP_VERSION)
    if(batch_copy.signal.handle != 0) hsa_signal_destroy(batch_copy.signal);
#endif
    HIP_CHECK(hipHostFree(pattern));
    HIP_CHECK(hipFree(copied));
    HIP_CHECK(hipFree(acc));

    printf("[app] mode=%s acc=%d\n", mode_env, acc_h);
    if(acc_h != kExpectedResult)
    {
        fprintf(stderr, "[app] FAIL: acc=%d (expected %d)\n", acc_h, kExpectedResult);
        return EXIT_FAILURE;
    }
    for(size_t i = 0; i < kCopyElements; ++i)
    {
        if(copied_h[i] != copy_pattern(i))
        {
            fprintf(stderr,
                    "[app] FAIL: copied[%zu]=%d (expected %d)\n",
                    i,
                    copied_h[i],
                    copy_pattern(i));
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
