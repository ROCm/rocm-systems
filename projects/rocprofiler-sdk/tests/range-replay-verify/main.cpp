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

// A range whose kernels a replay can reproduce only from the range-entry snapshot, replayed with
// verification on. Afterwards the application checks every element it computed against the same
// computation on the host.
//
// RR_VERIFY_MODE selects the range:
//   in-place          three kernels, each rewriting every element of a 16 MiB buffer from its own
//                     value (x = 3x + k for k = 1, 2, 3). A step applied to its own output gives a
//                     different number, and the steps do not commute.
//   atomic-histogram  a 256-bin histogram of a 16 MiB buffer built with global atomicAdd into bins
//                     zeroed before the range, then a kernel that adds each element's bin count to
//                     it in place. Thousands of threads increment each bin in an order that changes
//                     from run to run, but integer addition commutes, so the counts do not.
//
// The buffer is filled by a kernel on the range's stream before the range opens, so the module is
// loaded and the stream has its queue by then, and the entry snapshot is the only thing that can
// put the buffer and the bins back between passes.

#include "range.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// lowbias32 (Chris Wellons): a well-mixed value per index, so the bins hold different counts and an
// element computed from the wrong input stands out.
__host__ __device__ constexpr uint32_t
rr_verify_pattern(uint32_t i)
{
    i ^= i >> 16;
    i *= 0x7feb352dU;
    i ^= i >> 15;
    i *= 0x846ca68bU;
    i ^= i >> 16;
    return i;
}

__global__ void
rr_verify_fill(uint32_t* data, uint32_t count)
{
    const uint32_t stride = blockDim.x * gridDim.x;
    for(uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < count; i += stride)
        data[i] = rr_verify_pattern(i);
}

__global__ void
rr_verify_step(uint32_t* data, uint32_t count, uint32_t add)
{
    const uint32_t stride = blockDim.x * gridDim.x;
    for(uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < count; i += stride)
        data[i] = (data[i] * 3U) + add;
}

__global__ void
rr_verify_histogram(const uint32_t* data, uint32_t count, uint32_t* bins)
{
    const uint32_t stride = blockDim.x * gridDim.x;
    for(uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < count; i += stride)
        atomicAdd(&bins[data[i] % kBins], 1U);
}

// Reads the finished histogram, so the buffer's final bytes depend on every count.
__global__ void
rr_verify_add_bin_count(uint32_t* data, uint32_t count, const uint32_t* bins)
{
    const uint32_t stride = blockDim.x * gridDim.x;
    for(uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < count; i += stride)
        data[i] += bins[data[i] % kBins];
}

namespace
{
constexpr unsigned kBlocks    = 1024;
constexpr unsigned kThreads   = 256;
constexpr size_t   kDataBytes = size_t{kElements} * sizeof(uint32_t);
constexpr size_t   kBinsBytes = size_t{kBins} * sizeof(uint32_t);

int
skip(const char* reason)
{
    setenv(kSkipEnv, reason, 1);
    printf("[app] SKIP: %s\n", reason);
    return EXIT_SUCCESS;
}

// Reports the first element that differs and how many do, so a failure shows whether the whole
// buffer or a few elements came back wrong.
bool
matches(const char*                  what,
        const std::vector<uint32_t>& actual,
        const std::vector<uint32_t>& expected)
{
    auto wrong = size_t{0};
    auto first = size_t{0};
    for(size_t i = 0; i < actual.size(); ++i)
    {
        if(actual[i] == expected[i]) continue;
        if(wrong++ == 0) first = i;
    }
    if(wrong == 0) return true;

    fprintf(stderr,
            "[app] FAIL: %s[%zu]=%u (expected %u); %zu of %zu elements differ\n",
            what,
            first,
            actual[first],
            expected[first],
            wrong,
            actual.size());
    return false;
}
}  // namespace

int
main()
{
    auto device_count = 0;
    if(hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0)
        return skip("no GPU device");

    const char* mode_env = std::getenv("RR_VERIFY_MODE");
    const auto  mode     = std::string_view{mode_env != nullptr ? mode_env : ""};
    if(mode != "in-place" && mode != "atomic-histogram")
    {
        fprintf(stderr, "[app] FAIL: unknown RR_VERIFY_MODE '%s'\n", mode_env ? mode_env : "");
        return EXIT_FAILURE;
    }
    const bool histogram = (mode == "atomic-histogram");

    auto* begin_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_begin)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_begin"));
    auto* end_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_end)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_end"));
    if(begin_fn == nullptr || end_fn == nullptr)
    {
        fprintf(stderr, "[app] FAIL: the range replay API is not in this process\n");
        return EXIT_FAILURE;
    }

    // A device allocation inside a range is a decline reason, so everything the range touches is
    // allocated and settled before it opens.
    uint32_t*   data   = nullptr;
    uint32_t*   bins   = nullptr;
    hipStream_t stream = nullptr;
    HIP_CHECK(hipMalloc(&data, kDataBytes));
    if(histogram)
    {
        HIP_CHECK(hipMalloc(&bins, kBinsBytes));
        HIP_CHECK(hipMemset(bins, 0, kBinsBytes));
    }
    HIP_CHECK(hipStreamCreate(&stream));
    rr_verify_fill<<<kBlocks, kThreads, 0, stream>>>(data, kElements);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    if(const auto status = begin_fn(kRangeId); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: rocprofiler_range_replay_begin returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    if(histogram)
    {
        rr_verify_histogram<<<kBlocks, kThreads, 0, stream>>>(data, kElements, bins);
        HIP_CHECK(hipGetLastError());
        rr_verify_add_bin_count<<<kBlocks, kThreads, 0, stream>>>(data, kElements, bins);
        HIP_CHECK(hipGetLastError());
    }
    else
    {
        for(uint32_t add = 1; add <= kInPlaceSteps; ++add)
        {
            rr_verify_step<<<kBlocks, kThreads, 0, stream>>>(data, kElements, add);
            HIP_CHECK(hipGetLastError());
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    if(const auto status = end_fn(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "[app] FAIL: rocprofiler_range_replay_end returned status %d\n",
                static_cast<int>(status));
        return EXIT_FAILURE;
    }

    auto data_h = std::vector<uint32_t>(kElements);
    auto bins_h = std::vector<uint32_t>(kBins);
    HIP_CHECK(hipMemcpy(data_h.data(), data, kDataBytes, hipMemcpyDeviceToHost));
    if(histogram) HIP_CHECK(hipMemcpy(bins_h.data(), bins, kBinsBytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipStreamDestroy(stream));
    if(bins != nullptr) HIP_CHECK(hipFree(bins));
    HIP_CHECK(hipFree(data));

    // Whatever the replayed passes did, the application must get back what its own execution of
    // the range computed: each kernel applied exactly once, in order, to the filled buffer.
    auto expected_data = std::vector<uint32_t>(kElements);
    auto expected_bins = std::vector<uint32_t>(kBins, 0);
    for(uint32_t i = 0; i < kElements; ++i)
        expected_data[i] = rr_verify_pattern(i);
    if(histogram)
    {
        for(const auto value : expected_data)
            ++expected_bins[value % kBins];
        for(auto& value : expected_data)
            value += expected_bins[value % kBins];
    }
    else
    {
        for(auto& value : expected_data)
            for(uint32_t add = 1; add <= kInPlaceSteps; ++add)
                value = (value * 3U) + add;
    }

    auto ok = matches("data", data_h, expected_data);
    if(histogram) ok = matches("bins", bins_h, expected_bins) && ok;
    printf("[app] mode=%s data[0]=%u data[%u]=%u%s\n",
           mode_env,
           data_h.front(),
           kElements - 1,
           data_h.back(),
           ok ? "" : " (wrong)");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
