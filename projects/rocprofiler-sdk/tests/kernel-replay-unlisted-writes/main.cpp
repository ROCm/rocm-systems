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

/**
 * @file tests/kernel-replay-unlisted-writes/main.cpp
 *
 * @brief Workload for the kernel replay test of writes outside a reused restore list.
 *
 * One kernel writes a single element in the 16 KiB tile whose index the client tool stores in
 * pinned host memory before each pass, so every pass writes a different tile. The tool replays it
 * with four passes, restoring from a list kept after the first. The application keeps the last
 * pass's memory, so exactly one element, the first of tile 3, may be nonzero.
 *
 * The kernel reads the pinned pointer from device memory, because the SDK runs a dispatch once
 * without replay when one of its arguments points at host memory.
 */

#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        const hipError_t _status = (call);                                                         \
        if(_status != hipSuccess)                                                                  \
        {                                                                                          \
            std::fprintf(stderr,                                                                   \
                         "[unlisted-writes] FAIL: %s returned %s\n",                               \
                         #call,                                                                    \
                         hipGetErrorString(_status));                                              \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while(0)

namespace
{
// The client tool replays only dispatches with this workgroup size.
constexpr unsigned kWriteBlockX  = 128;
constexpr size_t   kTileFloats   = 16 * 1024 / sizeof(float);
constexpr size_t   kTiles        = 8;
constexpr size_t   kExpectedTile = 3;  // the last of the tool's four passes

__global__ void
write_selected_tile(float* data, const volatile int* const* tile_slot)
{
    if(threadIdx.x == 0) data[static_cast<size_t>(**tile_slot) * kTileFloats] = 1.0f;
}

using register_tile_fn = void (*)(int*);
}  // namespace

int
main()
{
    auto register_tile =
        reinterpret_cast<register_tile_fn>(dlsym(RTLD_DEFAULT, "kr_unlisted_register_tile"));
    if(register_tile == nullptr)
    {
        std::fprintf(stderr, "[unlisted-writes] FAIL: no tool handshake (run with the client)\n");
        return EXIT_FAILURE;
    }

    int* tile = nullptr;
    HIP_CHECK(hipHostMalloc(&tile, sizeof(int)));
    *tile = 0;
    register_tile(tile);

    int** tile_slot = nullptr;
    HIP_CHECK(hipMalloc(&tile_slot, sizeof(int*)));
    HIP_CHECK(hipMemcpy(tile_slot, &tile, sizeof(int*), hipMemcpyHostToDevice));

    const size_t n    = kTiles * kTileFloats;
    float*       data = nullptr;
    HIP_CHECK(hipMalloc(&data, n * sizeof(float)));
    HIP_CHECK(hipMemset(data, 0, n * sizeof(float)));
    hipLaunchKernelGGL(write_selected_tile, dim3(1), dim3(kWriteBlockX), 0, 0, data, tile_slot);
    HIP_CHECK(hipDeviceSynchronize());

    auto host = std::vector<float>(n);
    HIP_CHECK(hipMemcpy(host.data(), data, n * sizeof(float), hipMemcpyDeviceToHost));
    auto wrong = size_t{0};
    for(size_t i = 0; i < n; ++i)
    {
        const auto expected = (i == kExpectedTile * kTileFloats) ? 1.0f : 0.0f;
        if(host[i] != expected)
        {
            if(wrong < 4)
                std::fprintf(stderr,
                             "[unlisted-writes] element %zu (tile %zu) = %.0f, expected %.0f\n",
                             i,
                             i / kTileFloats,
                             host[i],
                             expected);
            ++wrong;
        }
    }

    HIP_CHECK(hipFree(data));
    HIP_CHECK(hipFree(tile_slot));
    HIP_CHECK(hipHostFree(tile));
    if(wrong != 0)
    {
        std::printf("[unlisted-writes] FAIL: %zu elements differ from one run of the last pass\n",
                    wrong);
        return EXIT_FAILURE;
    }
    std::printf("[unlisted-writes] only the last pass's tile was written\n");
    return EXIT_SUCCESS;
}
