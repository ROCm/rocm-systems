/*
##############################################################################bl
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el
*/

#include <hip/hip_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

// ============================================================
// Error handling
// ============================================================
#define HIP_CHECK(call)                                                      \
    do {                                                                     \
        hipError_t err = call;                                               \
        if (err != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, hipGetErrorString(err));             \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// ============================================================
// Helpers
// ============================================================
static inline bool is_pow2_u64(uint64_t x) {
    return x && ((x & (x - 1)) == 0);
}

static bool hasArg(int argc, char** argv, const char* arg) {
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], arg) == 0) return true;
    }
    return false;
}

static const char* getArg(int argc, char** argv, const char* arg, const char* def) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::strcmp(argv[i], arg) == 0) return argv[i + 1];
    }
    return def;
}

// ============================================================
// Baseline naive transpose
//   - 1 global load + 1 global store per thread
// ============================================================
__global__ void transposeNaive(
    const float* __restrict__ in,
    float* __restrict__ out,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[y * width + x] = in[x * width + y];
    }
}

// ============================================================
// Tagram stress: power-of-two stride
//   - 1 load + 1 store
//   - idx = base + (tid * stride)
// ============================================================
__global__ void tagramStride(
    const float* __restrict__ in,
    float* __restrict__ out,
    int width,
    int height,
    uint64_t base_offset_floats,
    uint64_t stride_floats,
    uint64_t index_mask
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint64_t tid = (uint64_t)y * (uint64_t)width + (uint64_t)x;
        uint64_t idx = (base_offset_floats + tid * stride_floats) & index_mask;
        out[tid] = in[idx];
    }
}

// ============================================================
// Tagram stress: diagonal / affine permutation (structured, not random)
//
// Key idea:
//   idx = base + ((tid * A + block_id * B) XOR (tid >> s)) * stride
//
// Where:
//   - A is odd => bijection mod 2^n (on power-of-two domains)
//   - XOR with shifted tid breaks "nice" linearity but remains deterministic
//   - stride can be power-of-two to force aliasing in low bits
//
// 1 load + 1 store
// ============================================================
__global__ void tagramDiagonal(
    const float* __restrict__ in,
    float* __restrict__ out,
    int width,
    int height,
    uint64_t base_offset_floats,
    uint64_t stride_floats,
    uint64_t A_odd,
    uint64_t B_odd,
    uint32_t xor_shift,
    uint64_t index_mask
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint64_t tid = (uint64_t)y * (uint64_t)width + (uint64_t)x;
        uint64_t block_id = (uint64_t)blockIdx.y * (uint64_t)gridDim.x + (uint64_t)blockIdx.x;

        // Affine permutation on 2^n domain if A is odd
        uint64_t p = tid * A_odd + block_id * B_odd;

        // Deterministic "diagonal-ish" mixing (still structured)
        p ^= (tid >> xor_shift);

        uint64_t idx = (base_offset_floats + p * stride_floats) & index_mask;

        // 1 load + 1 store
        out[tid] = in[idx];
    }
}

int main(int argc, char** argv) {
    const bool run_baseline = hasArg(argc, argv, "--baseline");
    const bool run_stride   = hasArg(argc, argv, "--tagram-stride");
    const bool run_diag     = hasArg(argc, argv, "--tagram-diagonal");

    if (!run_baseline && !run_stride && !run_diag) {
        std::cout <<
            "Usage:\n"
            "  --baseline                 Run naive transpose\n"
            "  --tagram-stride            Parity stress: idx = base + tid*stride\n"
            "  --tagram-diagonal          Parity stress: diagonal/affine permutation\n\n"
            "Common options:\n"
            "  --width N                  default 8192 (power of two)\n"
            "  --height N                 default 8192 (power of two)\n\n"
            "Stride mode options:\n"
            "  --stride_kb N              stride in KB (recommend powers of two; default 256)\n"
            "  --offset_b N               base offset in bytes (default 0)\n\n"
            "Diagonal mode options:\n"
            "  --stride_kb N              stride in KB (default 256)\n"
            "  --offset_b N               base offset in bytes (default 0)\n"
            "  --A N                      odd multiplier A (default 131071)\n"
            "  --B N                      odd multiplier B (default 524287)\n"
            "  --xor_shift N              shift for XOR mixing (default 5)\n\n"
            "Notes:\n"
            "  width*height must be power-of-two for mask wrapping.\n";
        return 0;
    }

    const int width  = std::atoi(getArg(argc, argv, "--width",  "8192"));
    const int height = std::atoi(getArg(argc, argv, "--height", "8192"));

    const uint64_t num_elems = (uint64_t)width * (uint64_t)height;
    if (!is_pow2_u64(num_elems)) {
        std::cerr << "ERROR: width*height must be a power of two (got " << num_elems << ")\n";
        return 2;
    }

    const uint64_t bytes = num_elems * sizeof(float);
    const uint64_t index_mask = num_elems - 1;

    // Host buffers
    float* h_in  = (float*)std::malloc((size_t)bytes);
    float* h_out = (float*)std::malloc((size_t)bytes);
    if (!h_in || !h_out) {
        std::cerr << "ERROR: host malloc failed\n";
        return 3;
    }

    for (uint64_t i = 0; i < num_elems; i++) {
        h_in[i] = (float)(i & 0xFF);
    }

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, (size_t)bytes));
    HIP_CHECK(hipMalloc(&d_out, (size_t)bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in, (size_t)bytes, hipMemcpyHostToDevice));

    // IDENTICAL launch geometry for all kernels (controlled comparison)
    dim3 block(32, 8);  // 256 threads
    dim3 grid(
        (width  + (int)block.x - 1) / (int)block.x,
        (height + (int)block.y - 1) / (int)block.y
    );

    if (run_baseline) {
        hipLaunchKernelGGL(transposeNaive, grid, block, 0, 0, d_in, d_out, width, height);
        HIP_CHECK(hipDeviceSynchronize());
        std::cout << "[baseline] done\n";
    }

    if (run_stride || run_diag) {
        const uint64_t stride_kb = std::stoull(getArg(argc, argv, "--stride_kb", "256"));
        const uint64_t offset_b  = std::stoull(getArg(argc, argv, "--offset_b", "0"));

        const uint64_t stride_floats = (stride_kb * 1024ULL) / sizeof(float);
        const uint64_t offset_floats = offset_b / sizeof(float);

        if (run_stride) {
            hipLaunchKernelGGL(
                tagramStride,
                grid, block, 0, 0,
                d_in, d_out,
                width, height,
                offset_floats, stride_floats, index_mask
            );
            HIP_CHECK(hipDeviceSynchronize());
            std::cout << "[tagram-stride] stride_kb=" << stride_kb
                      << " offset_b=" << offset_b << " done\n";
        }

        if (run_diag) {
            uint64_t A = std::stoull(getArg(argc, argv, "--A", "131071"));
            uint64_t B = std::stoull(getArg(argc, argv, "--B", "524287"));
            uint32_t xor_shift = (uint32_t)std::stoul(getArg(argc, argv, "--xor_shift", "5"));

            // Force A,B odd (required for good permutation properties on 2^n)
            A |= 1ULL;
            B |= 1ULL;

            hipLaunchKernelGGL(
                tagramDiagonal,
                grid, block, 0, 0,
                d_in, d_out,
                width, height,
                offset_floats, stride_floats,
                A, B, xor_shift,
                index_mask
            );
            HIP_CHECK(hipDeviceSynchronize());
            std::cout << "[tagram-diagonal] stride_kb=" << stride_kb
                      << " offset_b=" << offset_b
                      << " A=" << A << " B=" << B
                      << " xor_shift=" << xor_shift
                      << " done\n";
        }
    }

    HIP_CHECK(hipMemcpy(h_out, d_out, (size_t)bytes, hipMemcpyDeviceToHost));

    hipFree(d_in);
    hipFree(d_out);
    std::free(h_in);
    std::free(h_out);

    return 0;
}
