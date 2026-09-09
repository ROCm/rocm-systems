// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "utils.hpp"

// Tensor DMA load to LDS at a nonzero descriptor base.
//
// The companion of tensor_lds.hip, with the tile placed 128 bytes into the
// group segment instead of at its start, so the LDS the transfer writes and the
// LDS the kernel reads are both away from offset zero. Hazard tracking that
// assumed a tensor DMA writes the first dword of LDS therefore sees no overlap
// here at all: the RAW hazard is only reported when the write range comes from
// the descriptor's lds_addr, element size and tile dimensions.
//
// Hazard surface:
//   - tensor_load_to_lds writes sdata[32:63] (async, tracked by TENSORcnt)
//   - ds_read reads sdata[32:63]
//   - Removing s_wait_tensorcnt → RAW hazard on the tile's LDS data
//
// The builtin is gated by the "gfx1250-insts" target feature, so the harness is
// told to skip this kernel elsewhere:
// requires: gfx1250

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

__global__ void tensor_lds_offset_load(const v4i* __restrict__ d_group0,
                                       const v8i* __restrict__ d_group1,
                                       int* __restrict__ output,
                                       unsigned int n) {
    __shared__ int sdata[64];

    sdata[32 + threadIdx.x] = -999;
    __syncthreads();

    v4i group0 = d_group0[0];
    v8i group1 = d_group1[0];

    // The descriptor's lds_addr places the tile 128 bytes in, over sdata[32:63].
    // D# groups 2-4 are zero-initialized for this ≤2D descriptor; the trailing
    // operand is the cachepolicy (bits[0-2]=th, bits[3-4]=scope).
    const v4i zero4 = {0, 0, 0, 0};
    const v8i zero8 = {0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(group0, group1, zero4, zero4, zero8, 0);

    asm volatile("s_wait_tensorcnt 0" ::: "memory");

    if (threadIdx.x < n) {
        output[threadIdx.x] = sdata[32 + threadIdx.x];
    }
}

// Builds the same 20-DWORD descriptor tensor_lds.hip documents field by field,
// for a 1-row, N-element tile of 32-bit integers at the given LDS byte offset.
static void build_tensor_descriptor(uint32_t* desc,
                                    uint64_t  global_addr,
                                    uint32_t  lds_addr,
                                    uint32_t  n_elements) {
    std::memset(desc, 0, 20 * sizeof(uint32_t));

    desc[0] = 1;         // count = 1
    desc[1] = lds_addr;

    desc[2] = static_cast<uint32_t>(global_addr & 0xFFFFFFFFu);
    desc[3] = static_cast<uint32_t>((global_addr >> 32) & 0x1FFFFFFu);
    desc[3] |= (2u << 30);  // type = 2

    desc[4] = (2u << 16);   // data_size = 2 -> 4 bytes/element

    desc[5] = (n_elements & 0xFFFFu) << 16;
    desc[6] = (n_elements >> 16) & 0xFFFFu;
    desc[6] |= (1u << 16);  // tensor_dim1 = 1

    desc[7] = (n_elements & 0xFFFFu) << 16;  // tile_dim0
    desc[8] = 1;                             // tile_dim1
    desc[9] = n_elements;                    // tensor_dim0_stride
}

int main() {
    constexpr unsigned int N = 32;

    std::vector<int> h_tensor(N);
    for (unsigned int i = 0; i < N; ++i)
        h_tensor[i] = static_cast<int>(i);

    int* d_tensor;
    HIP_CHECK(hipMalloc(&d_tensor, N * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_tensor, h_tensor.data(), N * sizeof(int), hipMemcpyHostToDevice));

    uint32_t h_desc[20];
    build_tensor_descriptor(h_desc,
                            reinterpret_cast<uint64_t>(d_tensor),
                            /*lds_addr=*/128,
                            N);

    v4i* d_group0;
    v8i* d_group1;
    HIP_CHECK(hipMalloc(&d_group0, sizeof(v4i)));
    HIP_CHECK(hipMalloc(&d_group1, sizeof(v8i)));
    HIP_CHECK(hipMemcpy(d_group0, &h_desc[0], sizeof(v4i), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_group1, &h_desc[4], sizeof(v8i), hipMemcpyHostToDevice));

    int* d_output;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(int)));
    HIP_CHECK(hipMemset(d_output, 0, N * sizeof(int)));

    tensor_lds_offset_load<<<1, 32>>>(d_group0, d_group1, d_output, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> h_output(N, 0);
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, N * sizeof(int), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_tensor));
    HIP_CHECK(hipFree(d_group0));
    HIP_CHECK(hipFree(d_group1));
    HIP_CHECK(hipFree(d_output));

    std::cout << "[";
    for (unsigned int i = 0; i < N; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << h_output[i];
    }
    std::cout << "]" << std::endl;

    bool pass = true;
    for (unsigned int i = 0; i < N; ++i) {
        if (h_output[i] != static_cast<int>(i)) {
            std::cerr << "Mismatch at " << i << ": expected " << i << " got " << h_output[i] << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
