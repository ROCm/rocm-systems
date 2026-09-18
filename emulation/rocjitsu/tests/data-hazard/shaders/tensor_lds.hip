// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "utils.hpp"

// Tensor DMA load to LDS kernel.
//
// Exercises tensor_load_to_lds (TENSORcnt counter) → s_wait_tensorcnt → LDS
// read via the __builtin_amdgcn_tensor_load_to_lds compiler builtin.
// The Tensor DMA unit (TDM) copies a tile from a global-memory tensor
// directly into LDS — no VGPRs are involved and the EXEC mask is ignored.
// The DMA has to be synchronised with s_wait_tensorcnt before any subsequent
// LDS read. The compiler does not emit that wait for this builtin yet, so the
// kernel below writes it inline; it is the wait the harness mutates away, not
// a compiler-generated one.
//
// Hazard surface:
//   - tensor_load_to_lds writes to LDS (async, tracked by TENSORcnt)
//   - ds_read reads from LDS (via __shared__ access)
//   - Removing s_wait_tensorcnt → RAW hazard on LDS data: the ds_read may
//     observe stale sentinel values instead of the tensor-loaded data.
//
// The builtin is gated by the "gfx1250-insts" target feature and is
// unavailable on earlier targets, so the harness is told to skip this kernel
// everywhere else:
// requires: gfx1250

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

__global__ void tensor_lds_load(const v4i* __restrict__ d_group0,
                                const v8i* __restrict__ d_group1,
                                int* __restrict__ output,
                                unsigned int n) {
    __shared__ int sdata[32];

    sdata[threadIdx.x] = -999;
    __syncthreads();

    // Load D# groups from device memory (uniform across the wave).
    v4i group0 = d_group0[0];
    v8i group1 = d_group1[0];

    // Issue tensor_load_to_lds — DMA from global tensor into LDS. For a ≤2D
    // descriptor, D# groups 2-4 are zero-initialized; the trailing operand is
    // the cachepolicy (bits[0-2]=th, bits[3-4]=scope).
    const v4i zero4 = {0, 0, 0, 0};
    const v8i zero8 = {0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(group0, group1, zero4, zero4, zero8, 0);

    // Wait for the tensor DMA to complete before reading LDS.
    // The compiler's waitcnt insertion pass does not yet emit
    // s_wait_tensorcnt automatically, so we insert it explicitly.
    asm volatile("s_wait_tensorcnt 0" ::: "memory");

    if (threadIdx.x < n) {
        output[threadIdx.x] = sdata[threadIdx.x];
    }
}

// Build a 20-DWORD (640-bit) tensor descriptor (D#) for a 1-row, N-element
// tile of 32-bit integers.
//
// D# group 0 — val[0:3] (128 bits, loaded into 4 SGPRs):
//   val[0]  bits [ 1: 0]  count              = 1  (valid descriptor)
//   val[1]  bits [31: 0]  lds_addr           = 0  (LDS byte offset)
//   val[2]  bits [31: 0]  global_addr[31:0]       (tensor base low)
//   val[3]  bits [24: 0]  global_addr[56:32]      (tensor base high)
//           bits [31:30]  type               = 2
//
// D# group 1 — val[4:11] (256 bits, loaded into 8 SGPRs):
//   val[4]  bits [17:16]  data_size          = 2  (2^2 = 4 bytes/elem)
//   val[5]  bits [31:16]  tensor_dim0[15:0]  = N  (width in elements)
//   val[6]  bits [15: 0]  tensor_dim0[31:16]
//           bits [31:16]  tensor_dim1[15:0]  = 1  (height)
//   val[7]  bits [15: 0]  tensor_dim1[31:16]
//           bits [31:16]  tile_dim0          = N
//   val[8]  bits [15: 0]  tile_dim1          = 1
//   val[9]  bits [31: 0]  tensor_dim0_stride = N  (row stride in elems)
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

    desc[4] = (2u << 16);   // data_size = 2 → 4 bytes/element

    // tensor_dim0 (32-bit) spans val[5][31:16] and val[6][15:0]
    desc[5] = (n_elements & 0xFFFFu) << 16;
    desc[6] = (n_elements >> 16) & 0xFFFFu;

    // tensor_dim1 = 1 spans val[6][31:16] and val[7][15:0]
    desc[6] |= (1u << 16);

    // tile_dim0 = n_elements in val[7][31:16]
    desc[7] = (n_elements & 0xFFFFu) << 16;

    // tile_dim1 = 1 in val[8][15:0]
    desc[8] = 1;

    // tensor_dim0_stride (lower 32 of 48 bits) in val[9]
    desc[9] = n_elements;
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
                            /*lds_addr=*/0,
                            N);

    // D# group 0 = val[0:3] (4 DWORDs), group 1 = val[4:11] (8 DWORDs)
    v4i* d_group0;
    v8i* d_group1;
    HIP_CHECK(hipMalloc(&d_group0, sizeof(v4i)));
    HIP_CHECK(hipMalloc(&d_group1, sizeof(v8i)));
    HIP_CHECK(hipMemcpy(d_group0, &h_desc[0], sizeof(v4i), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_group1, &h_desc[4], sizeof(v8i), hipMemcpyHostToDevice));

    int* d_output;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(int)));
    HIP_CHECK(hipMemset(d_output, 0, N * sizeof(int)));

    tensor_lds_load<<<1, 32>>>(d_group0, d_group1, d_output, N);
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
