// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// mutate: memory
//
// Cross-workgroup sub-dword eviction pattern.
//
// Four single-thread workgroups touch bytes of ONE 4-byte global word:
//
//     WG0 reads data[0]   (byte 0)   ─┐
//     WG1 reads data[1]   (byte 1)    ├─ disjoint byte READS, one shared load
//     WG2 reads data[2]   (byte 2)   ─┘
//     WG3 reads data[1]   (byte 1)   ── the PROBE, a separate load
//
// Every access is an ordinary read, so the baseline is race-free and
// deterministic: reads never race reads, whatever order the workgroups run in.
//
// The `data_hazard` global shadow keeps a small array of per-workgroup access
// slots per 4-byte word. WG0/1/2's disjoint-byte reads accumulate in that
// array WITHOUT flagging (reads do not race reads). With only two slots, the
// middle byte's reader (WG1 @ byte 1) is evicted before it can be raced.
//
// The mutation harness turns the PROBE load into a byte store (read -> write).
// WG3 then WRITES byte 1, which must race WG1's read of byte 1 (WAR). With the
// buggy two-slot array WG1's read has already been evicted, so the race is
// MISSED -> false negative. Growing the slot array so every byte keeps its
// foreign readers restores detection -> the mutant is DETECTED.
//
// The disjoint-byte load `data[wg]` and the probe load `data[1]` must stay two
// distinct instructions so the harness can mutate the probe alone; `volatile`
// keeps the compiler from folding them into one address-selected load.
__global__ void global_byte_eviction(const volatile unsigned char* data, int* result) {
    unsigned int wg = blockIdx.x;
    int          v;
    if (wg < 3) {
        v = data[wg]; // disjoint byte reads: WG0->b0, WG1->b1, WG2->b2
    } else {
        v = data[1]; // PROBE read of byte 1 (mutated to a byte store)
    }
    result[wg] = v;
}

int main() {
    constexpr int numBlocks = 4;

    // One 4-byte word, one distinct nonzero value per byte.
    std::vector<unsigned char> data = {10, 20, 30, 40};
    std::vector<int>           result(numBlocks, 0);

    unsigned char* d_data;
    int*           d_result;
    HIP_CHECK(hipMalloc(&d_data, data.size() * sizeof(unsigned char)));
    HIP_CHECK(hipMalloc(&d_result, result.size() * sizeof(int)));
    HIP_CHECK(
        hipMemcpy(d_data, data.data(), data.size() * sizeof(unsigned char), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_result, 0, result.size() * sizeof(int)));

    global_byte_eviction<<<numBlocks, 1>>>(d_data, d_result);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(
        hipMemcpy(result.data(), d_result, result.size() * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_result));

    std::cout << "[";
    for (int i = 0; i < numBlocks; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << result[i];
    }
    std::cout << "]" << std::endl;

    // WG0..2 read their own byte; WG3 reads byte 1 like WG1.
    const int expected[numBlocks] = {data[0], data[1], data[2], data[1]};
    bool      pass                = true;
    for (int i = 0; i < numBlocks; ++i) {
        if (result[i] != expected[i]) {
            std::cerr << "Mismatch at " << i << ": expected " << expected[i] << " got " << result[i]
                      << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
