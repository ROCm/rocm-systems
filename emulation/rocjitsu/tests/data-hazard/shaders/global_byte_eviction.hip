// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// mutate: memory
//
// Cross-workgroup sub-dword eviction under a single-byte flood.
//
// Ten single-thread workgroups touch bytes of ONE 4-byte global word. Byte 0 is
// read heavily so it floods the shadow's access history; byte 1 keeps a lone
// reader that a later write to byte 1 must still race:
//
//     WG0..6 read data[0]  (byte 0)  ─ seven readers flood byte 0
//     WG7    reads data[1] (byte 1)  ─ byte 1's lone reader (must survive)
//     WG8    reads data[0] (byte 0)  ─ one more byte-0 read, forcing an eviction
//     WG9    reads data[1] (byte 1)  ─ the PROBE, a separate load
//
// Every access is an ordinary read, so the baseline is race-free and
// deterministic: reads never race reads, whatever order the workgroups run in.
//
// The `data_hazard` global shadow keeps a small, fixed number of per-workgroup
// access slots per 4-byte word. A FLAT pool of those slots is order-dependent:
// byte 0's seven readers fill the slots, byte 1's reader (WG7) takes the last
// one, then WG8's byte-0 read overwrites it — evicting byte 1's only reader.
//
// The mutation harness turns the PROBE load into a byte store (read -> write).
// WG9 then WRITES byte 1, which must race WG7's read of byte 1 (WAR). With the
// evicted reader gone the race is MISSED -> false negative. Partitioning
// retention per byte keeps byte 1's reader however busy byte 0 gets, so the
// mutant is DETECTED.
//
// Only bytes 0 and 1 are touched, and each by more than one workgroup, so every
// load the harness may rewrite as a store forms a real cross-workgroup race —
// no rewrite yields a spurious race-free "miss". The flood load `data[0]` and
// the byte-1 loads must stay distinct instructions so the harness can mutate the
// probe alone; `volatile` keeps the compiler from folding them into one
// address-selected load.
__global__ void global_byte_eviction(const volatile unsigned char* data, int* result) {
    unsigned int wg = blockIdx.x;
    int          v;
    if (wg < 7) {
        v = data[0]; // WG0..6: flood byte 0
    } else if (wg == 7) {
        v = data[1]; // WG7: byte 1's lone reader (must survive the flood)
    } else if (wg == 8) {
        v = data[0]; // WG8: one more byte-0 read, evicting the last flat slot
    } else {
        v = data[1]; // WG9: PROBE read of byte 1 (mutated to a byte store)
    }
    result[wg] = v;
}

int main() {
    constexpr int numBlocks = 10;

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

    // WG0..6 and WG8 read byte 0; WG7 and WG9 read byte 1.
    const int expected[numBlocks] = {data[0], data[0], data[0], data[0], data[0],
                                     data[0], data[0], data[1], data[0], data[1]};
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
