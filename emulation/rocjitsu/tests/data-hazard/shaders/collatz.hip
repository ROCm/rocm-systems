// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

#define MAX_STEPS           (1024 * 64)
#define NON_CONVERGENCE     (MAX_STEPS + 1)
#define COLLATZ_STEP(value) ((value) % 2 == 0 ? (value) / 2 : 3 * (value) + 1)

__host__ __device__ constexpr unsigned int calculate_collatz_steps(const unsigned int start_value) {
    unsigned int value      = start_value;
    unsigned int step_count = 0;
    while (value != 1 && step_count < MAX_STEPS) {
        value = COLLATZ_STEP(value);
        ++step_count;
    }
    if (value != 1) {
        step_count = NON_CONVERGENCE;
    }
    return step_count;
}

// Addition of two values.
__global__ void collatz_steps(const unsigned int* start, unsigned int* steps, const std::size_t size) {
    const std::size_t index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index >= size) {
        return;
    }

    steps[index] = calculate_collatz_steps(start[index]);
}

int main() {
    constexpr int         numOfBlocks     = 4;
    constexpr int         threadsPerBlock = 64;
    constexpr std::size_t arraySize       = numOfBlocks * threadsPerBlock;

    std::vector<unsigned int> start(arraySize), steps(arraySize);
    unsigned int *            d_start, *d_steps;
    // Setup input values.
    for (std::size_t i = 0; i < arraySize; ++i) {
        start[i] = static_cast<unsigned int>(i + 1); // Avoid starting at 0 which is not valid for Collatz
    }
    // Allocate device copies of start and steps.
    HIP_CHECK(hipMalloc(&d_start, arraySize * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_steps, arraySize * sizeof(unsigned int)));
    // Copy input values to device.
    HIP_CHECK(hipMemcpy(d_start, start.data(), arraySize * sizeof(unsigned int), hipMemcpyHostToDevice));
    // Launch collatz_steps() kernel on GPU.
    collatz_steps<<<numOfBlocks, threadsPerBlock>>>(d_start, d_steps, arraySize);
    // Check the kernel launch
    HIP_CHECK(hipGetLastError());
    // Check for kernel execution error
    HIP_CHECK(hipDeviceSynchronize());
    // Copy the result back to the host.
    HIP_CHECK(hipMemcpy(steps.data(), d_steps, arraySize * sizeof(unsigned int), hipMemcpyDeviceToHost));
    // Cleanup allocated memory.
    HIP_CHECK(hipFree(d_start));
    HIP_CHECK(hipFree(d_steps));

    // Print results as JSON array to stdout
    std::cout << "[";
    for (std::size_t i = 0; i < arraySize; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << steps[i];
    }
    std::cout << "]" << std::endl;

    // validate results
    bool failed = false;
    for (std::size_t i = 0; i < arraySize; ++i) {
        unsigned int expected_steps = calculate_collatz_steps(start[i]);
        if (steps[i] != expected_steps) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected_steps << " steps, got " << steps[i]
                      << " steps." << std::endl;
            failed = true;
        }
    }
    if (!failed) {
        std::cerr << "PASS" << std::endl;
        return EXIT_SUCCESS;
    }
    std::cerr << "FAIL" << std::endl;
    return EXIT_FAILURE;
}
