// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

#include <cstddef>
#include <cstdlib>
#include <thread>

/// \brief A simple matrix transpose kernel that using dynamic shared memory.
/// - The number of rows in the input and output matrices is equal, and given by the \p width parameter.
/// - Each thread in the grid is responsible for one element of the input and output matrices.
/// - Because the transposition is computed in shared memory, which cannot be accessed between different
/// blocks, the matrix has to be processed by a single block.
__global__ void matrix_transpose_kernel(float* out, const float* in, const unsigned int width)
{
    // Declare that this kernel is using dynamic shared memory to store a number of floats.
    // The unsized array type indicates that the total amount of memory that is going
    // to be used here is not known ahead of time, and will be computed at runtime and
    // passed to the kernel launch function.
    extern __shared__ float shared_matrix_memory[];

    // Compute the row and column index of the element this thread is going to process.
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    // Perform the transpose by reading an element of the input matrix from global memory and
    // by storing it in the tranposed index in shared memory.
    shared_matrix_memory[y * width + x] = in[x * width + y];

    // Synchronization is required to make sure that all threads have written
    // their part of the input matrix to the shared memory, before the values
    // are read by another thread.
    __syncthreads();

    // Copy the transposed matrix from shared memory to the output array, which
    // is in global memory.
    out[y * width + x] = shared_matrix_memory[y * width + x];
}

// CPU implementation of matrix transpose
std::vector<float> matrix_transpose_reference(const std::vector<float>& input,
                                              const unsigned int        width)
{
    std::vector<float> output(width * width);
    for(unsigned int j = 0; j < width; j++)
    {
        for(unsigned int i = 0; i < width; i++)
        {
            output[i * width + j] = input[j * width + i];
        }
    }
    return output;
}

// argv: Array of command-line arguments. Run with "--enable-sleep" to enable 
// the mode with delay implemented by thread sleep
int main(int argc, char* argv[])
{
    bool enable_sleep = false;

    // Check command-line arguments
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg == "--enable-sleep")
        {
            enable_sleep = true;
        }
    }

    constexpr unsigned int width = 4;
    constexpr unsigned int threads_per_block_x = width;
    constexpr unsigned int threads_per_block_y = width;
    constexpr unsigned int size = width * width;
    constexpr size_t size_bytes = sizeof(float) * size;
    constexpr size_t shared_memory_bytes = size_bytes;

    std::cout << "Run transpose continuously" << std::endl;

    if(enable_sleep)
        std::this_thread::sleep_for(std::chrono::seconds(30));

    unsigned int pass_count = 0;
    unsigned int fail_count = 0;
    unsigned int cycle_count = 0;

    constexpr float eps = 1.0E-6f;

    while (true) 
    {
        if(enable_sleep)
            std::this_thread::sleep_for(std::chrono::seconds(5));

        // Allocate host vectors
        std::vector<float> h_matrix(size);
        std::vector<float> h_transposed_matrix(size);

        // Set up input data
        for(unsigned int i = 0; i < size; i++)
        {
            h_matrix[i] = i * 10.0f;
        }

        // Allocate device memory
        float* d_matrix{};
        float* d_transposed_matrix{};
        HIP_CHECK(hipMalloc(&d_matrix, size_bytes));
        HIP_CHECK(hipMalloc(&d_transposed_matrix, size_bytes));

        // Copy input to device
        HIP_CHECK(hipMemcpy(d_matrix, h_matrix.data(), size_bytes, hipMemcpyHostToDevice));

        // Launch kernel
        matrix_transpose_kernel<<<dim3(width / threads_per_block_x, width / threads_per_block_y),
                                  dim3(threads_per_block_x, threads_per_block_y),
                                  shared_memory_bytes,
                                  hipStreamDefault>>>(d_transposed_matrix, d_matrix, width);

        HIP_CHECK(hipGetLastError());

        // Copy result back
        HIP_CHECK(hipMemcpy(h_transposed_matrix.data(),
                            d_transposed_matrix,
                            size_bytes,
                            hipMemcpyDeviceToHost));

        // Free device memory
        HIP_CHECK(hipFree(d_matrix));
        HIP_CHECK(hipFree(d_transposed_matrix));

        // CPU reference transpose
        std::vector<float> ref_transposed_matrix = matrix_transpose_reference(h_matrix, width);

        // Validate
        unsigned int errors = 0;
        for(unsigned int i = 0; i < size; i++)
        {
            if(std::fabs(h_transposed_matrix[i] - ref_transposed_matrix[i]) > eps)
            {
                errors++;
            }
        }

        // Update pass/fail counters
        if(errors == 0)
            pass_count++;
        else
            fail_count++;

        cycle_count++;

        // Every 10000 cycles, print summary and reset counters
        if(cycle_count == 10000)
        {
            std::cout << "10000 Validation cycles completed: "
                      << "Passes = " << pass_count
                      << ", Failures = " << fail_count << std::endl;

            // Reset counters
            cycle_count = 0;
            pass_count = 0;
            fail_count = 0;
        }
    }
}