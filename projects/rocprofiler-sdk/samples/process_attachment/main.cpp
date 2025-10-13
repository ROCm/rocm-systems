// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * @file samples/process_attachment/main.cpp
 *
 * @brief Target application for demonstrating process attachment
 * This is a HIP application that can be attached to by profilers
 * Usage: ./process_attachment_app -t <run_time_in_seconds>
 * If -t is not specified, it runs 60 seconds until interrupted
 */

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error %d: %s\n",                                                 \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    (int) error_,                                                                  \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

// Simple kernel for vector addition
__global__ void vector_add_kernel(float* a, float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Matrix multiplication kernel
__global__ void matrix_multiply_kernel(float* a, float* b, float* c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < n && col < n)
    {
        float sum = 0.0f;
        for(int k = 0; k < n; ++k)
        {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Simple compute-intensive kernel
__global__ void compute_kernel(float* data, int n, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        float val = data[idx];
        for(int i = 0; i < iterations; ++i)
        {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
        }
        data[idx] = val;
    }
}

class TargetApplication
{
private:
    bool keep_running = true;
    int device_count = 0;
    std::vector<float*> device_buffers;
    std::vector<float*> host_buffers;
    const size_t buffer_size = 1024 * 1024;  // 1M floats
    const size_t buffer_bytes = buffer_size * sizeof(float);
    
public:
    TargetApplication()
    {
        initialize_hip();
        allocate_buffers();
    }
    
    ~TargetApplication()
    {
        cleanup_buffers();
    }
    
    void initialize_hip()
    {
        roctxMark("Application initialization start");
        
        HIP_API_CALL(hipGetDeviceCount(&device_count));
        std::cout << "Found " << device_count << " HIP device(s)" << std::endl;
        
        if(device_count == 0)
        {
            throw std::runtime_error("No HIP devices found");
        }
        
        // Get device properties
        hipDeviceProp_t device_prop;
        HIP_API_CALL(hipGetDeviceProperties(&device_prop, 0));
        std::cout << "Using device: " << device_prop.name << std::endl;
        std::cout << "Global memory: " << (device_prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
        
        roctxMark("Application initialization complete");
    }
    
    void allocate_buffers()
    {
        roctxRangePushA("Buffer allocation");
        
        // Allocate device buffers
        device_buffers.resize(4);
        for(size_t i = 0; i < device_buffers.size(); ++i)
        {
            HIP_API_CALL(hipMalloc(&device_buffers[i], buffer_bytes));
        }
        
        // Allocate host buffers
        host_buffers.resize(4);
        for(size_t i = 0; i < host_buffers.size(); ++i)
        {
            host_buffers[i] = new float[buffer_size];
            
            // Initialize with random data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            
            for(size_t j = 0; j < buffer_size; ++j)
            {
                host_buffers[i][j] = dis(gen);
            }
        }
        
        roctxRangePop();
        std::cout << "Allocated " << device_buffers.size() << " device buffers of " 
                  << (buffer_bytes / (1024*1024)) << " MB each" << std::endl;
    }
    
    void cleanup_buffers()
    {
        roctxRangePushA("Buffer cleanup");
        
        for(auto* ptr : device_buffers)
        {
            if(ptr) HIP_API_CALL(hipFree(ptr));
        }
        device_buffers.clear();
        
        for(auto* ptr : host_buffers)
        {
            delete[] ptr;
        }
        host_buffers.clear();
        
        roctxRangePop();
    }
    
    void run_vector_addition_workload()
    {
        roctxRangePushA("Vector Addition Workload");
        
        // Copy data to device
        HIP_API_CALL(hipMemcpy(device_buffers[0], host_buffers[0], buffer_bytes, hipMemcpyHostToDevice));
        HIP_API_CALL(hipMemcpy(device_buffers[1], host_buffers[1], buffer_bytes, hipMemcpyHostToDevice));
        
        // Launch kernel
        dim3 block_size(256);
        dim3 grid_size((buffer_size + block_size.x - 1) / block_size.x);
        
        roctxMark("Vector addition kernel launch");
        hipLaunchKernelGGL(vector_add_kernel, grid_size, block_size, 0, 0,
                          device_buffers[0], device_buffers[1], device_buffers[2], (int)buffer_size);
        
        HIP_API_CALL(hipDeviceSynchronize());
        
        // Copy result back
        HIP_API_CALL(hipMemcpy(host_buffers[2], device_buffers[2], buffer_bytes, hipMemcpyDeviceToHost));
        
        roctxRangePop();
    }
    
    void run_matrix_multiply_workload()
    {
        roctxRangePushA("Matrix Multiplication Workload");
        
        const int matrix_size = 512;  // 512x512 matrix
        const size_t matrix_bytes = matrix_size * matrix_size * sizeof(float);
        
        // Copy data to device (using subset of our buffers)
        HIP_API_CALL(hipMemcpy(device_buffers[0], host_buffers[0], matrix_bytes, hipMemcpyHostToDevice));
        HIP_API_CALL(hipMemcpy(device_buffers[1], host_buffers[1], matrix_bytes, hipMemcpyHostToDevice));
        
        // Launch kernel
        dim3 block_size(16, 16);
        dim3 grid_size((matrix_size + block_size.x - 1) / block_size.x,
                      (matrix_size + block_size.y - 1) / block_size.y);
        
        roctxMark("Matrix multiplication kernel launch");
        hipLaunchKernelGGL(matrix_multiply_kernel, grid_size, block_size, 0, 0,
                          device_buffers[0], device_buffers[1], device_buffers[2], matrix_size);
        
        HIP_API_CALL(hipDeviceSynchronize());
        
        // Copy result back
        HIP_API_CALL(hipMemcpy(host_buffers[2], device_buffers[2], matrix_bytes, hipMemcpyDeviceToHost));
        
        roctxRangePop();
    }
    
    void run_compute_workload()
    {
        roctxRangePushA("Compute Intensive Workload");
        
        // Copy data to device
        HIP_API_CALL(hipMemcpy(device_buffers[0], host_buffers[0], buffer_bytes, hipMemcpyHostToDevice));
        
        // Launch compute-intensive kernel
        dim3 block_size(256);
        dim3 grid_size((buffer_size + block_size.x - 1) / block_size.x);
        
        roctxMark("Compute intensive kernel launch");
        hipLaunchKernelGGL(compute_kernel, grid_size, block_size, 0, 0,
                          device_buffers[0], (int)buffer_size, 1000);  // 1000 iterations
        
        HIP_API_CALL(hipDeviceSynchronize());
        
        // Copy result back
        HIP_API_CALL(hipMemcpy(host_buffers[0], device_buffers[0], buffer_bytes, hipMemcpyDeviceToHost));
        
        roctxRangePop();
    }
    
    void run_workload_cycle()
    {
        int cycle = 0;
        
        while(keep_running)
        {
            ++cycle;
            
            roctxRangePushA(("Workload Cycle " + std::to_string(cycle)).c_str());
            
            std::cout << "=== Cycle " << cycle << " ===" << std::endl;
            
            // Run different workloads in sequence
            run_vector_addition_workload();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            run_matrix_multiply_workload();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            run_compute_workload();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            roctxRangePop();
            
            std::cout << "Completed cycle " << cycle << " (PID: " << getpid() << ")" << std::endl;
            
            // Sleep between cycles
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    
    void stop()
    {
        keep_running = false;
    }
};

void print_usage(const char* prog_name)
{
    std::cout << "Usage: " << prog_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -t, --time <seconds>     Run for specified time (default: run indefinitely)" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "This application runs GPU workloads continuously and can be attached to" << std::endl;
    std::cout << "by process attachment tools for profiling." << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << prog_name << "                    # Run indefinitely" << std::endl;
    std::cout << "  " << prog_name << " -t 60              # Run for 60 seconds" << std::endl;
}

int main(int argc, char* argv[])
{
    int run_time_seconds = 60;  // Will default run for these many seconds
    
    // Parse command line arguments
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        
        if(arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
        else if(arg == "-t" || arg == "--time")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "Error: Time option requires a value" << std::endl;
                return 1;
            }
            run_time_seconds = std::atoi(argv[++i]);
            if(run_time_seconds <= 0)
            {
                std::cerr << "Error: Time must be a positive integer" << std::endl;
                return 1;
            }
        }
        else
        {
            std::cerr << "Error: Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "=== Target Application for Process Attachment ===" << std::endl;
    std::cout << "PID: " << getpid() << std::endl;
    
    if(run_time_seconds > 0)
    {
        std::cout << "Will run for " << run_time_seconds << " seconds" << std::endl;
    }
    else
    {
        std::cout << "Will run for 60 seconds (Ctrl+C to stop)" << std::endl;
    }
    
    try
    {
        TargetApplication app;
        
        // Set up timer if specified
        std::thread timer_thread;
        if(run_time_seconds > 0)
        {
            timer_thread = std::thread([&app, run_time_seconds]() {
                std::this_thread::sleep_for(std::chrono::seconds(run_time_seconds));
                app.stop();
            });
        }
        
        // Run the main workload
        app.run_workload_cycle();
        
        if(timer_thread.joinable())
        {
            timer_thread.join();
        }
        
        std::cout << "Application finished normally" << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}