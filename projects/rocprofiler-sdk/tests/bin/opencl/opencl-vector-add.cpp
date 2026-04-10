/*
Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <CL/cl.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <vector>

#define VECTOR_SIZE 1024
#define GROUP_SIZE  256

namespace
{
using auto_lock_t = std::unique_lock<std::mutex>;
auto print_lock   = std::mutex{};
}  // namespace

#define CL_CHECK(CALL)                                                                             \
    {                                                                                              \
        cl_int err_ = (CALL);                                                                      \
        if(err_ != CL_SUCCESS)                                                                     \
        {                                                                                          \
            auto _cl_print_lk = auto_lock_t{print_lock};                                           \
            fprintf(stderr, "%s:%d :: OpenCL error : %d\n", __FILE__, __LINE__, err_);             \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

const char* kernel_source = R"(
__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* c,
                         const unsigned int n)
{
    int id = get_global_id(0);
    if(id < n)
    {
        c[id] = a[id] + b[id];
    }
}
)";

void
print_device_info(cl_device_id device)
{
    char    device_name[256];
    char    vendor[256];
    size_t  max_work_group_size;
    cl_uint compute_units;

    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
    clGetDeviceInfo(device,
                    CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(max_work_group_size),
                    &max_work_group_size,
                    nullptr);
    clGetDeviceInfo(
        device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);

    auto _print_lk = auto_lock_t{print_lock};
    std::cout << "OpenCL Device Info:\n";
    std::cout << "  Device Name: " << device_name << "\n";
    std::cout << "  Vendor: " << vendor << "\n";
    std::cout << "  Max Work Group Size: " << max_work_group_size << "\n";
    std::cout << "  Compute Units: " << compute_units << "\n\n";
}

int
main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    roctxMark("opencl_application_start");

    cl_int err;

    // Get platform
    roctxRangePush("get_platform");
    cl_platform_id platform;
    CL_CHECK(clGetPlatformIDs(1, &platform, nullptr));
    roctxRangePop();

    // Get device
    roctxRangePush("get_device");
    cl_device_id device;
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
    print_device_info(device);
    roctxRangePop();

    // Create context
    roctxRangePush("create_context");
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);
    roctxRangePop();

    // Create command queue
    roctxRangePush("create_command_queue");
#ifdef CL_VERSION_2_0
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    CL_CHECK(err);
    roctxRangePop();

    // Create program
    roctxRangePush("create_program");
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, &err);
    CL_CHECK(err);
    roctxRangePop();

    // Build program
    roctxRangePush("build_program");
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);

        auto _print_lk = auto_lock_t{print_lock};
        std::cerr << "Build log:\n" << log.data() << "\n";
        exit(EXIT_FAILURE);
    }
    roctxRangePop();

    // Create kernel
    roctxRangePush("create_kernel");
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    CL_CHECK(err);
    roctxRangePop();

    // Prepare host data
    roctxRangePush("prepare_data");
    std::vector<float> h_a(VECTOR_SIZE);
    std::vector<float> h_b(VECTOR_SIZE);
    std::vector<float> h_c(VECTOR_SIZE);

    for(size_t i = 0; i < VECTOR_SIZE; i++)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    roctxRangePop();

    // Create buffers
    roctxRangePush("create_buffers");
    cl_mem d_a =
        clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    cl_mem d_b =
        clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(float), nullptr, &err);
    CL_CHECK(err);

    cl_mem d_c =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(float), nullptr, &err);
    CL_CHECK(err);
    roctxRangePop();

    // Copy data to device
    roctxRangePush("copy_to_device");
    CL_CHECK(clEnqueueWriteBuffer(
        queue, d_a, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), h_a.data(), 0, nullptr, nullptr));

    CL_CHECK(clEnqueueWriteBuffer(
        queue, d_b, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), h_b.data(), 0, nullptr, nullptr));
    roctxRangePop();

    // Set kernel arguments
    roctxRangePush("set_kernel_args");
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c));
    cl_uint vector_size = VECTOR_SIZE;
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uint), &vector_size));
    roctxRangePop();

    // Execute kernel
    roctxRangePush("execute_kernel");
    size_t global_size = VECTOR_SIZE;
    size_t local_size  = GROUP_SIZE;
    CL_CHECK(clEnqueueNDRangeKernel(
        queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr));
    CL_CHECK(clFinish(queue));
    roctxRangePop();

    // Copy result back to host
    roctxRangePush("copy_to_host");
    CL_CHECK(clEnqueueReadBuffer(
        queue, d_c, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), h_c.data(), 0, nullptr, nullptr));
    roctxRangePop();

    // Verify results
    roctxRangePush("verify_results");
    bool   passed = true;
    size_t errors = 0;
    for(size_t i = 0; i < VECTOR_SIZE; i++)
    {
        float expected = h_a[i] + h_b[i];
        if(std::fabs(h_c[i] - expected) > 1e-5)
        {
            passed = false;
            if(errors < 10)
            {
                auto _print_lk = auto_lock_t{print_lock};
                std::cerr << "Error at index " << i << ": expected " << expected << ", got "
                          << h_c[i] << "\n";
            }
            errors++;
        }
    }
    roctxRangePop();

    {
        auto _print_lk = auto_lock_t{print_lock};
        if(passed)
        {
            std::cout << "PASSED: Vector addition verified successfully!\n";
        }
        else
        {
            std::cout << "FAILED: Found " << errors << " errors in vector addition.\n";
        }
    }

    // Cleanup
    roctxRangePush("cleanup");
    CL_CHECK(clReleaseMemObject(d_a));
    CL_CHECK(clReleaseMemObject(d_b));
    CL_CHECK(clReleaseMemObject(d_c));
    CL_CHECK(clReleaseKernel(kernel));
    CL_CHECK(clReleaseProgram(program));
    CL_CHECK(clReleaseCommandQueue(queue));
    CL_CHECK(clReleaseContext(context));
    roctxRangePop();

    roctxMark("opencl_application_end");

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
