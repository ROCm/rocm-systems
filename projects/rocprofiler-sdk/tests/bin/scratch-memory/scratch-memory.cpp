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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "common/defines.hpp"

#define hipCheckErr(errval)                                                                        \
    do                                                                                             \
    {                                                                                              \
        hipCheckAndFail((errval), __FILE__, __LINE__);                                             \
    } while(0)

#define hipCheckLastError()                                                                        \
    do                                                                                             \
    {                                                                                              \
        hipCheckErr(hipGetLastError());                                                            \
    } while(0)

#define HSA_CALL2(cmd)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hsa_status_t error = (cmd);                                                                \
        if(error != HSA_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            const char* errorStr;                                                                  \
            hsa_status_string(error, &errorStr);                                                   \
            std::cout << "Encountered HSA error (" << errorStr << ") at line " << __LINE__         \
                      << " in file " << __FILE__ << "\n";                                          \
            exit(-1);                                                                              \
        }                                                                                          \
    } while(0)

namespace
{
inline void
hipCheckAndFail(hipError_t errval, const char* file, int line)
{
    if(errval != hipSuccess)
    {
        std::cerr << "hip error: " << hipGetErrorString(errval) << std::endl;
        std::cerr << "    Location: " << file << ":" << line << std::endl;
        exit(errval);
    }
}

hsa_status_t
find_gpu_agents(hsa_agent_t agent, void* data)
{
    hsa_status_t      status;
    hsa_device_type_t device_type;
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    if(status == HSA_STATUS_SUCCESS && device_type == HSA_DEVICE_TYPE_GPU)
    {
        std::vector<hsa_agent_t>* agents = reinterpret_cast<std::vector<hsa_agent_t>*>(data);
        agents->push_back(agent);
    }
    return HSA_STATUS_SUCCESS;
}
}  // namespace

__global__ void
test_kern_large(uint64_t* output)
{
    uint64_t result = 0;
    int      test[4000];
    memset(test, 5, 4000);
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    *output ^= result;
    *output ^= result;
}

__global__ void
test_kern_medium(uint64_t* output)
{
    uint64_t result = 0;
    int      test[175];
    memset(test, 5, 175);
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    *output ^= result;
    *output ^= result;
}

__global__ void
test_kern_small(uint64_t* output)
{
    uint64_t result = 0;
    int      test[2];
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    *output ^= result;
    *output ^= result;
}

// Checks whether we get a request-more-scratch when grid-x is incremented
int
test_gridx(uint64_t* data_ptr)
{
    *data_ptr = 0;
    printf("Running Medium\n");
    test_kern_medium<<<1000, 1>>>(data_ptr);
    hipCheckErr(hipDeviceSynchronize());
    printf("Running Medium - done\n");

    printf("Running Medium-2 - should trigger more-scratch requests\n");
    test_kern_medium<<<1500, 1>>>(data_ptr);
    hipCheckErr(hipDeviceSynchronize());

    printf("Running Medium-2 - done\n");
    return 0;
}

// 1st allocation should go to primary, then large should still trigger a USO
int
test_primary_then_uso(uint64_t* data_ptr)
{
    printf("Running Medium - all slots\n");
    test_kern_medium<<<10000, 1>>>(data_ptr);
    hipCheckErr(hipDeviceSynchronize());
    printf("Running Medium - done\n");

    printf("Running Large - should trigger USO\n");
    test_kern_large<<<1100, 1>>>(data_ptr);
    hipCheckErr(hipDeviceSynchronize());
    printf("Running Large - done\n");
    return 0;
}

int
test_scratch()
{
    uint64_t* data_ptr = nullptr;
    hipCheckErr(HIP_HOST_ALLOC_FUNC(&data_ptr, sizeof(uint64_t), 0));

    auto   host_floats = std::vector<float>(1024, 0.0f);
    float* dev         = nullptr;

    std::iota(host_floats.begin(), host_floats.end(), 1.0f);

    hipCheckErr(hipMalloc((void**) &dev, host_floats.size() * sizeof(float)));
    hipCheckErr(hipMemcpy(
        dev, host_floats.data(), host_floats.size() * sizeof(float), hipMemcpyHostToDevice));

    *data_ptr = 0;

    printf("Running test_primary_then_uso========================\n");
    test_primary_then_uso(data_ptr);
    printf("=====================================================\n");

    printf("Running test_gridx===================================\n");
    test_gridx(data_ptr);
    printf("=====================================================\n");

    printf("Running Small\n");
    test_kern_small<<<1000, 1>>>(data_ptr);
    hipCheckErr(hipDeviceSynchronize());
    printf("Running Small - done\n");

    printf("Running Medium\n");
    test_kern_medium<<<1000, 1>>>(data_ptr);
    hipCheckErr(hipDeviceSynchronize());
    printf("Running Medium - done\n");

    printf("Running Small\n");
    test_kern_small<<<1000, 1>>>(data_ptr);
    hipCheckErr(hipDeviceSynchronize());
    printf("Running Small - done\n");

    printf("Running Large\n");
    test_kern_large<<<1100, 1>>>(data_ptr);
    hipCheckErr(hipDeviceSynchronize());
    printf("Running Large - done\n");

    printf("Running Large\n");
    test_kern_large<<<1000, 1>>>(data_ptr);
    hipCheckErr(hipDeviceSynchronize());
    printf("Running Large - done\n");

    printf("Running Large\n");
    test_kern_large<<<1000, 1>>>(data_ptr);
    hipCheckErr(hipFree(dev));
    hipCheckErr(hipDeviceSynchronize());
    printf("Running Large - done\n");

    return 0;
}

int
main()
{
    hipCheckErr(hipInit(0));

    std::vector<hsa_agent_t> agents;
    HSA_CALL2(hsa_iterate_agents(find_gpu_agents, &agents));
    size_t numAgents = agents.size();
    printf("Detected %ld agents\n", numAgents);

    for(size_t i = 0; i < agents.size(); ++i)
    {
        printf("Testing scratch on device %zu\n", i);
        hipCheckErr(hipSetDevice(i));
        test_scratch();
    }

    return 0;
}
