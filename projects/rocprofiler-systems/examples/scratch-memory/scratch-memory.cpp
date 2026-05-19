// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <numeric>
#include <vector>

#define hipCheckErr(errval)                                                              \
    do                                                                                   \
    {                                                                                    \
        hipCheckAndFail((errval), __FILE__, __LINE__);                                   \
    } while(0)

#define HSA_CALL2(cmd)                                                                   \
    do                                                                                   \
    {                                                                                    \
        hsa_status_t error = (cmd);                                                      \
        if(error != HSA_STATUS_SUCCESS)                                                  \
        {                                                                                \
            const char* errorStr;                                                        \
            hsa_status_string(error, &errorStr);                                         \
            std::cout << "Encountered HSA error (" << errorStr << ") at line "           \
                      << __LINE__ << " in file " << __FILE__ << "\n";                    \
            exit(-1);                                                                    \
        }                                                                                \
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
        std::vector<hsa_agent_t>* agents =
            reinterpret_cast<std::vector<hsa_agent_t>*>(data);
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
    memset(test, 5, sizeof(test));
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    // Double XOR cancels out but forces the compiler to keep the computation
    *output ^= result;
    *output ^= result;
}

__global__ void
test_kern_medium(uint64_t* output)
{
    uint64_t result = 0;
    int      test[175];
    memset(test, 5, sizeof(test));
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    // Double XOR cancels out but forces the compiler to keep the computation
    *output ^= result;
    *output ^= result;
}

__global__ void
test_kern_small(uint64_t* output)
{
    uint64_t result  = 0;
    int      test[2] = { 0 };
    for(int& i : test)
    {
        i = i + 7;
        *output += i;
        result += i;
    }
    // Double XOR cancels out but forces the compiler to keep the computation
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
    hipCheckErr(hipHostMalloc(&data_ptr, sizeof(uint64_t), 0));
    *data_ptr = 0;

    auto host_floats = std::vector<float>(1024, 0.0f);
    std::iota(host_floats.begin(), host_floats.end(), 1.0f);

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
    hipCheckErr(hipDeviceSynchronize());
    printf("Running Large - done\n");

    hipCheckErr(hipHostFree(data_ptr));

    return 0;
}

int
main()
{
    hipCheckErr(hipInit(0));

    std::vector<hsa_agent_t> agents;
    HSA_CALL2(hsa_iterate_agents(find_gpu_agents, &agents));
    size_t numAgents = agents.size();
    printf("Detected %zu agents\n", numAgents);

    for(size_t i = 0; i < agents.size(); ++i)
    {
        printf("Testing scratch on device %zu\n", i);
        hipCheckErr(hipSetDevice(i));
        test_scratch();
    }

    return 0;
}
