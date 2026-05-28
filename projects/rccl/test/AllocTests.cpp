/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <alloc.h>

#include "TestBed.hpp"

template ncclResult_t ncclCudaMemcpy<float>(float*, float*, size_t);
namespace RcclUnitTesting
{
    TEST(Alloc, ncclIbMallocDebugNonZero) {
        void* ptr = nullptr;
        size_t size = 4096;

        ncclResult_t result = ncclIbMalloc(&ptr, size);

        EXPECT_EQ(result, ncclSuccess);
        ASSERT_NE(ptr, nullptr);

        char* char_ptr = static_cast<char*>(ptr);
        for (size_t i = 0; i < size; ++i) {
          ASSERT_EQ(char_ptr[i], 0);
        }

        free(ptr);
    }

    TEST(Alloc, ncclIbMallocDebugZeroSize) {
        void* ptr = (void*)0xdeadbeef;
        ncclResult_t result = ncclIbMalloc(&ptr, 0);

        EXPECT_EQ(result, ncclSuccess);
        EXPECT_EQ(ptr, nullptr);
    }


    TEST(Alloc, ncclCuMemHostAlloc) {
        void* ptr = NULL;
        void* handle = NULL;
        size_t size = 1024;
        ncclResult_t result = ncclCuMemHostAlloc(&ptr, handle, size);
        ASSERT_EQ(result, ncclInternalError);
    }

    TEST(Alloc, ncclCuMemHostFree)
    {
        void* dummyPtr = reinterpret_cast<void*>(0x1234); // any dummy address
        ncclResult_t result = ncclCuMemHostFree(dummyPtr);
        ASSERT_EQ(result, ncclInternalError);
    }

#if ROCM_VERSION < 70000
    // This test is only valid for ROCm versions < 7.0.0
    // In ROCm 7.0.0+, the ncclCuMemAlloc signature changed
    TEST(Alloc, ncclCuMemAlloc)
    {
        void* ptr = reinterpret_cast<void*>(0x1234);     // dummy non-null input
        void* handle = reinterpret_cast<void*>(0x5678);  // dummy non-null input
        size_t size = 1024;
        hipMemAllocationHandleType type = hipMemHandleTypeNone;
        ncclResult_t result = ncclCuMemAlloc(&ptr, &handle, type, size);
        EXPECT_EQ(result, ncclInternalError);
    }

    TEST(Alloc, ncclCuMemFree)
    {
        void* dummyPtr = reinterpret_cast<void*>(0xdeadbeef); // arbitrary non-null
        ncclResult_t result = ncclCuMemFree(dummyPtr);
        EXPECT_EQ(result, ncclInternalError);
    }

    TEST(Alloc, ncclCuMemAllocAddr)
    {
        void* ptr = reinterpret_cast<void*>(0x1111);  // Dummy non-null input
        hipMemGenericAllocationHandle_t handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1234);
        size_t size = 4096;
        ncclResult_t result = ncclCuMemAllocAddr(&ptr, &handle, size);
        ASSERT_EQ(result, ncclInternalError);
    }

    TEST(Alloc, ncclCuMemFreeAddr)
    {
        void* testPtr = reinterpret_cast<void*>(0xbeefcafe); // Arbitrary non-null pointer
        ncclResult_t result = ncclCuMemFreeAddr(testPtr);
        ASSERT_EQ(result, ncclInternalError);
    }
#endif // ROCM_VERSION < 70000

    TEST(Alloc, NcclCudaMemcpy) {
        constexpr size_t N = 128;
        float *d_src = nullptr, *d_dst = nullptr;
        float h_src[N], h_dst[N];

        for (size_t i = 0; i < N; ++i) h_src[i] = static_cast<float>(i + 1);
        // Allocate device memory

        ASSERT_EQ(hipMalloc(&d_src, N * sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMalloc(&d_dst, N * sizeof(float)), hipSuccess);

        // Copy from host to device (source buffer)
        ASSERT_EQ(hipMemcpy(d_src, h_src, N * sizeof(float), hipMemcpyHostToDevice), hipSuccess);

        // Perform the tested function
        ncclResult_t result = ncclCudaMemcpy<float>(d_dst, d_src, N);

        ASSERT_EQ(result, ncclSuccess);  // Fixed typo: was ncclSsuccess

        // Copy result back to host
        ASSERT_EQ(hipMemcpy(h_dst, d_dst, N * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);

        // Check correctness
        for (size_t i = 0; i < N; ++i) {
            EXPECT_EQ(h_src[i], h_dst[i]) << "Mismatch at index " << i;
        }
        // Free memory
        hipFree(d_src);
        hipFree(d_dst);

    }

    TEST(Alloc, ZeroElementMemcpy) {
        float *d_src = nullptr, *d_dst = nullptr;
        ASSERT_EQ(hipMalloc(&d_src, sizeof(float)), hipSuccess);
        ASSERT_EQ(hipMalloc(&d_dst, sizeof(float)), hipSuccess);

        ncclResult_t result = ncclCudaMemcpy<float>(d_dst, d_src, 0);
        EXPECT_EQ(result, ncclSuccess) << "Zero-element copy should succeed (no-op)";

        hipFree(d_src);
        hipFree(d_dst);
    }

    TEST(Alloc, MemcpyNullSrcOrDstPointer) {
        constexpr size_t N = 16;
        float* d_valid = nullptr;
        ASSERT_EQ(hipMalloc(&d_valid, N * sizeof(float)), hipSuccess);

        // Case 1: src is nullptr
        ncclResult_t result = ncclCudaMemcpy<float>(d_valid, nullptr, N);
        EXPECT_EQ(result, ncclUnhandledCudaError) << "Expected ncclUnhandledCudaError when src is nullptr";

        // Case 2: dst is nullptr
        result = ncclCudaMemcpy<float>(nullptr, d_valid, N);
        EXPECT_EQ(result, ncclUnhandledCudaError) << "Expected ncclUnhandledCudaError when dst is nullptr";

        hipFree(d_valid);
    }
} //namespace rccl