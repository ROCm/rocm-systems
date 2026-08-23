/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <alloc.h>
#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <unordered_map>

#include "TestBed.hpp"
#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

template ncclResult_t ncclCudaMemcpy<float>(float*, float*, size_t);

namespace RcclUnitTesting
{
TEST(Alloc, ncclIbMallocDebugNonZero)
{
    void*  ptr  = nullptr;
    size_t size = 4096;

    ncclResult_t result = ncclIbMalloc(&ptr, size);

    EXPECT_EQ(result, ncclSuccess);
    ASSERT_NE(ptr, nullptr);

    char* char_ptr = static_cast<char*>(ptr);
    for(size_t i = 0; i < size; ++i)
    {
        ASSERT_EQ(char_ptr[i], 0);
    }

    free(ptr);
}

TEST(Alloc, ncclIbMallocDebugZeroSize)
{
    void*        ptr    = (void*)0xdeadbeef;
    ncclResult_t result = ncclIbMalloc(&ptr, 0);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(ptr, nullptr);
}

#if ROCM_VERSION < 71200
// These tests exercise the unsupported-fallback path of ncclCuMemHostAlloc/Free
// that returns ncclInternalError. On ROCm 7.12+ the real implementation is
// compiled in, so the fallback no longer exists and these tests are not applicable.
TEST(Alloc, ncclCuMemHostAlloc)
{
    RUN_ISOLATED_TEST(
        "ncclCuMemHostAlloc",
        []()
        {
            void*        ptr    = NULL;
            void*        handle = NULL;
            size_t       size   = 1024;
            ncclResult_t result = ncclCuMemHostAlloc(&ptr, handle, size);
            ASSERT_EQ(result, ncclInternalError);
        }
    );
}

TEST(Alloc, ncclCuMemHostFree)
{
    RUN_ISOLATED_TEST(
        "ncclCuMemHostFree",
        []()
        {
            void*        dummyPtr = reinterpret_cast<void*>(0x1234);
            ncclResult_t result   = ncclCuMemHostFree(dummyPtr);
            ASSERT_EQ(result, ncclInternalError);
        }
    );
}
#endif // ROCM_VERSION < 71200

#if ROCM_VERSION < 70000
// This test is only valid for ROCm versions < 7.0.0
// In ROCm 7.0.0+, the ncclCuMemAlloc signature changed
TEST(Alloc, ncclCuMemAlloc)
{
    RUN_ISOLATED_TEST(
        "ncclCuMemAlloc",
        []()
        {
            void*                      ptr    = reinterpret_cast<void*>(0x1234);
            void*                      handle = reinterpret_cast<void*>(0x5678);
            size_t                     size   = 1024;
            hipMemAllocationHandleType type   = hipMemHandleTypeNone;
            ncclResult_t               result = ncclCuMemAlloc(&ptr, &handle, type, size, /*manager=*/nullptr);
            EXPECT_EQ(result, ncclInternalError);
        }
    );
}

TEST(Alloc, ncclCuMemFree)
{
    RUN_ISOLATED_TEST(
        "ncclCuMemFree",
        []()
        {
            void*        dummyPtr = reinterpret_cast<void*>(0xdeadbeef);
            ncclResult_t result   = ncclCuMemFree(dummyPtr, /*manager=*/nullptr);
            EXPECT_EQ(result, ncclInternalError);
        }
    );
}

TEST(Alloc, ncclCuMemAllocAddr)
{
    RUN_ISOLATED_TEST(
        "ncclCuMemAllocAddr",
        []()
        {
            void* ptr = reinterpret_cast<void*>(0x1111);
            hipMemGenericAllocationHandle_t handle
                = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1234);
            size_t       size   = 4096;
            ncclResult_t result = ncclCuMemAllocAddr(&ptr, &handle, size);
            ASSERT_EQ(result, ncclInternalError);
        }
    );
}

TEST(Alloc, ncclCuMemFreeAddr)
{
    RUN_ISOLATED_TEST(
        "ncclCuMemFreeAddr",
        []()
        {
            void*        testPtr = reinterpret_cast<void*>(0xbeefcafe);
            ncclResult_t result  = ncclCuMemFreeAddr(testPtr, /*manager=*/nullptr);
            ASSERT_EQ(result, ncclInternalError);
        }
    );
}

TEST(Alloc, ncclCuMemGetAddressRange)
{
    RUN_ISOLATED_TEST(
        "ncclCuMemGetAddressRange",
        []()
        {
            CUdeviceptr  base         = 0;
            size_t       baseSize     = 0;
            int          numSegments  = 0;
            bool         hasSysmem    = false;
            ncclResult_t result       = ncclCuMemGetAddressRange(
                static_cast<CUdeviceptr>(0x1000), 4096, &base, &baseSize, &numSegments, &hasSysmem);
            ASSERT_EQ(result, ncclInternalError);
        }
    );
}
#endif // ROCM_VERSION < 70000

TEST(Alloc, ncclCudaHostCalloc)
{
    RUN_ISOLATED_TEST(
        "ncclCudaHostCalloc",
        []()
        {
            // Initialize HIP device in forked process
            ASSERT_EQ(hipSetDevice(0), hipSuccess);

            constexpr size_t N   = 256;
            float*           ptr = nullptr;

            ncclResult_t result = ncclCudaHostCalloc(&ptr, N);
            ASSERT_EQ(result, ncclSuccess);
            ASSERT_NE(ptr, nullptr);

            // Verify the allocation is host (CPU) memory, not device memory.
            hipPointerAttribute_t attr;
            ASSERT_EQ(hipPointerGetAttributes(&attr, ptr), hipSuccess);
            EXPECT_EQ(attr.type, hipMemoryTypeHost)
                << "ncclCudaHostCalloc should allocate host (CPU) memory";

#if defined(HIP_HOST_UNCACHED_MEMORY)
            // Verify the uncached flag was actually honored, not just requested.
            unsigned int flags = 0;
            ASSERT_EQ(hipHostGetFlags(&flags, ptr), hipSuccess);
            EXPECT_TRUE(flags & hipHostMallocUncached)
                << "HIP_HOST_UNCACHED_MEMORY is enabled but ncclCudaHostCalloc did not "
                   "request uncached host memory (flags = "
                << flags << ")";
#endif

            // The memory must be directly readable/writable from the CPU.
            for(size_t i = 0; i < N; ++i)
                EXPECT_EQ(ptr[i], 0.0f) << "Host memory not zero-initialized at index " << i;
            for(size_t i = 0; i < N; ++i)
                ptr[i] = static_cast<float>(i + 1);
            for(size_t i = 0; i < N; ++i)
                EXPECT_EQ(ptr[i], static_cast<float>(i + 1)) << "Host write/read mismatch at index " << i;

            ASSERT_EQ(ncclCudaHostFree(ptr), ncclSuccess);
        }
    );
}

TEST(Alloc, NcclCudaMemcpy)
{
    RUN_ISOLATED_TEST(
        "NcclCudaMemcpy",
        []()
        {
            // Initialize HIP device in forked process
            ASSERT_EQ(hipSetDevice(0), hipSuccess);

            constexpr size_t N     = 128;
            float *          d_src = nullptr, *d_dst = nullptr;
            float            h_src[N], h_dst[N];

            for(size_t i = 0; i < N; ++i)
                h_src[i] = static_cast<float>(i + 1);
            // Allocate device memory

            ASSERT_EQ(hipMalloc(&d_src, N * sizeof(float)), hipSuccess);
            ASSERT_EQ(hipMalloc(&d_dst, N * sizeof(float)), hipSuccess);

            // Copy from host to device (source buffer)
            ASSERT_EQ(
                hipMemcpy(d_src, h_src, N * sizeof(float), hipMemcpyHostToDevice),
                hipSuccess
            );

            // Perform the tested function
            ncclResult_t result = ncclCudaMemcpy<float>(d_dst, d_src, N);

            ASSERT_EQ(result, ncclSuccess);

            // Copy result back to host
            ASSERT_EQ(
                hipMemcpy(h_dst, d_dst, N * sizeof(float), hipMemcpyDeviceToHost),
                hipSuccess
            );

            // Check correctness
            for(size_t i = 0; i < N; ++i)
            {
                EXPECT_EQ(h_src[i], h_dst[i]) << "Mismatch at index " << i;
            }
            // Free memory
            ASSERT_EQ(hipFree(d_src), hipSuccess);
            ASSERT_EQ(hipFree(d_dst), hipSuccess);
        }
    );
}

TEST(Alloc, ZeroElementMemcpy)
{
    RUN_ISOLATED_TEST(
        "ZeroElementMemcpy",
        []()
        {
            // Initialize HIP device in forked process
            ASSERT_EQ(hipSetDevice(0), hipSuccess);

            float *d_src = nullptr, *d_dst = nullptr;
            ASSERT_EQ(hipMalloc(&d_src, sizeof(float)), hipSuccess);
            ASSERT_EQ(hipMalloc(&d_dst, sizeof(float)), hipSuccess);

            ncclResult_t result = ncclCudaMemcpy<float>(d_dst, d_src, 0);
            EXPECT_EQ(result, ncclSuccess) << "Zero-element copy should succeed (no-op)";

            ASSERT_EQ(hipFree(d_src), hipSuccess);
            ASSERT_EQ(hipFree(d_dst), hipSuccess);
        }
    );
}

TEST(Alloc, MemcpyNullSrcOrDstPointer)
{
    RUN_ISOLATED_TEST(
        "MemcpyNullSrcOrDstPointer",
        []()
        {
            // Initialize HIP device in forked process
            ASSERT_EQ(hipSetDevice(0), hipSuccess);

            constexpr size_t N       = 16;
            float*           d_valid = nullptr;
            ASSERT_EQ(hipMalloc(&d_valid, N * sizeof(float)), hipSuccess);

            // Case 1: src is nullptr
            ncclResult_t result = ncclCudaMemcpy<float>(d_valid, nullptr, N);
            EXPECT_EQ(result, ncclUnhandledCudaError)
                << "Expected ncclUnhandledCudaError when src is nullptr";

            // Case 2: dst is nullptr
            result = ncclCudaMemcpy<float>(nullptr, d_valid, N);
            EXPECT_EQ(result, ncclUnhandledCudaError)
                << "Expected ncclUnhandledCudaError when dst is nullptr";

            ASSERT_EQ(hipFree(d_valid), hipSuccess);
        }
    );
}

// ---------------------------------------------------------------------------
// Scoped side-stream pool (alloc.h: ncclSideStreamAcquire / ncclSideStreamRelease
// / getSideStream / ncclClampStreamPriority / ncclSideStreamScope).
//
// The feature under test: side streams are created on first acquire and
// destroyed on last release so that no side stream (and thus no scarce GPU
// hardware queue) persists through the steady-state collective phase. These
// whitebox tests drive the internal pool API directly and inspect its state
// via getSideStream(), which returns the pooled stream only while a scope is
// active and nullptr otherwise.
// ---------------------------------------------------------------------------

// Core guarantee: with no active scope the pool holds nothing, so a side stream
// never lingers to occupy a HW queue during collectives. Acquiring makes the
// pooled stream visible; releasing the last ref destroys it and getSideStream
// reports nullptr again.
TEST(Alloc, SideStreamScopeGatesPool)
{
    RUN_ISOLATED_TEST(
        "SideStreamScopeGatesPool",
        []()
        {
            ASSERT_EQ(hipSetDevice(0), hipSuccess);
            constexpr int dev = 0;

            // No scope active yet: pool must be empty.
            hipStream_t stream = reinterpret_cast<hipStream_t>(0xdeadbeef);
            ASSERT_EQ(getSideStream(&stream), ncclSuccess);
            EXPECT_EQ(stream, nullptr) << "Side stream must not exist before any acquire";

            // Acquire: stream is created and pooled.
            ASSERT_EQ(ncclSideStreamAcquire(dev), ncclSuccess);
            stream = nullptr;
            ASSERT_EQ(getSideStream(&stream), ncclSuccess);
            EXPECT_NE(stream, nullptr) << "Side stream must be available inside an active scope";

            // Release the last ref: stream is destroyed, HW queue freed.
            ASSERT_EQ(ncclSideStreamRelease(dev), ncclSuccess);
            stream = reinterpret_cast<hipStream_t>(0xdeadbeef);
            ASSERT_EQ(getSideStream(&stream), ncclSuccess);
            EXPECT_EQ(stream, nullptr) << "Side stream must be destroyed after last release";
        }
    );
}

// Nested acquires for the same (dev, priority) share one pooled stream and it is
// destroyed only when the final ref is released (reference counting).
TEST(Alloc, SideStreamRefCountPooling)
{
    RUN_ISOLATED_TEST(
        "SideStreamRefCountPooling",
        []()
        {
            ASSERT_EQ(hipSetDevice(0), hipSuccess);
            constexpr int dev = 0;

            ASSERT_EQ(ncclSideStreamAcquire(dev), ncclSuccess);
            hipStream_t first = nullptr;
            ASSERT_EQ(getSideStream(&first), ncclSuccess);
            ASSERT_NE(first, nullptr);

            // Second acquire reuses the same stream (no churn), bumping the refcount.
            ASSERT_EQ(ncclSideStreamAcquire(dev), ncclSuccess);
            hipStream_t second = nullptr;
            ASSERT_EQ(getSideStream(&second), ncclSuccess);
            EXPECT_EQ(second, first) << "Overlapping scopes must reuse the pooled stream";

            // First release: refcount drops to 1, stream still alive.
            ASSERT_EQ(ncclSideStreamRelease(dev), ncclSuccess);
            hipStream_t stillHere = nullptr;
            ASSERT_EQ(getSideStream(&stillHere), ncclSuccess);
            EXPECT_EQ(stillHere, first) << "Stream must survive while a ref remains";

            // Final release: refcount hits 0, stream destroyed.
            ASSERT_EQ(ncclSideStreamRelease(dev), ncclSuccess);
            hipStream_t gone = reinterpret_cast<hipStream_t>(0xdeadbeef);
            ASSERT_EQ(getSideStream(&gone), ncclSuccess);
            EXPECT_EQ(gone, nullptr) << "Stream must be destroyed once all refs are released";
        }
    );
}

// The RAII ncclSideStreamScope holds a ref for its lifetime and releases it on
// destruction, so allocations inside the scope share the pooled stream and the
// HW queue is freed as soon as the scope exits.
TEST(Alloc, SideStreamScopeRAII)
{
    RUN_ISOLATED_TEST(
        "SideStreamScopeRAII",
        []()
        {
            ASSERT_EQ(hipSetDevice(0), hipSuccess);
            constexpr int dev = 0;

            {
                ncclSideStreamScope scope(dev);
                hipStream_t         inScope = nullptr;
                ASSERT_EQ(getSideStream(&inScope), ncclSuccess);
                EXPECT_NE(inScope, nullptr) << "Scope must acquire the pooled stream on entry";
            }

            // Scope destroyed: pooled stream released and destroyed.
            hipStream_t afterScope = reinterpret_cast<hipStream_t>(0xdeadbeef);
            ASSERT_EQ(getSideStream(&afterScope), ncclSuccess);
            EXPECT_EQ(afterScope, nullptr) << "Scope must release the pooled stream on exit";
        }
    );
}

// ncclClampStreamPriority folds an out-of-range priority into the device's
// supported [greatest, least] window and leaves in-range values untouched.
TEST(Alloc, ClampStreamPriority)
{
    RUN_ISOLATED_TEST(
        "ClampStreamPriority",
        []()
        {
            ASSERT_EQ(hipSetDevice(0), hipSuccess);

            int least = 0, greatest = 0;
            ASSERT_EQ(hipDeviceGetStreamPriorityRange(&least, &greatest), hipSuccess);
            // Convention: 'greatest' is the most-prioritized (numerically smallest).
            ASSERT_LE(greatest, least);

            // Below range clamps up to greatest; above range clamps down to least.
            EXPECT_EQ(ncclClampStreamPriority(greatest - 100), greatest);
            EXPECT_EQ(ncclClampStreamPriority(least + 100), least);

            // Endpoints and any in-range value are returned unchanged.
            EXPECT_EQ(ncclClampStreamPriority(greatest), greatest);
            EXPECT_EQ(ncclClampStreamPriority(least), least);
            EXPECT_EQ(ncclClampStreamPriority((greatest + least) / 2), (greatest + least) / 2);
        }
    );
}

// Keys with the same busId but different priorities must remain distinct.
TEST(Alloc, SideStreamKeySeparatesPriorities)
{
    constexpr int64_t                                                        busId = 0x1234;
    std::unordered_map<ncclSideStreamKey, int, ncclSideStreamKeyHash> streams;
    streams.emplace(ncclSideStreamKey{busId, -1}, 1);
    streams.emplace(ncclSideStreamKey{busId, 0}, 2);
    ASSERT_EQ(streams.size(), 2);
    EXPECT_EQ(streams.at(ncclSideStreamKey{busId, -1}), 1);
    EXPECT_EQ(streams.at(ncclSideStreamKey{busId, 0}), 2);
}

// When the device supports multiple priorities, the pool creates and returns
// separate streams for the same device at each priority.
TEST(Alloc, SideStreamPoolSeparatesPriorities)
{
    RUN_ISOLATED_TEST(
        "SideStreamPoolSeparatesPriorities",
        []()
        {
            ASSERT_EQ(hipSetDevice(0), hipSuccess);
            constexpr int dev = 0;

            int least = 0, greatest = 0;
            ASSERT_EQ(hipDeviceGetStreamPriorityRange(&least, &greatest), hipSuccess);
            ASSERT_LE(greatest, least);
            if(greatest == least)
            {
                GTEST_SKIP() << "Device exposes only one stream priority";
            }

            ASSERT_EQ(ncclSideStreamAcquire(dev, greatest), ncclSuccess);
            ASSERT_EQ(ncclSideStreamAcquire(dev, least), ncclSuccess);

            hipStream_t greatestStream = nullptr;
            hipStream_t leastStream    = nullptr;
            ASSERT_EQ(getSideStream(&greatestStream, greatest), ncclSuccess);
            ASSERT_EQ(getSideStream(&leastStream, least), ncclSuccess);
            ASSERT_NE(greatestStream, nullptr);
            ASSERT_NE(leastStream, nullptr);
            EXPECT_NE(greatestStream, leastStream)
                << "Same busId at different priorities must not share a stream";

            ASSERT_EQ(ncclSideStreamRelease(dev, greatest), ncclSuccess);
            ASSERT_EQ(ncclSideStreamRelease(dev, least), ncclSuccess);

            greatestStream = reinterpret_cast<hipStream_t>(0xdeadbeef);
            leastStream    = reinterpret_cast<hipStream_t>(0xdeadbeef);
            ASSERT_EQ(getSideStream(&greatestStream, greatest), ncclSuccess);
            ASSERT_EQ(getSideStream(&leastStream, least), ncclSuccess);
            EXPECT_EQ(greatestStream, nullptr);
            EXPECT_EQ(leastStream, nullptr);
        }
    );
}
} // namespace RcclUnitTesting
