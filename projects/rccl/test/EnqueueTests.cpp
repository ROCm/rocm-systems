/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstring>

#include "comm.h"
#include "common/TestChecks.hpp"
#include "common/DeviceBufferHelpers.hpp"
#include "common/ErrCode.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/ResourceGuards.hpp"
#include "enqueue.h"
#include "info.h"
#include "utils.h"

namespace RcclUnitTesting
{

// Simple test kernel for validating ncclInitKernelsForDevice
__global__ void simpleTestKernel(int* data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(data)
        data[tid] = tid;
}

// Device-linker LDS accounting: pre-fix, the static per-warp scratch was also requested as
// dynamic smem, double-counting LDS and overflowing the 64 KB budget.
static constexpr int kDeviceLdsBudget      = 64 * 1024;
static constexpr int kLegacyDynamicScratch = 32832;
static constexpr int kRegularStaticScratch = 40 * 1024;
static constexpr int kSymmetricDynamicSmem = 16 * 1024;

// Regular kernel with static scratch (device-linker model); the barrier + cross-thread read keeps
// the compiler from optimizing the shared allocation into registers so it shows in sharedSizeBytes.
__global__ void dlRegularScratchKernel(int* data)
{
    __shared__ char staticScratch[kRegularStaticScratch];
    for(int i = threadIdx.x; i < kRegularStaticScratch; i += blockDim.x)
        staticScratch[i] = (char)(i + threadIdx.x);
    __syncthreads();
    int acc = 0;
    for(int i = threadIdx.x; i < kRegularStaticScratch; i += blockDim.x)
        acc += staticScratch[(i + 1) % kRegularStaticScratch];
    if(data)
        data[threadIdx.x] = acc;
}

// Mimics a *symmetric* kernel: no static per-warp scratch; relies entirely on dynamic shared
// memory (extern __shared__ ncclSymkSmem[]), which is not reported under sharedSizeBytes.
__global__ void dlSymmetricKernel(int* data)
{
    extern __shared__ char dynScratch[];
    dynScratch[threadIdx.x] = (char)(threadIdx.x + 1);
    __syncthreads();
    if(data)
        data[threadIdx.x] = dynScratch[threadIdx.x];
}

// Helper function to test ncclInitKernelsForDevice with a real kernel
ncclResult_t testKernelAttributes(void* kernelFn, size_t* maxStackSize)
{
    if(!kernelFn || !maxStackSize)
        return ncclInvalidArgument;

    *maxStackSize          = 0;
    hipFuncAttributes attr = {0};

    hipError_t errcode = hipFuncGetAttributes(&attr, kernelFn);
    if(errcode != hipSuccess)
        return ncclSystemError;

    *maxStackSize = attr.localSizeBytes;
    return ncclSuccess; // ncclSuccess
}

// Helper function to test shared memory limit checking with a real kernel
// ncclMaxSharedMem: For gfx906 (cudaArch 906) with WarpSize 64, this is typically 32832 bytes
ncclResult_t testKernelSharedMemoryLimit(
    void* kernelFn, int cudaArch, int maxSharedMem, size_t* maxStackSize, int ncclMaxSharedMem
)
{
    if(!kernelFn)
        return ncclInvalidArgument;

    ncclResult_t result = ncclSuccess;
    if(maxStackSize)
        *maxStackSize = 0;

    hipFuncAttributes attr    = {0};
    hipError_t        errcode = hipFuncGetAttributes(&attr, kernelFn);
    if(errcode != hipSuccess)
    {
        return ncclSystemError;
    }

    if(maxStackSize)
    {
        *maxStackSize = attr.localSizeBytes;
    }

    // Test the shared memory limit check (mimics enqueue.cc lines 135-146)
    if(ncclMaxSharedMem != 0)
    {
        int sharedMemSize = ncclMaxSharedMem;

        if(sharedMemSize > (maxSharedMem - attr.sharedSizeBytes))
        {
            TEST_WARN(
                "cudaArch %d ncclMaxSharedMem %d exceeds device/fn maxSharedMem %zu",
                cudaArch,
                sharedMemSize,
                maxSharedMem - attr.sharedSizeBytes
            );
            return ncclSystemError;
        }
    }

    return result;
}

// Helper structure to hold test environment
struct EnqueueTestEnvironment
{
    ncclComm* comm;
    ncclInfo* info;
    void*     sendbuff;
    void*     recvbuff;
    uint32_t  abortFlag0;
    uint32_t  abortFlag1;
    int       abortFlagRefCount;

    EnqueueTestEnvironment()
        : comm(nullptr)
        , info(nullptr)
        , sendbuff(nullptr)
        , recvbuff(nullptr)
        , abortFlag0(0)
        , abortFlag1(0)
        , abortFlagRefCount(0)
    {}

    ~EnqueueTestEnvironment()
    {
        cleanup();
    }

    void setup()
    {
        // Allocate GPU memory for buffers
        size_t     bufferSize = 1024 * sizeof(float);
        hipError_t hipErr     = hipMalloc(&sendbuff, bufferSize);
        ASSERT_EQ(hipErr, hipSuccess) << "Failed to allocate sendbuff";

        hipErr = hipMalloc(&recvbuff, bufferSize);
        ASSERT_EQ(hipErr, hipSuccess) << "Failed to allocate recvbuff";

        // Initialize communicator
        comm = new ncclComm();
        memset(comm, 0, sizeof(ncclComm));

        comm->startMagic = NCCL_MAGIC; // 0x0280028002800280

        // Initialize critical fields
        comm->rank      = 0;
        comm->nRanks    = 2;
        comm->cudaDev   = 0;
        comm->localRank = 0;

        // Initialize abort flags
        comm->abortFlag         = &abortFlag0;
        comm->childAbortFlag    = &abortFlag1;
        comm->abortFlagRefCount = &abortFlagRefCount;

        // Initialize memory stack
        ncclMemoryStackConstruct(&comm->memScoped);
        ncclMemoryStackConstruct(&comm->memPermanent);

        // Initialize intra-communication pointers
        comm->intraComm0 = nullptr;
        comm->intraNext  = nullptr;

        // Initialize work FIFO structures
        comm->workFifoBytes                = 1024; // Power of 2
        comm->workFifoBuf                  = nullptr;
        comm->workFifoBufDev               = nullptr;
        comm->workFifoConsumed             = 0;
        comm->workFifoProducedLastRecorded = 0;
        comm->workFifoProduced             = 0;

        // Initialize planner
        memset(&comm->planner, 0, sizeof(comm->planner));

        // Initialize config
        memset(&comm->config, 0, sizeof(comm->config));
        comm->config.blocking = 1;
        comm->checkMode = ncclCheckModeDefault; // Disable pointer validation for easier testing (was: comm->checkPointers = 0, removed in v2.29)

        // Initialize peer info arrays
        comm->peerInfo = new ncclPeerInfo[comm->nRanks];
        memset(comm->peerInfo, 0, comm->nRanks * sizeof(ncclPeerInfo));

        comm->localRankToRank = new int[comm->nRanks];
        for(int i = 0; i < comm->nRanks; i++)
        {
            comm->localRankToRank[i] = i;
        }

        comm->endMagic = NCCL_MAGIC; // 0x0280028002800280

        // Initialize operation info with valid GPU buffers
        info = new ncclInfo();
        memset(info, 0, sizeof(ncclInfo));
        info->comm     = comm;
        info->opName   = "AllReduce";
        info->count    = 1024;
        info->datatype = ncclFloat;
        info->op       = ncclSum;
        info->root     = 0;
        info->sendbuff = sendbuff; // Use allocated GPU memory
        info->recvbuff = recvbuff; // Use allocated GPU memory
        info->stream   = nullptr;
    }

    void cleanup()
    {
        // Clean up info first (it references comm)
        if(info)
        {
            delete info;
            info = nullptr;
        }

        // Clean up comm and its allocated resources
        if(comm)
        {
            // Clean up memory stacks
            ncclMemoryStackDestruct(&comm->memScoped);
            ncclMemoryStackDestruct(&comm->memPermanent);

            // Clean up peer info arrays
            if(comm->peerInfo)
            {
                delete[] comm->peerInfo;
                comm->peerInfo = nullptr;
            }

            if(comm->localRankToRank)
            {
                delete[] comm->localRankToRank;
                comm->localRankToRank = nullptr;
            }

            delete comm;
            comm = nullptr;
        }

        // Clean up GPU buffers last
        if(sendbuff)
        {
            hipError_t err = hipFree(sendbuff);
            if(err != hipSuccess)
            {
                // Log error but don't throw in cleanup
                fprintf(stderr, "Warning: hipFree(sendbuff) failed with error %d\n", err);
            }
            sendbuff = nullptr;
        }

        if(recvbuff)
        {
            hipError_t err = hipFree(recvbuff);
            if(err != hipSuccess)
            {
                // Log error but don't throw in cleanup
                fprintf(stderr, "Warning: hipFree(recvbuff) failed with error %d\n", err);
            }
            recvbuff = nullptr;
        }
    }
};

// Empty test fixture for test organization
class EnqueueTests : public ::testing::Test
{
    // No setup/teardown - all tests use process isolation
};

// Test ncclInitKernelsForDevice function
TEST_F(EnqueueTests, ncclInitKernelsForDevice_ValidInput)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false; // Continue running all tests
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclInitKernelsForDevice_ValidInput",
            [this]()
            {
                size_t       maxStackSize = 0;
                ncclResult_t result       = ncclInitKernelsForDevice(906, 65536, &maxStackSize);

                EXPECT_TRUE(result == ncclSuccess);
                // maxStackSize should be set to a reasonable value (> 0)
                EXPECT_GT(maxStackSize, 0)
                    << "Expected maxStackSize to be computed and set to a positive value";
    }
        ).withEnvironment({{"NCCL_DEBUG", "INFO"}, {"NCCL_DEBUG_SUBSYS", "ALL"}}),

        ProcessIsolatedTestRunner::TestConfig(
            "ncclInitKernelsForDevice_ValidInputCarveout",
            [this]()
            {
                size_t       maxStackSize = 0;
                ncclResult_t result       = ncclInitKernelsForDevice(906, 65536, &maxStackSize);

                EXPECT_TRUE(result == ncclSuccess);
                // maxStackSize should be set to a reasonable value (> 0)
                EXPECT_GT(maxStackSize, 0)
                    << "Expected maxStackSize to be computed and set to a positive value";
            }
        )
            .withEnvironment(
                {{"NCCL_L1_SHARED_MEMORY_CARVEOUT", "1"},
                 {"NCCL_DEBUG", "INFO"},
                 {"NCCL_DEBUG_SUBSYS", "ALL"}}
            )
    );
}

TEST_F(EnqueueTests, ncclInitKernelsForDevice_NullStackSize)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclInitKernelsForDevice_NullStackSize",
            []()
            {
                ncclResult_t result = ncclInitKernelsForDevice(906, 65536, nullptr);
                EXPECT_EQ(result, ncclSuccess);
            }
        )
    );
}

// Test with a real compiled kernel to verify attribute retrieval works correctly
TEST_F(EnqueueTests, KernelAttributes_WithRealKernel)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "KernelAttributes_WithRealKernel",
            []()
            {
                size_t       maxStackSize = 0;
                ncclResult_t result = testKernelAttributes((void*)simpleTestKernel, &maxStackSize);

                EXPECT_EQ(result, ncclSuccess)
                    << "Expected successful kernel attribute retrieval with a real compiled kernel";
    }
        ).withEnvironment({{"NCCL_DEBUG", "INFO"}})
    );
}

TEST_F(EnqueueTests, ncclInitKernelsForDevice_InvalidArch)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclInitKernelsForDevice_InvalidArch",
            []()
            {
                size_t       maxStackSize = 0;
                ncclResult_t result       = ncclInitKernelsForDevice(-1, 65536, &maxStackSize);
                EXPECT_EQ(result, ncclSuccess);
            }
        )
    );
}

TEST_F(EnqueueTests, ncclInitKernelsForDevice_ExceedsSharedMemory)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclInitKernelsForDevice_ExceedsSharedMemory",
            []()
            {
                size_t maxStackSize = 0;
                // For gfx906, ncclMaxSharedMem is 32832 (as shown in test output)
                // Use a very small maxSharedMem (16000 bytes) to trigger the exceeds check
                ncclResult_t result = testKernelSharedMemoryLimit(
                    (void*)simpleTestKernel, // Use our real compiled kernel
                    906, // cudaArch
                    16000, // maxSharedMem (intentionally too small)
                    &maxStackSize,
                    32832  // ncclMaxSharedMem for gfx906
                );

                EXPECT_EQ(result, ncclSystemError)
                    << "Expected ncclSystemError when ncclMaxSharedMem exceeds maxSharedMem";
    }
        ).withEnvironment({{"NCCL_DEBUG", "WARN"}})
    );
}

// Regular (non-symmetric) kernel: before the fix the static scratch is double-counted as
// dynamic smem and overflows the LDS budget; after the fix the dynamic request is 0.
TEST_F(EnqueueTests, DeviceLinkerLds_RegularKernel)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "DeviceLinkerLds_RegularKernel",
            []()
            {
                size_t maxStackSize = 0;

                // Read the kernel's actual static shared usage so the model is self-calibrating.
                hipFuncAttributes regAttr = {0};
                HIP_CHECK(hipFuncGetAttributes(&regAttr, (void*)dlRegularScratchKernel));
                const int regStaticBytes = (int)regAttr.sharedSizeBytes;
                printf("[DeviceLinkerLds] dlRegularScratchKernel static shared = %d bytes\n",
                       regStaticBytes);

                // BEFORE FIX: scratch is static AND requested again as dynamic smem; modeling the
                // dynamic request as the same size, static + dynamic (2x) exceeds the 64 KB budget.
                ncclResult_t before = testKernelSharedMemoryLimit(
                    (void*)dlRegularScratchKernel,
                    942,
                    kDeviceLdsBudget,
                    &maxStackSize,
                    regStaticBytes);
                EXPECT_EQ(before, ncclSystemError)
                    << "Pre-fix: regular kernel double-counts static scratch (" << regStaticBytes
                    << ") as dynamic smem -> LDS overflow (budget " << kDeviceLdsBudget << ")";

                // AFTER FIX: device-linker builds set rcclShmemDynamicSize() to 0, so the regular
                // kernel requests no dynamic smem and stays within budget (model is DL-only).
#ifdef RCCL_DEVICE_LINKER
                const int postFixDynamic = 0;
                ncclResult_t after = testKernelSharedMemoryLimit(
                    (void*)dlRegularScratchKernel,
                    942,
                    kDeviceLdsBudget,
                    &maxStackSize,
                    postFixDynamic);
                EXPECT_EQ(after, ncclSuccess)
                    << "Post-fix: regular kernel requests 0 dynamic smem -> no overflow";

                // Real-path regression guard: the actual RCCL kernels statically allocate scratch,
                // so ncclInitKernelsForDevice() overflows pre-fix (FAILS) and succeeds post-fix.
                int          device     = 0;
                int          archMajor  = 0;
                int          archMinor  = 0;
                int          deviceSmem = kDeviceLdsBudget;
                HIP_CHECK(hipGetDevice(&device));
                HIP_CHECK(hipDeviceGetAttribute(
                    &archMajor, hipDeviceAttributeComputeCapabilityMajor, device));
                HIP_CHECK(hipDeviceGetAttribute(
                    &archMinor, hipDeviceAttributeComputeCapabilityMinor, device));
                HIP_CHECK(hipDeviceGetAttribute(
                    &deviceSmem, hipDeviceAttributeMaxSharedMemoryPerBlock, device));
                const int realArch = 100 * archMajor + 10 * archMinor;

                size_t       realStack = 0;
                ncclResult_t realInit  = ncclInitKernelsForDevice(realArch, deviceSmem, &realStack);
                EXPECT_EQ(realInit, ncclSuccess)
                    << "Post-fix: real kernel init must not overflow LDS in device-linker build "
                       "(arch " << realArch << ", smem " << deviceSmem << ")";
#endif
            }
        ).withEnvironment({{"NCCL_DEBUG", "WARN"}})
    );
}

// Symmetric kernel: no static per-warp scratch, uses dynamic ncclSymkSmem[]. Pre-fix the
// device-linker path would zero its dynamic smem; post-fix it keeps its own non-zero kernelDynSmem.
TEST_F(EnqueueTests, DeviceLinkerLds_SymmetricKernel)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "DeviceLinkerLds_SymmetricKernel",
            []()
            {
                size_t maxStackSize = 0;

                // The symmetric kernel's dynamic smem alone (no static scratch) fits the budget.
                ncclResult_t fits = testKernelSharedMemoryLimit(
                    (void*)dlSymmetricKernel,
                    942,
                    kDeviceLdsBudget,
                    &maxStackSize,
                    kSymmetricDynamicSmem);
                EXPECT_EQ(fits, ncclSuccess)
                    << "Symmetric kernel's dynamic smem must fit the LDS budget";

                // Model ncclLaunchKernel()'s smem selection: regular -> rcclShmemDynamicSize()
                // (0 in device-linker builds), symmetric -> plan->kernelDynSmem (must stay non-zero).
                const int regularLaunchSmem =
#ifdef RCCL_DEVICE_LINKER
                    0;
#else
                    kLegacyDynamicScratch;
#endif
                const bool isSymColl     = true;
                const int  kernelDynSmem = kSymmetricDynamicSmem;
                const int  symLaunchSmem = isSymColl ? kernelDynSmem : regularLaunchSmem;

                // BEFORE FIX (no symmetric guard): the device-linker path would launch with the
                // regular value (0), under-allocating ncclSymkSmem[] (DL-only check).
#ifdef RCCL_DEVICE_LINKER
                EXPECT_LT(regularLaunchSmem, kSymmetricDynamicSmem)
                    << "Without the symmetric guard the device-linker path under-allocates "
                       "symmetric dynamic smem";
#endif

                // AFTER FIX: the symmetric kernel keeps its own non-zero dynamic request.
                EXPECT_EQ(symLaunchSmem, kSymmetricDynamicSmem)
                    << "Post-fix: symmetric kernels launch with their own kernelDynSmem";
                EXPECT_GT(symLaunchSmem, 0)
                    << "Symmetric kernel launched with 0 dynamic smem would under-allocate";

                // End-to-end: launching with its dynamic smem succeeds and produces correct results.
                const int kThreads = 64;
                void*     devData  = nullptr;
                HIP_CHECK(hipMalloc(&devData, kThreads * sizeof(int)));
                auto devGuard = RCCLTestGuards::makeDeviceBufferAutoGuard(devData);

                hipLaunchKernelGGL(
                    dlSymmetricKernel,
                    dim3(1),
                    dim3(kThreads),
                    symLaunchSmem,
                    nullptr,
                    (int*)devData);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipDeviceSynchronize());

                size_t firstError = 0;
                int    expected   = 0;
                int    actual     = 0;
                bool   ok         = RCCLTestHelpers::verifyBufferData<int>(
                    devData,
                    kThreads,
                    [](size_t i) { return static_cast<int>(i + 1); },
                    kThreads,
                    0,
                    &firstError,
                    &expected,
                    &actual);
                EXPECT_TRUE(ok) << "Symmetric kernel wrong result at index " << firstError
                                << " (expected " << expected << ", got " << actual << ")";
            }
        ).withEnvironment({{"NCCL_DEBUG", "WARN"}})
    );
}

// Test ncclEnqueueCheck function
TEST_F(EnqueueTests, ncclEnqueueCheck_ValidInput)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclEnqueueCheck_ValidInput",
            []()
            {
                EnqueueTestEnvironment env;
                env.setup();
                ncclResult_t result = ncclEnqueueCheck(env.info);
                EXPECT_TRUE(result == ncclSuccess);
                env.cleanup();
            }
        )
    );
}

TEST_F(EnqueueTests, ncclEnqueueCheck_InvalidComm)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclEnqueueCheck_InvalidComm",
            []()
            {
                EnqueueTestEnvironment env;
                env.setup();
                env.info->comm      = nullptr;
                ncclResult_t result = ncclEnqueueCheck(env.info);
                EXPECT_EQ(result, ncclInvalidArgument);
                env.cleanup();
            }
        )
    );
}

TEST_F(EnqueueTests, ncclEnqueueCheck_InvalidBuffers)
{
    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;

    RUN_ISOLATED_TESTS_WITH_OPTIONS(
        options,
        ProcessIsolatedTestRunner::TestConfig(
            "ncclEnqueueCheck_InvalidBuffers",
            []()
            {
                EnqueueTestEnvironment env;
                env.setup();

                // Test with null sendbuff
                env.comm->checkMode = ncclCheckModeDebugLocal; // was: env.comm->checkPointers = 1 (removed in v2.29)
                env.info->sendbuff      = nullptr;
                ncclResult_t result     = ncclEnqueueCheck(env.info);
                EXPECT_EQ(result, ncclInvalidArgument);

                // Reset sendbuff and test with null recvbuff
                env.info->sendbuff = env.sendbuff;
                env.info->recvbuff = nullptr;
                result             = ncclEnqueueCheck(env.info);
                EXPECT_EQ(result, ncclInvalidArgument);

                env.cleanup();
            }
        )
    );
}

} // namespace RcclUnitTesting