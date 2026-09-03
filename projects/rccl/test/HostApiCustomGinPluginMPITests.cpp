/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file HostApiCustomGinPluginMPITests.cpp
 * @brief Regression test for NCCL 2.30.7 fix: one-sided host APIs with a custom GIN plugin.
 *
 * When NCCL_GIN_PLUGIN loads the stub example plugin (NCCL_NET_DEVICE_GIN_PROXY),
 * NCCL must skip it and use the built-in GIN proxy over the RMA backend so host
 * one-sided operations (ncclPutSignal / ncclWaitSignal) keep working.
 *
 * Run (example):
 *   mpirun -np 2 ./rccl-UnitTestsMPI \
 *     --gtest_filter=HostApiCustomGinPluginTest.PutSignalWithExternalGinPlugin
 */

#if defined(MPI_TESTS_ENABLED) && defined(RCCL_ENABLE_HOST_API_TESTS)

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "HostApiHelpers.hpp"
#include "TestChecks.hpp"

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <unistd.h>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLHostApiHelpers;

namespace RcclUnitTesting
{

namespace
{

constexpr size_t kTransferSize = 256 * 1024;
constexpr size_t kOneMB        = 2 * kTransferSize;
constexpr size_t kSendOffset   = 0;
constexpr size_t kRecvOffset   = kTransferSize;
constexpr int    kSigIdx       = 0;
constexpr int    kCtx          = 0;
constexpr unsigned int kFlags  = 0;

#ifdef RCCL_GIN_EXAMPLE_PLUGIN_DIR
constexpr const char* kGinExamplePluginDir = RCCL_GIN_EXAMPLE_PLUGIN_DIR;
#else
constexpr const char* kGinExamplePluginDir = nullptr;
#endif

std::string ginExamplePluginPath()
{
    if(!kGinExamplePluginDir || kGinExamplePluginDir[0] == '\0')
        return {};
    return std::string(kGinExamplePluginDir) + "/libnccl-gin-example.so";
}

bool fileExists(const std::string& path)
{
    return !path.empty() && access(path.c_str(), F_OK) == 0;
}

void prependLdLibraryPath(const char* dir)
{
    if(!dir || dir[0] == '\0')
        return;

    const char* existing = std::getenv("LD_LIBRARY_PATH");
    std::string updated  = dir;
    if(existing && existing[0] != '\0')
    {
        updated += ':';
        updated += existing;
    }
    ASSERT_EQ(setenv("LD_LIBRARY_PATH", updated.c_str(), 1), 0);
}

void configureCustomGinPluginEnv()
{
    ASSERT_EQ(setenv("NCCL_GIN_PLUGIN", "example", 1), 0);
    ASSERT_EQ(setenv("NCCL_GIN_ENABLE", "1", 1), 0);
    ASSERT_EQ(setenv("NCCL_CUMEM_ENABLE", "1", 1), 0);

    if(kGinExamplePluginDir && kGinExamplePluginDir[0] != '\0')
        prependLdLibraryPath(kGinExamplePluginDir);
}

} // namespace

/**
 * @class HostApiCustomGinPluginTest
 * @brief Host API tests with NCCL_GIN_PLUGIN=example loaded before comm init.
 */
class HostApiCustomGinPluginTest : public MPITestBase
{
protected:
    void SetUp() override
    {
        const std::string pluginPath = ginExamplePluginPath();
        if(pluginPath.empty() || !fileExists(pluginPath))
        {
            GTEST_SKIP() << "libnccl-gin-example.so not built; rebuild tests with ENABLE_HOST_API_TESTS=ON";
        }

        configureCustomGinPluginEnv();

        MPITestBase::SetUp();
        ASSERT_EQ(ncclSuccess, createTestCommunicator());
        if(!getActiveCommunicator()->hostRmaSupport)
        {
            GTEST_SKIP() << "Host one-sided RMA is unavailable: run with a supported RMA plugin, "
                            "NCCL_CUMEM_ENABLE=1, and NCCL_NUM_RMA_CTX>0 on multi-node IB hardware";
        }
    }

    int rank() const
    {
        int r = -1;
        ncclCommUserRank(const_cast<HostApiCustomGinPluginTest*>(this)->getActiveCommunicator(), &r);
        return r;
    }
};

/**
 * @test HostApiCustomGinPluginTest.PutSignalWithExternalGinPlugin
 * @brief ncclPutSignal + ncclWaitSignal while a custom GIN proxy plugin is loaded.
 *
 * Regression for NCCL 2.30.7: external GIN proxy plugins must be skipped so host
 * one-sided APIs route through the built-in GIN proxy over the RMA backend.
 */
TEST_F(HostApiCustomGinPluginTest, PutSignalWithExternalGinPlugin)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int      myRank = rank();
    ncclComm_t     comm   = getActiveCommunicator();
    hipStream_t    stream = getActiveStream();
    const int      winFlags = NCCL_WIN_COLL_SYMMETRIC;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, winFlags);

    ASSERT_MPI_NE(win, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    ncclResult_t putRes = ncclSuccess;
    if(myRank == 0)
    {
        void* srcBuf = static_cast<uint8_t*>(winBuf) + kSendOffset;
        FillBuf(srcBuf, kTransferSize, /*senderRank=*/0);
        putRes = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/kRecvOffset,
            kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, putRes);

    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{/*opCnt=*/1, /*peer=*/0, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(/*nDesc=*/1, &desc, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    const bool ok = (myRank != 1) ||
                    VerifyBuf(static_cast<const uint8_t*>(winBuf) + kRecvOffset, kTransferSize, /*seed=*/0);
    ASSERT_MPI_TRUE(ok);

    TEST_INFO("HostApiCustomGinPlugin rank %d: PutSignalWithExternalGinPlugin passed.", myRank);
}

} // namespace RcclUnitTesting

#endif // MPI_TESTS_ENABLED && RCCL_ENABLE_HOST_API_TESTS
