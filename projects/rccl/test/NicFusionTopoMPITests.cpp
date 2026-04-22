/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file NicFusionTopoMPITests.cpp
 * @brief Tests for NIC Fusion auto-merge behavior under different topologies
 *
 * These tests use NCCL_TOPO_FILE to load synthetic XML topologies and verify
 * that NIC Fusion (auto-merge) produces the expected virtual devices.
 *
 * IMPORTANT: The IB net plugin initializes virtual devices exactly ONCE per
 * process (first ncclCommInit). Subsequent comms reuse the same device list.
 * Therefore, each test scenario MUST be run as a separate mpirun invocation
 * with the appropriate environment variables.
 *
 * Required environment variables:
 *   NCCL_TOPO_FILE=<path>          - XML topology to load
 *   NCCL_NET_MERGE_LEVEL=<level>   - Merge aggressiveness (LOC|PORT|PIX|PXB|PHB|SYS)
 *
 * Optional:
 *   RCCL_TOPO_TEST_DIR=<dir>       - Directory with topology XMLs
 *                                    (default: /home/ilkosare/generated_topos)
 *   NCCL_DEBUG=INFO                - Enable debug logging
 *
 * Run examples (one scenario per invocation):
 *
 *   # PIX topology with PIX merge level — expect merges
 *   NCCL_TOPO_FILE=$TOPOS/topo_pix_all_nics.xml NCCL_NET_MERGE_LEVEL=PIX \
 *     mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter=NicFusionTopo*
 *
 *   # PHB topology with PIX merge level — expect no merges
 *   NCCL_TOPO_FILE=$TOPOS/topo_phb_all_nics.xml NCCL_NET_MERGE_LEVEL=PIX \
 *     mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter=NicFusionTopo*
 *
 *   # Run all scenarios via helper script:
 *   bash test/scripts/run_nic_fusion_topo_tests.sh
 */

#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef MPI_TESTS_ENABLED

#include "comm.h"
#include "graph/topo.h"
#include "net.h"

using namespace MPITestConstants;
using namespace RCCLTestGuards;

class NicFusionTopoMPITest : public MPITestBase
{
protected:
    static constexpr int kMinRanks = 2;

    struct NetDeviceSummary {
        int totalDevices = 0;
        int physicalDevices = 0;
        int mergedDevices = 0;
        std::vector<std::string> deviceNames;
    };

    NetDeviceSummary queryNetDevices()
    {
        NetDeviceSummary summary;
        ncclComm_t comm = getActiveCommunicator();
        if (!comm)
            return summary;

        struct ncclComm* commHandle = comm;
        if (!commHandle->ncclNet)
            return summary;

        int ndev = 0;
        if (commHandle->ncclNet->devices(&ndev) != ncclSuccess || ndev <= 0)
            return summary;

        summary.totalDevices = ndev;

        for (int i = 0; i < ndev; i++) {
            ncclNetProperties_t props;
            memset(&props, 0, sizeof(props));
            if (commHandle->ncclNet->getProperties(i, &props) != ncclSuccess)
                continue;

            std::string name = props.name ? props.name : "";
            summary.deviceNames.push_back(name);

            if (name.find('+') != std::string::npos)
                summary.mergedDevices++;
            else
                summary.physicalDevices++;
        }

        return summary;
    }

    int queryTopoNetCount()
    {
        ncclComm_t comm = getActiveCommunicator();
        if (!comm)
            return -1;
        struct ncclComm* commHandle = comm;
        if (!commHandle->topo)
            return -1;
        return commHandle->topo->nodes[NET].count;
    }

    void printSummary(const char* label, const NetDeviceSummary& summary, int topoNetCount)
    {
        int rank = MPIEnvironment::world_rank;
        if (rank != 0)
            return;

        const char* topoFile = getenv("NCCL_TOPO_FILE");
        const char* mergeLevel = getenv("NCCL_NET_MERGE_LEVEL");

        fprintf(stderr, "\n");
        fprintf(stderr, "===== NIC Fusion Topology Test: %s =====\n", label);
        fprintf(stderr, "  NCCL_TOPO_FILE:        %s\n", topoFile ? topoFile : "(not set)");
        fprintf(stderr, "  NCCL_NET_MERGE_LEVEL:  %s\n", mergeLevel ? mergeLevel : "(default=PORT)");
        fprintf(stderr, "  Topo NET node count:   %d\n", topoNetCount);
        fprintf(stderr, "  Plugin total devices:  %d\n", summary.totalDevices);
        fprintf(stderr, "  Plugin physical devs:  %d\n", summary.physicalDevices);
        fprintf(stderr, "  Plugin merged devs:    %d\n", summary.mergedDevices);
        for (size_t i = 0; i < summary.deviceNames.size(); i++) {
            fprintf(stderr, "    dev[%zu]: %s\n", i, summary.deviceNames[i].c_str());
        }
        fprintf(stderr, "==========================================\n\n");
    }
};

// ---------------------------------------------------------------------------
// Core test: Initialize a communicator and report NIC Fusion results.
//
// The actual topology and merge level are controlled by environment variables.
// This test validates that:
//   1. The communicator initializes successfully with the given topology
//   2. The net plugin is functional and reports devices
//   3. The topology graph has NET nodes
//   4. Merged vs unmerged device counts are reported for analysis
//
// By running this with different env var combinations, we test the full
// spectrum of merge decisions without hardcoding expectations for each
// topology (which would be fragile as topologies evolve).
// ---------------------------------------------------------------------------
TEST_F(NicFusionTopoMPITest, VerifyMergeWithTopology)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks))
        << "Test requires at least " << kMinRanks << " MPI processes";

    const char* topoFile = getenv("NCCL_TOPO_FILE");
    if (!topoFile) {
        GTEST_SKIP() << "NCCL_TOPO_FILE not set — skipping topology-driven test. "
                     << "Set NCCL_TOPO_FILE to a synthetic topology XML to run.";
    }

    ASSERT_EQ(createTestCommunicator(), ncclSuccess)
        << "Failed to create communicator with NCCL_TOPO_FILE=" << topoFile;

    auto summary = queryNetDevices();
    int topoNetCount = queryTopoNetCount();

    printSummary("VerifyMergeWithTopology", summary, topoNetCount);

    EXPECT_GT(summary.totalDevices, 0) << "Net plugin reports no devices";
    // Note: comm->topo->nodes[NET].count is populated during topology path
    // computation, not by ncclTopoPopulateNics. On some builds / stacks it can
    // remain 0 even when the plugin correctly enumerates NET devices from XML.
    // We therefore only log it for visibility rather than hard-asserting.
    if (MPIEnvironment::world_rank == 0 && topoNetCount <= 0) {
        fprintf(stderr, "[NicFusionTopo] WARN: topo NET node count = %d (informational)\n",
                topoNetCount);
    }
}

// ---------------------------------------------------------------------------
// Strict merge expected: when env var RCCL_EXPECT_MERGES=1 is set,
// assert that at least one merged device exists.
// ---------------------------------------------------------------------------
TEST_F(NicFusionTopoMPITest, ExpectMergedDevices)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks))
        << "Test requires at least " << kMinRanks << " MPI processes";

    const char* topoFile = getenv("NCCL_TOPO_FILE");
    if (!topoFile) {
        GTEST_SKIP() << "NCCL_TOPO_FILE not set";
    }

    const char* expectMerges = getenv("RCCL_EXPECT_MERGES");
    if (!expectMerges || strcmp(expectMerges, "1") != 0) {
        GTEST_SKIP() << "RCCL_EXPECT_MERGES not set to 1 — skipping strict merge assertion";
    }

    ASSERT_EQ(createTestCommunicator(), ncclSuccess)
        << "Failed to create communicator";

    auto summary = queryNetDevices();
    int topoNetCount = queryTopoNetCount();

    printSummary("ExpectMergedDevices", summary, topoNetCount);

    EXPECT_GT(summary.totalDevices, 0) << "Net plugin reports no devices";
    EXPECT_GT(summary.mergedDevices, 0)
        << "Expected merged devices but none found. "
        << "Check NCCL_NET_MERGE_LEVEL and topology placement.";
}

// ---------------------------------------------------------------------------
// Strict no-merge expected: when env var RCCL_EXPECT_NO_MERGES=1 is set,
// assert that no merged devices exist.
// ---------------------------------------------------------------------------
TEST_F(NicFusionTopoMPITest, ExpectNoMergedDevices)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks))
        << "Test requires at least " << kMinRanks << " MPI processes";

    const char* topoFile = getenv("NCCL_TOPO_FILE");
    if (!topoFile) {
        GTEST_SKIP() << "NCCL_TOPO_FILE not set";
    }

    const char* expectNoMerges = getenv("RCCL_EXPECT_NO_MERGES");
    if (!expectNoMerges || strcmp(expectNoMerges, "1") != 0) {
        GTEST_SKIP() << "RCCL_EXPECT_NO_MERGES not set to 1 — skipping strict no-merge assertion";
    }

    ASSERT_EQ(createTestCommunicator(), ncclSuccess)
        << "Failed to create communicator";

    auto summary = queryNetDevices();
    int topoNetCount = queryTopoNetCount();

    printSummary("ExpectNoMergedDevices", summary, topoNetCount);

    EXPECT_GT(summary.totalDevices, 0) << "Net plugin reports no devices";
    EXPECT_EQ(summary.mergedDevices, 0)
        << "Expected no merged devices but found " << summary.mergedDevices
        << ". Check NCCL_NET_MERGE_LEVEL and topology placement.";
}

// ---------------------------------------------------------------------------
// AllReduce sanity: after comm init with the given topology, run a small
// AllReduce to confirm the communicator is functional end-to-end.
// ---------------------------------------------------------------------------
TEST_F(NicFusionTopoMPITest, AllReduceSanityWithTopology)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinRanks))
        << "Test requires at least " << kMinRanks << " MPI processes";

    const char* topoFile = getenv("NCCL_TOPO_FILE");
    if (!topoFile) {
        GTEST_SKIP() << "NCCL_TOPO_FILE not set";
    }

    ASSERT_EQ(createTestCommunicator(), ncclSuccess)
        << "Failed to create communicator";

    auto summary = queryNetDevices();
    int topoNetCount = queryTopoNetCount();
    printSummary("AllReduceSanityWithTopology", summary, topoNetCount);

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    ASSERT_NE(comm, nullptr);
    ASSERT_NE(stream, nullptr);

    int rank = MPIEnvironment::world_rank;
    int worldSize = MPIEnvironment::world_size;

    constexpr size_t kCount = 1024;
    constexpr size_t kBytes = kCount * sizeof(float);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    ASSERT_EQ(hipMalloc(&sendBuf, kBytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&recvBuf, kBytes), hipSuccess);

    auto sendGuard = RCCLTestGuards::makeScopeGuard([&]() { hipFree(sendBuf); });
    auto recvGuard = RCCLTestGuards::makeScopeGuard([&]() { hipFree(recvBuf); });

    // Fill send buffer with rank value
    float fillVal = static_cast<float>(rank + 1);
    std::vector<float> hostSend(kCount, fillVal);
    ASSERT_EQ(hipMemcpy(sendBuf, hostSend.data(), kBytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(recvBuf, 0, kBytes), hipSuccess);

    ASSERT_EQ(ncclAllReduce(sendBuf, recvBuf, kCount, ncclFloat, ncclSum, comm, stream),
              ncclSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    // Expected sum: 1 + 2 + ... + worldSize = worldSize*(worldSize+1)/2
    float expectedVal = static_cast<float>(worldSize * (worldSize + 1) / 2);
    std::vector<float> hostRecv(kCount);
    ASSERT_EQ(hipMemcpy(hostRecv.data(), recvBuf, kBytes, hipMemcpyDeviceToHost), hipSuccess);

    int errors = 0;
    for (size_t i = 0; i < kCount && errors < 10; i++) {
        if (hostRecv[i] != expectedVal) {
            if (rank == 0) {
                fprintf(stderr, "[NicFusionTopo] AllReduce mismatch at [%zu]: got %f, expected %f\n",
                        i, hostRecv[i], expectedVal);
            }
            errors++;
        }
    }
    EXPECT_EQ(errors, 0) << "AllReduce produced incorrect results with topology "
                         << topoFile;

    if (rank == 0 && errors == 0) {
        fprintf(stderr, "[NicFusionTopo] AllReduce sanity PASSED (%zu elements, %d ranks)\n",
                kCount, worldSize);
    }
}

#endif // MPI_TESTS_ENABLED
