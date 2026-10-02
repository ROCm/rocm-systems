/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-side checks for the 8-GPU MI350P PCIe switch model
// (tools/topo_expl/models/topo_8p_950_pcie.xml).
//
// The same gfx950 SKU is also sold on an all-XGMI fabric (topo_8p_950.xml).
// These tests load the PCIe model with no GPU present and check that RCCL
// keeps every GPU pair on PCIe: no XGMI path, same-socket pairs stay inside
// one CPU, and cross-socket pairs cross the CPU link.
//
// Target: rccl-UnitTestsFixturesDebug.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

#include "graph.h"
#include "graph/topo.h"
#include "graph/xml.h"

#ifndef RCCL_TEST_SOURCE_DIR
#define RCCL_TEST_SOURCE_DIR "."
#endif

namespace RcclUnitTesting
{
namespace
{

constexpr int kGpuCount = 8;
constexpr int kGpusPerSocket = 4;
constexpr uint64_t kMi350pDevice = 0x75a8;
constexpr uint64_t kPex890Device = 0xc030;

std::string mi350pPcieTopoPath()
{
    return std::string(RCCL_TEST_SOURCE_DIR) +
           "/../tools/topo_expl/models/topo_8p_950_pcie.xml";
}

struct ncclXml* allocateXml(int maxNodes)
{
    size_t size = offsetof(struct ncclXml, nodes) + sizeof(struct ncclXmlNode) * maxNodes;
    struct ncclXml* xml = static_cast<struct ncclXml*>(malloc(size));
    if (xml)
    {
        memset(xml, 0, size);
        xml->maxNodes = maxNodes;
        xml->maxIndex = 0;
    }
    return xml;
}

uint64_t pciDeviceId(uint64_t packed)
{
    return (packed >> 32) & 0xffff;
}

struct ncclTopoSystem* loadMi350pPcieTopo()
{
    const std::string path = mi350pPcieTopoPath();
    if (access(path.c_str(), R_OK) != 0)
    {
        ADD_FAILURE() << "Topology model not found: " << path;
        return nullptr;
    }

    struct ncclXml* xml = allocateXml(NCCL_TOPO_XML_MAX_NODES);
    EXPECT_NE(xml, nullptr);
    if (xml == nullptr) return nullptr;

    ncclResult_t res = ncclTopoGetXmlFromFile(path.c_str(), xml, /*warn=*/0);
    EXPECT_EQ(res, ncclSuccess);
    struct ncclTopoSystem* system = nullptr;
    if (res == ncclSuccess)
    {
        // host_hash is omitted in the model, so the CPU hash is 0.
        res = ncclTopoGetSystemFromXml(xml, &system, /*localHostHash=*/0);
        EXPECT_EQ(res, ncclSuccess);
    }
    free(xml);
    if (res != ncclSuccess) return nullptr;
    EXPECT_EQ(ncclTopoComputePaths(system, /*comm=*/nullptr), ncclSuccess);
    return system;
}

} // namespace

TEST(Mi350pPcieTopo, EightGfx950GpusNoXgmi)
{
    struct ncclTopoSystem* system = loadMi350pPcieTopo();
    ASSERT_NE(system, nullptr);

    ASSERT_EQ(system->nodes[GPU].count, kGpuCount);
    ASSERT_EQ(system->nodes[CPU].count, 2);
    EXPECT_EQ(system->type & RCCL_TOPO_XGMI_ALL, 0);

    bool sawSwitch = false;
    for (int i = 0; i < system->nodes[PCI].count; i++)
    {
        if (pciDeviceId(system->nodes[PCI].nodes[i].pci.device) == kPex890Device) sawSwitch = true;
    }
    EXPECT_TRUE(sawSwitch) << "expected a PEX890 (0xc030) switch in the model";

    for (int i = 0; i < system->nodes[GPU].count; i++)
    {
        const struct ncclTopoNode* gpu = system->nodes[GPU].nodes + i;
        EXPECT_STREQ(gpu->gpu.gcn, "gfx950") << "GPU rank " << gpu->gpu.rank;
        EXPECT_GE(gpu->gpu.rank, 0);
        EXPECT_LT(gpu->gpu.rank, kGpuCount);
    }

    int mi350pDevs = 0;
    for (int i = 0; i < system->nodes[DEV].count; i++)
    {
        if (pciDeviceId(system->nodes[DEV].nodes[i].dev.device) == kMi350pDevice) mi350pDevs++;
    }
    EXPECT_EQ(mi350pDevs, kGpuCount);

    for (int i = 0; i < system->nodes[GPU].count; i++)
    {
        const struct ncclTopoNode* a = system->nodes[GPU].nodes + i;
        for (int j = 0; j < system->nodes[GPU].count; j++)
        {
            if (i == j) continue;
            const struct ncclTopoNode* b = system->nodes[GPU].nodes + j;
            ASSERT_NE(a->paths[GPU], nullptr);
            const int pathType = a->paths[GPU][j].type;
            EXPECT_NE(pathType, PATH_NVL) << "rank " << a->gpu.rank << " -> " << b->gpu.rank;

            bool xgmi = true;
            EXPECT_EQ(ncclTopoGetLinkType(system, a->gpu.dev, b->gpu.dev, &xgmi), ncclSuccess);
            EXPECT_FALSE(xgmi) << "rank " << a->gpu.rank << " -> " << b->gpu.rank;

            const bool sameSocket = (a->gpu.rank / kGpusPerSocket) == (b->gpu.rank / kGpusPerSocket);
            if (sameSocket)
            {
                EXPECT_LT(pathType, PATH_SYS) << "same-socket rank " << a->gpu.rank << " -> " << b->gpu.rank;
            }
            else
            {
                EXPECT_EQ(pathType, PATH_SYS) << "cross-socket rank " << a->gpu.rank << " -> " << b->gpu.rank;
            }
        }
    }

    ncclTopoFree(system);
}

} // namespace RcclUnitTesting
