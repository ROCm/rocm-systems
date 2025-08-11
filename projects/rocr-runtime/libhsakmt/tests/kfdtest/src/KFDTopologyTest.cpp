/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "KFDTopologyTest.hpp"
#include <vector>
#include <string>

const HSAuint64 KFDTopologyTest::c_4Gigabyte = (1ull << 32) - 1;
const HSAuint64 KFDTopologyTest::c_40BitAddressSpace = (1ull << 40);

TEST_F(KFDTopologyTest , BasicTest) {
    TEST_START(TESTPROFILE_RUNALL)

    const HsaNodeProperties *pNodeProperties;

    // Goes over all nodes in the sytem properties and check the basic info received
    for (unsigned node = 0; node < m_SystemProperties.NumNodes; node++) {
        pNodeProperties = m_NodeInfo.GetNodeProperties(node);
        if (pNodeProperties != NULL) {
            HSAuint64 uniqueid;
            if (!pNodeProperties->UniqueID)
                uniqueid = 0;
            else
                uniqueid = pNodeProperties->UniqueID;
            LOG() << "UniqueID : " << std::dec << uniqueid <<
                     " Node index: " << node << std::endl;
            // Checking for cpu core only if it's a cpu only node or if its KAVERI apu.
            if (pNodeProperties->DeviceId == 0 || FamilyIdFromNode(pNodeProperties) == FAMILY_KV) {
                EXPECT_GT(pNodeProperties->NumCPUCores, HSAuint32(0)) << "Node index: " << node
                                                                      << " No CPUs core are connected for node index";
            }

            // If it's not a cpu only node, look for a gpu core
            if (pNodeProperties->DeviceId != 0) {
                EXPECT_GT(pNodeProperties->NumFComputeCores, HSAuint32(0)) << "Node index: " << node
                                                                           << "No GPUs core are connected.";
                // EngineId only applies to GPU, not CPU-only nodes
                EXPECT_GT(pNodeProperties->EngineId.ui32.uCode, 0) << "uCode version is 0";
                EXPECT_GE(pNodeProperties->EngineId.ui32.Major, 7) << "Major Version is less than 7";
                EXPECT_LT(pNodeProperties->EngineId.ui32.Minor, 10) << "Minor Version is greater than 9";
                EXPECT_GT(pNodeProperties->uCodeEngineVersions.uCodeSDMA, 0) << "sDMA firmware version is 0";

                LOG() << "VGPR Size is " << pNodeProperties->VGPRSizePerCU <<
                         "  SGPR Size is " << pNodeProperties->SGPRSizePerCU << std::endl;
            }
            EXPECT_GT(pNodeProperties->NumMemoryBanks, HSAuint32(0)) << "Node index: " << node << "No MemoryBanks.";
            if (pNodeProperties->NumCaches ==0)
                // SWDEV-420270
                // For "Intel Meteor lake Mobile", the cache info is not in sysfs,
                // That means /sys/devices/system/node/node%d/%s/cache is not exist.
                LOG() <<  "Node index: " << node << "  No Caches or not available to read ." << std::endl;
        }
    }

    TEST_END
}

// This test verifies failure status on hsaKmtGetNodeProperties with invalid params
TEST_F(KFDTopologyTest, GetNodePropertiesInvalidParams) {
    TEST_START(TESTPROFILE_RUNALL)

    EXPECT_EQ(HSAKMT_STATUS_INVALID_PARAMETER, hsaKmtGetNodeProperties(0, NULL));

    TEST_END
}

// This test verifies failure status on hsaKmtGetNodeProperties with invalid params
TEST_F(KFDTopologyTest, GetNodePropertiesInvalidNodeNum) {
    TEST_START(TESTPROFILE_RUNALL)

    HsaNodeProperties nodeProperties;
    memset(&nodeProperties, 0, sizeof(nodeProperties));
    EXPECT_EQ(HSAKMT_STATUS_INVALID_NODE_UNIT, hsaKmtGetNodeProperties(m_SystemProperties.NumNodes, &nodeProperties));

    TEST_END
}

// Test that we can get memory properties successfully per node
// TODO: Check validity of values returned
TEST_F(KFDTopologyTest, GetNodeMemoryProperties) {
    TEST_START(TESTPROFILE_RUNALL)
    const HsaNodeProperties *pNodeProperties;

    for (unsigned node = 0; node < m_SystemProperties.NumNodes; node++) {
        pNodeProperties = m_NodeInfo.GetNodeProperties(node);

        if (pNodeProperties != NULL) {
            HsaMemoryProperties *memoryProperties = new HsaMemoryProperties[pNodeProperties->NumMemoryBanks];
            EXPECT_SUCCESS(hsaKmtGetNodeMemoryProperties(node, pNodeProperties->NumMemoryBanks, memoryProperties));
            delete [] memoryProperties;
        }
    }

    TEST_END
}


// Test that the GPU local memory aperture is valid.
TEST_F(KFDTopologyTest, GpuvmApertureValidate) {
    TEST_REQUIRE_NO_ENV_CAPABILITIES(ENVCAPS_32BITLINUX);

    TEST_START(TESTPROFILE_RUNALL)
    const HsaNodeProperties *pNodeProperties;
    const std::vector<int> GpuNodes = m_NodeInfo.GetNodesWithGPU();

    for (unsigned i = 0; i < GpuNodes.size(); i++) {
        pNodeProperties = m_NodeInfo.GetNodeProperties(GpuNodes.at(i));
        if (pNodeProperties != NULL) {
            if (!hsakmt_is_dgpu() && !(FamilyIdFromNode(pNodeProperties) == FAMILY_KV)) {
                LOG() << "Skipping test: GPUVM framebuffer heap not exposed on APU except Kaveri." << std::endl;
                return;
            }
            HsaMemoryProperties *memoryProperties =  new HsaMemoryProperties[pNodeProperties->NumMemoryBanks];
            EXPECT_SUCCESS(hsaKmtGetNodeMemoryProperties(GpuNodes.at(i), pNodeProperties->NumMemoryBanks,
                                                         memoryProperties));
            bool GpuVMHeapFound = false;
            for (unsigned int bank = 0 ; bank  < pNodeProperties->NumMemoryBanks ; bank++) {
                // Check for either private (small-bar/APU) or public (large-bar)
                if ((HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE == memoryProperties[bank].HeapType) ||
                     (HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC == memoryProperties[bank].HeapType))
                    GpuVMHeapFound = true;
            }
            EXPECT_TRUE(GpuVMHeapFound);
            delete [] memoryProperties;
        }
    }

    TEST_END
}

// Test that we can get cache property successfully per node
// TODO: Check validity of values returned
TEST_F(KFDTopologyTest, GetNodeCacheProperties) {
    TEST_START(TESTPROFILE_RUNALL)

    const HsaNodeProperties *pNodeProperties;

    for (unsigned node = 0; node < m_SystemProperties.NumNodes; node++) {
        pNodeProperties = m_NodeInfo.GetNodeProperties(node);
        if (pNodeProperties != NULL) {
            HsaCacheProperties *cacheProperties = new HsaCacheProperties[pNodeProperties->NumCaches];
            EXPECT_SUCCESS(hsaKmtGetNodeCacheProperties(node, pNodeProperties->CComputeIdLo,
                           pNodeProperties->NumCaches, cacheProperties));
            if (pNodeProperties->NumCPUCores > 0) {  // this is a CPU node
                LOG() << "CPU Node " << std::dec << node << ": " << pNodeProperties->NumCaches << " caches"
                      << std::endl;
                for (unsigned n = 0; n < pNodeProperties->NumCaches; n++) {
                    LOG()<< n << " - Level " << cacheProperties[n].CacheLevel <<
                    " Type " << cacheProperties[n].CacheType.Value <<
                    " Size " << (cacheProperties[n].CacheSize >> 10) << "K " <<
                    " Associativity " << cacheProperties[n].CacheAssociativity <<
                    " LineSize " << cacheProperties[n].CacheLineSize <<
                    " LinesPerTag " << cacheProperties[n].CacheLinesPerTag << std::endl;
                    char string[1024] = "";
                    char sibling[5] = "";
                    for (unsigned i = 0; i < 256; i++) {
                        if (cacheProperties[n].SiblingMap[i]) {
                            sprintf(sibling, "%d,", i);
                            strcat(string, sibling);
                        }
                    }
                    LOG() << "     ProcIdLow " << cacheProperties[n].ProcessorIdLow <<
                    " SiblingMap " << string << std::endl;
                }
            } else {  // this is a GPU node
                LOG() << "GPU Node " << std::dec << node << ": " << pNodeProperties->NumCaches << " caches"
                      << std::endl;
                for (unsigned n = 0; n < pNodeProperties->NumCaches; n++) {
                    LOG()<< n << " - Level " << cacheProperties[n].CacheLevel <<
                    " Type " << cacheProperties[n].CacheType.Value <<
                    " Size " << cacheProperties[n].CacheSize << "K " <<
                    " Associativity " << cacheProperties[n].CacheAssociativity <<
                    " LineSize " << cacheProperties[n].CacheLineSize <<
                    " LinesPerTag " << cacheProperties[n].CacheLinesPerTag << std::endl;
                    char string[1024] = "";
                    char sibling[5] = "";
                    for (unsigned i = 0; i < 256; i++) {
                        if (cacheProperties[n].SiblingMap[i]) {
                            snprintf(sibling, 5, "%d,", i);
                            strcat(string, sibling);
                        }
                    }
                    LOG() << "     ProcIdLow " << cacheProperties[n].ProcessorIdLow <<
                    " SiblingMap " << string << std::endl;
                }
            }
            delete [] cacheProperties;
        }
    }

    TEST_END
}

// Test that we can get NodeIoLink property successfully per node
// TODO: Check validity of values returned
// GetNodeIoLinkProperties is disabled for now, test fails due to bug in BIOS
TEST_F(KFDTopologyTest, GetNodeIoLinkProperties) {
    TEST_START(TESTPROFILE_RUNALL)
    const HsaNodeProperties *pNodeProperties;
    int linkId;
    char c;

    LOG() << "Topology. [FromNode]--(Weight)-->[ToNode]" << std::endl;

    for (unsigned node = 0; node < m_SystemProperties.NumNodes; node++) {
        pNodeProperties = m_NodeInfo.GetNodeProperties(node);

        if (pNodeProperties != NULL) {
            HsaIoLinkProperties  *IolinkProperties =  new HsaIoLinkProperties[pNodeProperties->NumIOLinks];
            EXPECT_SUCCESS(hsaKmtGetNodeIoLinkProperties(node, pNodeProperties->NumIOLinks, IolinkProperties));
            if (pNodeProperties->NumIOLinks == 0) {
                // No io_links. Just print the node
                LOG() << "[" << node << "]" << std::endl;
                continue;
            }

            for (linkId = 0; linkId < pNodeProperties->NumIOLinks; linkId++) {
                if (linkId == 0) {
                    // First io_link. Print Parent Node and io_link Node
                    EXPECT_EQ(node, IolinkProperties[linkId].NodeFrom);
                    LOG() << "[" << IolinkProperties[linkId].NodeFrom << "]--(" <<
                        IolinkProperties[linkId].Weight << ")-->" <<
                        "[" << IolinkProperties[linkId].NodeTo << "]" << std::endl;
                    continue;
                }
                if (linkId == (pNodeProperties->NumIOLinks - 1))
                    c = '`';  // last node
                else
                    c = '|';
                LOG() << "  " << c << "--(" << IolinkProperties[linkId].Weight << ")-->" <<
                    "[" << IolinkProperties[linkId].NodeTo << "]" << std::endl;
            }
            LOG() << std::endl;
            delete [] IolinkProperties;
        }
    }

    TEST_END
}
