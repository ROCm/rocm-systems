/*
 * Copyright (C) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include <sys/time.h>
#include <vector>
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "SDMAPacket.hpp"
#include "SDMAQueue.hpp"
#include "AqlQueue.hpp"
#include "KFDTestUtilQueue.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include "KFDBaseComponentTest.hpp"

class KFDPerformanceTest: public KFDBaseComponentTest {
 protected:
    virtual void SetUp();
    virtual void TearDown();

    /* Shared body of P2PBandWidthTest and P2PBandWidthBlitTest. The matrix of
     * node pairs and directions is the same either way; only the engine that
     * moves the bytes differs.
     */
    void P2PBandWidth(CopyEngine engine);

    void BlitCopySanityCheck(HSAuint32 node);
};

void KFDPerformanceTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDPerformanceTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

enum P2PDirection {
    IN = 1,
    OUT = 2,
    IN_OUT = 3,
    NONE = 4,
};

/*
 * Do the copy of one GPU from & to multiple GPUs.
 */
static void
testNodeToNodes(CopyEngine engine, HSAuint32 n1, const HSAuint32 *const n2Array, int n, P2PDirection n1Direction,
        P2PDirection n2Direction, HSAuint64 size, HSAuint64 *speed, HSAuint64 *speed2, std::stringstream *msg,
        bool isTestOverhead = false, HSAuint64 *time = 0) {
    HSAuint32 n2[n];
    void *n1Mem, *n2Mem[n];
    HsaMemFlags memFlags = {0};
    memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    memFlags.ui32.HostAccess = 0;
    memFlags.ui32.NonPaged = 1;
    GpuCopyParams array[n * 4];
    int array_count = 0;
    HSAuint64 alloc_size = ALIGN_UP(size, PAGE_SIZE);
    std::vector<GpuCopyParams> copyArray;
    int i;

    ASSERT_SUCCESS(HSAKMT_CALL(hsaKmtAllocMemory, g_baseTest->m_hsakmt_current_ctx, n1, alloc_size, memFlags, &n1Mem));
    ASSERT_SUCCESS(HSAKMT_CALL(hsaKmtMapMemoryToGPU, g_baseTest->m_hsakmt_current_ctx, n1Mem, alloc_size, NULL));

    for (i = 0; i < n; i++) {
        n2[i] = n2Array[i];
        ASSERT_SUCCESS(HSAKMT_CALL(hsaKmtAllocMemory, g_baseTest->m_hsakmt_current_ctx, n2[i], alloc_size, memFlags, &n2Mem[i]));
        ASSERT_SUCCESS(HSAKMT_CALL(hsaKmtMapMemoryToGPU, g_baseTest->m_hsakmt_current_ctx, n2Mem[i], alloc_size, NULL));
    }

    for (i = 0; i < n; i++) {
        if (n1Direction != NONE)
            ASSERT_NE(n1, 0);
        if (n2Direction != NONE)
            ASSERT_NE(n2[i], 0);

        do {
            if (n1Direction == IN || n1Direction == IN_OUT)
                /* n2Mem -> n1Mem*/
                array[array_count++] = {n1, n2Mem[i], n1Mem, size, n1/*group id, just a hint*/};
            if (n1Direction == OUT || n1Direction == IN_OUT)
                /* n1Mem -> n2Mem*/
                array[array_count++] = {n1, n1Mem, n2Mem[i], size, n1};
            /* Issue two copies to make full use of sdma.*/
        } while (n1Direction < IN_OUT && n == 1 && array_count % 2);
        /* Do nothing if no IN or OUT specified.*/

        do {
            if (n2Direction == IN || n2Direction == IN_OUT)
                /* n1Mem -> n2Mem*/
                array[array_count++] = {n2[i], n1Mem, n2Mem[i], size, n2[i]};
            if (n2Direction == OUT || n2Direction == IN_OUT)
                /* n2Mem -> n1Mem*/
                array[array_count++] = {n2[i], n2Mem[i], n1Mem, size, n2[i]};
        } while (n2Direction < IN_OUT && array_count % 2);
    }

    /* We measure a bunch of packets.*/
    if (isTestOverhead) {
            for (i = 0; i < 1000; i++)
                for (int j = 0; j < array_count; j++)
                    copyArray.push_back(array[j]);
        gpu_multicopy(copyArray, engine, 1, HEAD_TAIL);
        *time = CounterToNanoSec(copyArray[0].timeConsumption / (1000 * array_count));
    } else
        /* It did not respect the group id we set above.*/
        gpu_multicopy(array, array_count, engine, speed, speed2, msg);

    EXPECT_SUCCESS(HSAKMT_CALL(hsaKmtUnmapMemoryToGPU, g_baseTest->m_hsakmt_current_ctx, n1Mem));
    EXPECT_SUCCESS(HSAKMT_CALL(hsaKmtFreeMemory, g_baseTest->m_hsakmt_current_ctx, n1Mem, alloc_size));

    for (i = 0; i < n; i++) {
        EXPECT_SUCCESS(HSAKMT_CALL(hsaKmtUnmapMemoryToGPU, g_baseTest->m_hsakmt_current_ctx, n2Mem[i]));
        EXPECT_SUCCESS(HSAKMT_CALL(hsaKmtFreeMemory, g_baseTest->m_hsakmt_current_ctx, n2Mem[i], alloc_size));
    }
}

/* A blit copy is a shader, and a broken shader can report a spectacular
 * bandwidth for copying nothing at all. Prove the bytes really move before
 * trusting any of the numbers below.
 *
 * Uses the same size as the measurement so that both paths through the shader
 * get exercised: the uniform main loop, and the guarded remainder pass for the
 * chunks that do not divide evenly across the grid.
 */
void KFDPerformanceTest::BlitCopySanityCheck(HSAuint32 node) {
    const HSAuint64 size = 32ULL << 20;
    const HSAuint64 dwords = size / sizeof(HSAuint32);
    HsaMemoryBuffer srcBuf(size, node);
    HsaMemoryBuffer dstBuf(size, node);
    HSAuint32 *src = srcBuf.As<HSAuint32 *>();
    HSAuint32 *dst = dstBuf.As<HSAuint32 *>();
    std::vector<GpuCopyParams> copy;
    HSAuint64 i;

    /* Pattern varies per dword, so a copy that lands at the wrong offset is
     * caught, not just one that does not copy at all.
     */
    for (i = 0; i < dwords; i++) {
        src[i] = 0xa5000000 | (HSAuint32)i;
        dst[i] = 0;
    }

    copy.push_back({node, src, dst, size, node});
    gpu_multicopy(copy, COPY_BLIT, 0, NOTS);

    for (i = 0; i < dwords; i++)
        if (dst[i] != src[i])
            break;

    ASSERT_EQ(dwords, i) << "Blit copy is wrong at dword " << i << " of " << dwords
        << ": expected 0x" << std::hex << src[i] << ", got 0x" << dst[i] << std::dec
        << std::endl;
}

void KFDPerformanceTest::P2PBandWidth(CopyEngine engine) {
    if (!hsakmt_is_dgpu()) {
        LOG() << "Skipping test: Can't have 2 APUs on the same system." << std::endl;
        return;
    }

    const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
    std::vector<int> nodes;
    const bool isSpecified = g_TestDstNodeId != -1 && g_TestNodeId != -1;
    int numPeers = 0;
    /* How many copies can run concurrently on one node. A blit copy occupies a
     * compute queue instead of an SDMA engine, so the batching limit differs.
     * Never let this reach 0: it is used as a batch stride below, and a stride
     * of 0 would loop forever.
     */
    const unsigned int maxQueues = std::max(1u, engine == COPY_BLIT ?
                                   m_numCpQueues : m_numSdmaEngines * m_numSdmaQueuesPerEngine);

    if (isSpecified) {
        if (g_TestNodeId != g_TestDstNodeId) {
            nodes.push_back(g_TestNodeId);
            nodes.push_back(g_TestDstNodeId);
            if ((m_NodeInfo.IsPeerAccessibleByNode(g_TestNodeId, g_TestDstNodeId) &&
                 m_NodeInfo.IsPeerAccessibleByNode(g_TestDstNodeId, g_TestNodeId)))
                numPeers = 2;
        }
    } else {
        nodes = m_NodeInfo.GetNodesWithGPU();
        numPeers = nodes.size();
    }

    if (numPeers < 2) {
        LOG() << "Skipping test: Need at least two large bar GPU or XGMI connected." << std::endl;
        return;
    }

    if (engine == COPY_BLIT) {
        LOG() << "Copy engine: CU (blit)" << std::endl;
        BlitCopySanityCheck(nodes[0]);
        if (::testing::Test::HasFatalFailure())
            return;
    }

    g_TestTimeOut *= numPeers;

    std::vector<int> sysNodes(nodes); // include sysMem node 0...
    sysNodes.insert(sysNodes.begin(),0);

    const int total_tests = 7;
    const char *test_suits_string[total_tests] = {
        "Copy from node to node by [push, NONE]",
        "Copy from node to node by [pull, NONE]",
        "Full duplex copy from node to node by [push|pull, NONE]",
        "Full duplex copy from node to node by [push, push]",
        "Full duplex copy from node to node by [pull, pull]",
        "Copy from node to multiple nodes by [push, NONE]",
        "Copy from multiple nodes to node by [push, NONE]",
    };
    const P2PDirection test_suits[total_tests][2] = {
        /* One node used.*/
        {OUT,   NONE},
        {IN,    NONE},
        {IN_OUT,NONE},
        /* two nodes used.*/
        {OUT,   OUT},
        {IN,    IN},
        /* Multi nodes used*/
        {OUT,   NONE},
        {NONE,  OUT},
    };
    const int twoNodesIdx = 3;
    const int multiNodesIdx = 5;
    const HSAuint32 size = 32ULL << 20;
    int s = 0; //test index;
    std::stringstream msg;
    char str[64];

    if (isSpecified) {
        HSAuint32 n1 = g_TestNodeId;
        HSAuint32 n2 = g_TestDstNodeId;
        HSAuint64 speed, speed2;

        LOG() << "Copy from node to node by [push, pull]" << std::endl;
        snprintf(str, sizeof(str), "[%d -> %d] ", n1, n2);
        testNodeToNodes(engine, n1, &n2, 1, OUT, IN, size, &speed, &speed2, &msg);

        LOG() << std::dec << str << (float)speed / 1024 << " - " <<
                                 (float)speed2 / 1024 << " GB/s" << std::endl;
        goto exit;

    }

    for (; s < twoNodesIdx; s++) {
        LOG() << test_suits_string[s] << std::endl;
        msg << test_suits_string[s] << std::endl;

        for (unsigned i = 0; i < nodes.size(); i++) {
            /* Src node is a GPU.*/
            HSAuint32 n1 = nodes[i];
            HSAuint64 speed, speed2;

            /* Pick up dst node which can be sysMem.*/
            for (unsigned j = 0; j < sysNodes.size(); j++) {
                HSAuint32 n2 = sysNodes[j];
                if (n1 == n2)
                    continue;

                if (!m_NodeInfo.IsPeerAccessibleByNode(n2, n1))
                    continue;

                snprintf(str, sizeof(str), "[%d -> %d] ", n1, n2);
                msg << str << std::endl;
                testNodeToNodes(engine, n1, &n2, 1, test_suits[s][0], test_suits[s][1], size, &speed, &speed2, &msg);

                LOG() << std::dec << str << (float)speed / 1024 << " - " <<
                                            (float)speed2 / 1024 << " GB/s" << std::endl;
            }
        }
    }

    for (; s < multiNodesIdx; s++) {
        LOG() << test_suits_string[s] << std::endl;
        msg << test_suits_string[s] << std::endl;

        for (unsigned i = 0; i < nodes.size(); i++) {
            HSAuint32 n1 = nodes[i];
            HSAuint64 speed, speed2;

            for (unsigned j = i + 1; j < nodes.size(); j++) {
                HSAuint32 n2 = nodes[j];

                if (!m_NodeInfo.IsPeerAccessibleByNode(n2, n1) ||
                    !m_NodeInfo.IsPeerAccessibleByNode(n1, n2))
                    continue;

                snprintf(str, sizeof(str), "[%d <-> %d] ", n1, n2);
                msg << str << std::endl;
                testNodeToNodes(engine, n1, &n2, 1, test_suits[s][0], test_suits[s][1], size, &speed, &speed2, &msg);

                LOG() << std::dec << str << (float)speed / 1024 << " - " <<
                                            (float)speed2 / 1024 << " GB/s" << std::endl;
            }
        }
    }

    for (; s < total_tests && !isSpecified; s++) {
        LOG() << test_suits_string[s] << std::endl;
        msg << test_suits_string[s] << std::endl;
        /* Just use GPU nodes to do copy.*/
        std::vector<int> &src = test_suits[s][0] != NONE ? nodes : sysNodes;
        std::vector<int> &dst = test_suits[s][1] != NONE ? nodes : sysNodes;

        for (unsigned i = 0; i < src.size(); i++) {
            HSAuint32 n1 = src[i];
            HSAuint64 speed, speed2;
            HSAuint32 n2[dst.size()];
            int n = 0;
            char str[64];

            for (unsigned j = 0; j < dst.size(); j++) {
                if (dst[j] != n1) {
                    if (test_suits[s][0] != NONE &&
                        !m_NodeInfo.IsPeerAccessibleByNode(dst[j], n1))
                            continue;
                    if (test_suits[s][1] != NONE &&
                        !m_NodeInfo.IsPeerAccessibleByNode(n1, dst[j]))
                            continue;
                    n2[n++] = dst[j];
                }
            }

            /* At least 2 dst GPUs.*/
            if (n < 2)
                continue;

            if (test_suits[s][1] == OUT) {
                snprintf(str, sizeof(str), "[[%d...%d] -> %d] ", dst.front(), dst.back(), n1);
                msg << str << std::endl;
                testNodeToNodes(engine, n1, n2, n, test_suits[s][0], test_suits[s][1], size, &speed, &speed2, &msg);

                LOG() << std::dec << str << (float)speed / 1024 << " - " <<
                                        (float)speed2 / 1024 << " GB/s" << std::endl;
            } else {
                /* If the total number of peers is greater than the number of SDMA queues supported,
                 * then we test in the following way:
                 * 1. Test peers in batches where each batch consists of number of peers equal to the
                 *    max number of SDMA queues.
                 * 2. Keep repeating step 1 if number of peers left is greater than number of SDMA queues
                 *    supported.
                 * 3. Test the last batch with the remaining peers left which can be less than the number of
                 *    SDMA queues supported.
                 * For example, if there are 24 peers and max number of SDMA queues supported is 16, then
                 * the test will test 16 peers/nodes first and then remaining 8 in the next round.
                 */
                unsigned int j=0;
                unsigned int start_index;
                unsigned int end_index;
                do {
                    start_index = maxQueues * j++;
                    end_index = start_index + maxQueues - 1;

                    if (end_index + 1 > n)
                        end_index = n - 1;

                    snprintf(str, sizeof(str), "[%d -> [%d...%d]] ", n1, n2[start_index], n2[end_index]);
                    msg << str << std::endl;
                    testNodeToNodes(engine, n1, &n2[start_index], end_index - start_index + 1,
                                    test_suits[s][0], test_suits[s][1], size, &speed, &speed2, &msg);
                    LOG() << std::dec << str << (float)speed / 1024 << " - " <<
                                                (float)speed2 / 1024 << " GB/s" << std::endl;
                } while(end_index < (n - 1));
            }
        }
    }

    g_TestTimeOut /= numPeers;
exit:
    /* New line.*/
    LOG() << std::endl << msg.str() << std::endl;
}

/* Bandwidth over the sDMA engines. */
TEST_F(KFDPerformanceTest, P2PBandWidthTest) {
    TEST_START(TESTPROFILE_RUNALL);

    P2PBandWidth(COPY_SDMA);

    TEST_END
}

/* The same matrix driven by a copy kernel on the CUs, which is how ROCr moves
 * large buffers. SDMA and blit have different peak bandwidth and behave
 * differently over XGMI, so both numbers are needed to characterise a link -
 * and under partitioning they scale differently, since a partition changes how
 * many CUs and how many SDMA engines a logical node owns.
 */
TEST_F(KFDPerformanceTest, P2PBandWidthBlitTest) {
    TEST_START(TESTPROFILE_RUNALL);

    P2PBandWidth(COPY_BLIT);

    TEST_END
}

TEST_F(KFDPerformanceTest, P2POverheadTest) {
    TEST_START(TESTPROFILE_RUNALL);
    if (!hsakmt_is_dgpu()) {
        LOG() << "Skipping test: Can't have 2 APUs on the same system." << std::endl;
        return;
    }

    const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
    std::vector<int> nodes;

    nodes = m_NodeInfo.GetNodesWithGPU();
    int numPeers = nodes.size();

    if (numPeers < 2) {
        LOG() << "Skipping test: Need at least two large bar GPU or XGMI connected." << std::endl;
        return;
    }

    std::vector<int> sysNodes(nodes); // include sysMem node 0...
    sysNodes.insert(sysNodes.begin(),0);

    /* size should be small.*/
    const HSAuint32 sizeArray[] = {4, 8, 16, 64, 256, 1024};
    const int total_tests = 3;
    const char *test_suits_string[total_tests] = {
        "[push]     ",
        "[pull]     ",
        "[push|pull]",
    };
    const P2PDirection test_suits[total_tests] = {OUT, IN, IN_OUT};
    std::stringstream msg;
    int s; //test index;

    msg << "Test (avg. ns) | Size";
    for (auto &size : sizeArray)
        msg << "\t" << size;
    LOG() << msg.str() << std::endl;
    LOG() << "-----------------------------------------------------------------------" << std::endl;

    for (s = 0; s < total_tests; s++) {

        for (unsigned i = 0; i < nodes.size(); i++) {
            /* Src node is a GPU.*/
            HSAuint32 n1 = nodes[i];
            HSAuint64 time;

            /* Pick up dst node which can be sysMem.*/
            for (unsigned j = 0; j < sysNodes.size(); j++) {
                HSAuint32 n2 = sysNodes[j];
                std::stringstream msg;

                if (n1 != n2 && !m_NodeInfo.IsPeerAccessibleByNode(n2, n1))
                    continue;

                msg << test_suits_string[s] << "[" << n1 << " -> " << n2 << "]";
                for (auto &size : sizeArray) {
                    testNodeToNodes(COPY_SDMA, n1, &n2, 1, test_suits[s], NONE, size, 0, 0, 0, 1, &time);
                    msg << "\t" << time;
                }
                LOG() << msg.str() << std::endl;
            }
        }
    }

    TEST_END
}
