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

#include <sys/time.h>
#include <sys/mman.h>
#include <vector>
#include <utility>
#include <mutex>

#include "KFDQMTest.hpp"
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "SDMAPacket.hpp"
#include "XgmiOptimizedSDMAQueue.hpp"
#include "AqlQueue.hpp"
#include <algorithm>

#include "Dispatch.hpp"

extern unsigned int g_TestGPUsNum;

void KFDQMTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDQMTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

static void CreateDestroyCpQueue(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    PM4Queue queue;

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDQMTest, CreateDestroyCpQueue) {
    TEST_START(TESTPROFILE_RUNALL)

   ASSERT_SUCCESS(KFDTest_Launch(CreateDestroyCpQueue));

    TEST_END
}

static void SubmitNopCpQueue(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    PM4Queue queue;
    HsaEvent *event;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event), gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    queue.PlaceAndSubmitPacket(PM4NopPacket());

    queue.Wait4PacketConsumption(event);

    hsaKmtDestroyEvent(event);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

}

TEST_F(KFDQMTest, SubmitNopCpQueue) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(SubmitNopCpQueue));

    TEST_END
}

static void SubmitPacketCpQueue(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    HsaMemoryBuffer destBuf(PAGE_SIZE, gpuNode, false);

    destBuf.Fill(0xFF);
    HsaEvent *event;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event), gpuNode);

    PM4Queue queue;
    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    queue.PlaceAndSubmitPacket(PM4WriteDataPacket(destBuf.As<unsigned int*>(), 0, 0));

    queue.Wait4PacketConsumption(event);

    EXPECT_TRUE_GPU(WaitOnValue(destBuf.As<unsigned int*>(), 0), gpuNode);

    hsaKmtDestroyEvent(event);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDQMTest, SubmitPacketCpQueue) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(SubmitPacketCpQueue));

    TEST_END
}

static void AllCpQueues(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;
    int gpuIndex = pKFDQMTest->Get_NodeInfo()->HsaGPUindexFromGpuNode(gpuNode);
    HSAuint32 m_FamilyId = pKFDQMTest->GetFamilyIdFromNodeId(gpuNode);

    HsaMemoryBuffer destBuf(PAGE_SIZE, gpuNode, false);

    destBuf.Fill(0xFF);

    unsigned int  m_numCpQueues = pKFDQMTest->Get_NumCpQueues(gpuIndex);
    std::vector<PM4Queue> queues(m_numCpQueues);

    for (unsigned int qidx = 0; qidx < m_numCpQueues; ++qidx)
        ASSERT_SUCCESS_GPU(queues[qidx].Create(gpuNode), gpuNode) << " QueueId=" << qidx;

    for (unsigned int qidx = 0; qidx < m_numCpQueues; ++qidx) {
        queues[qidx].PlaceAndSubmitPacket(PM4WriteDataPacket(destBuf.As<unsigned int*>()+qidx*2, qidx, qidx));
        queues[qidx].PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, true, 0, 0));
        queues[qidx].Wait4PacketConsumption();

        EXPECT_TRUE_GPU(WaitOnValue(destBuf.As<unsigned int*>()+qidx*2, qidx), gpuNode);
    }

    for (unsigned int qidx = 0; qidx < m_numCpQueues; ++qidx)
       EXPECT_SUCCESS_GPU(queues[qidx].Destroy(), gpuNode);
}

TEST_F(KFDQMTest, AllCpQueues) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(AllCpQueues));

    TEST_END
}

static void CreateDestroySdmaQueue(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;

    SDMAQueue queue;

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

}

TEST_F(KFDQMTest, CreateDestroySdmaQueue) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(CreateDestroySdmaQueue));

    TEST_END
}

static void SubmitNopSdmaQueue(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;

    SDMAQueue queue;

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    queue.PlaceAndSubmitPacket(SDMANopPacket());

    queue.Wait4PacketConsumption();

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

}

TEST_F(KFDQMTest, SubmitNopSdmaQueue) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(SubmitNopSdmaQueue));

    TEST_END
}

static void SubmitPacketSdmaQueue(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;

    HsaMemoryBuffer destBuf(PAGE_SIZE, gpuNode, false);

    destBuf.Fill(0xFF);

    SDMAQueue queue;

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    queue.PlaceAndSubmitPacket(SDMAWriteDataPacket(queue.GetFamilyId(), destBuf.As<void *>(), 0x02020202));

    queue.Wait4PacketConsumption();

    EXPECT_TRUE_GPU(WaitOnValue(destBuf.As<unsigned int*>(), 0x02020202), gpuNode);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDQMTest, SubmitPacketSdmaQueue) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(SubmitPacketSdmaQueue));

    TEST_END
}

static void AllSdmaQueues(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;
    int gpuIndex = pKFDQMTest->Get_NodeInfo()->HsaGPUindexFromGpuNode(gpuNode);

    unsigned int m_numSdmaEngines = pKFDQMTest->Get_NumSdmaEngines(gpuIndex);
    unsigned int m_numSdmaQueuesPerEngine = pKFDQMTest->Get_NumSdmaSdmaQueuesPerEngine(gpuIndex);

    int bufSize = PAGE_SIZE;
    const unsigned int numSdmaQueues = m_numSdmaEngines * m_numSdmaQueuesPerEngine;

    LOG() << "Regular SDMA engines number: " << m_numSdmaEngines
          << " SDMA queues per engine: " << m_numSdmaQueuesPerEngine << std::endl;

    HsaMemoryBuffer destBuf(bufSize << 1 , gpuNode, false);
    HsaMemoryBuffer srcBuf(bufSize, gpuNode, false);
    destBuf.Fill(0xFF);

    std::vector<SDMAQueue> queues(numSdmaQueues);

    for (unsigned int qidx = 0; qidx < numSdmaQueues; ++qidx)
        ASSERT_SUCCESS_GPU(queues[qidx].Create(gpuNode), gpuNode);

    for (unsigned int qidx = 0; qidx < numSdmaQueues; ++qidx) {
        destBuf.Fill(0x0);
        srcBuf.Fill(qidx + 0xa0);
        queues[qidx].PlaceAndSubmitPacket(
            SDMACopyDataPacket(queues[qidx].GetFamilyId(), destBuf.As<unsigned int*>(), srcBuf.As<unsigned int*>(), bufSize));
        queues[qidx].PlaceAndSubmitPacket(
            SDMAWriteDataPacket(queues[qidx].GetFamilyId(), destBuf.As<unsigned int*>() + bufSize/4, 0x02020202));

        queues[qidx].Wait4PacketConsumption();

        EXPECT_TRUE_GPU(WaitOnValue(destBuf.As<unsigned int*>() + bufSize/4, 0x02020202), gpuNode);

        EXPECT_SUCCESS_GPU(memcmp(
            destBuf.As<unsigned int*>(), srcBuf.As<unsigned int*>(), bufSize), gpuNode);
    }

    for (unsigned int qidx = 0; qidx < numSdmaQueues; ++qidx)
        EXPECT_SUCCESS_GPU(queues[qidx].Destroy(), gpuNode);

}

TEST_F(KFDQMTest, AllSdmaQueues) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(AllSdmaQueues));

    TEST_END
}

static void AllXgmiSdmaQueues(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;
    int gpuIndex = pKFDQMTest->Get_NodeInfo()->HsaGPUindexFromGpuNode(gpuNode);
    unsigned int m_numSdmaXgmiEngines = pKFDQMTest->Get_NumSdmaSdmaXgmiEngines(gpuIndex);
    unsigned int m_numSdmaQueuesPerEngine = pKFDQMTest->Get_NumSdmaSdmaQueuesPerEngine(gpuIndex);

    int bufSize = PAGE_SIZE;
    int j;

    const unsigned int numXgmiSdmaQueues =
            m_numSdmaXgmiEngines * m_numSdmaQueuesPerEngine;

    LOG() << "XGMI SDMA engines number: " << m_numSdmaXgmiEngines
            << " SDMA queues per engine: " << m_numSdmaQueuesPerEngine << std::endl;

    HsaMemoryBuffer destBuf(bufSize << 1 , gpuNode, false);
    HsaMemoryBuffer srcBuf(bufSize, gpuNode, false);
    destBuf.Fill(0xFF);

    std::vector<XgmiOptimizedSDMAQueue> xgmiSdmaQueues(numXgmiSdmaQueues);

    for (j = 0; j < numXgmiSdmaQueues; ++j)
        ASSERT_SUCCESS_GPU(xgmiSdmaQueues[j].Create(gpuNode), gpuNode);

    for (j = 0; j < numXgmiSdmaQueues; ++j) {
        destBuf.Fill(0x0);
        srcBuf.Fill(j + 0xa0);
        xgmiSdmaQueues[j].PlaceAndSubmitPacket(
            SDMACopyDataPacket(xgmiSdmaQueues[j].GetFamilyId(),
                    destBuf.As<unsigned int*>(), srcBuf.As<unsigned int*>(), bufSize));
        xgmiSdmaQueues[j].PlaceAndSubmitPacket(
            SDMAWriteDataPacket(xgmiSdmaQueues[j].GetFamilyId(),
                    destBuf.As<unsigned int*>() + bufSize/4, 0x02020202));

        xgmiSdmaQueues[j].Wait4PacketConsumption();

        EXPECT_TRUE_GPU(WaitOnValue(destBuf.As<unsigned int*>() + bufSize/4, 0x02020202), gpuNode);

        EXPECT_SUCCESS_GPU(memcmp(
            destBuf.As<unsigned int*>(), srcBuf.As<unsigned int*>(), bufSize), gpuNode);
    }

    for (j = 0; j < numXgmiSdmaQueues; ++j)
        EXPECT_SUCCESS_GPU(xgmiSdmaQueues[j].Destroy(), gpuNode);

}

TEST_F(KFDQMTest, AllXgmiSdmaQueues) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(AllXgmiSdmaQueues));

    TEST_END
}

static void AllQueues(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    int gpuIndex = pKFDQMTest->Get_NodeInfo()->HsaGPUindexFromGpuNode(gpuNode);
    HSAuint32 m_FamilyId = pKFDQMTest->GetFamilyIdFromNodeId(gpuNode);

    unsigned int m_numSdmaXgmiEngines = pKFDQMTest->Get_NumSdmaSdmaXgmiEngines(gpuIndex);
    unsigned int m_numSdmaQueuesPerEngine = pKFDQMTest->Get_NumSdmaSdmaQueuesPerEngine(gpuIndex);
    unsigned int m_numSdmaEngines = pKFDQMTest->Get_NumSdmaEngines(gpuIndex);
    unsigned int m_numCpQueues = pKFDQMTest->Get_NumCpQueues(gpuIndex);

    int bufSize = PAGE_SIZE;
    unsigned int i, j;

    const unsigned int numCpQueues = m_numCpQueues;
    const unsigned int numSdmaQueues = m_numSdmaEngines * m_numSdmaQueuesPerEngine;
    const unsigned int numXgmiSdmaQueues =
            m_numSdmaXgmiEngines * m_numSdmaQueuesPerEngine;

    HsaMemoryBuffer destBufCp(PAGE_SIZE, gpuNode, false);
    destBufCp.Fill(0xFF);

    HsaMemoryBuffer destBuf(bufSize << 1 , gpuNode, false);
    HsaMemoryBuffer srcBuf(bufSize, gpuNode, false);
    destBuf.Fill(0xFF);

    std::vector<PM4Queue> cpQueues(numCpQueues);
    std::vector<SDMAQueue> sdmaQueues(numSdmaQueues);
    std::vector<XgmiOptimizedSDMAQueue> xgmiSdmaQueues(numXgmiSdmaQueues);

    for (i = 0; i < numCpQueues; ++i)
        ASSERT_SUCCESS_GPU(cpQueues[i].Create(gpuNode), gpuNode) << " QueueId=" << i;

    for (j = 0; j < numSdmaQueues; ++j)
        ASSERT_SUCCESS_GPU(sdmaQueues[j].Create(gpuNode), gpuNode);

    for (j = 0; j < numXgmiSdmaQueues; ++j)
        ASSERT_SUCCESS_GPU(xgmiSdmaQueues[j].Create(gpuNode), gpuNode);


    for (i = 0; i < numCpQueues; ++i) {
        cpQueues[i].PlaceAndSubmitPacket(PM4WriteDataPacket(destBufCp.As<unsigned int*>()+i*2, i, i));
        cpQueues[i].PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, true, 0, 0));

        cpQueues[i].Wait4PacketConsumption();

        EXPECT_TRUE_GPU(WaitOnValue(destBufCp.As<unsigned int*>()+i*2, i), gpuNode);
    }

    for (j = 0; j < numSdmaQueues; ++j) {
        destBuf.Fill(0x0);
        srcBuf.Fill(j + 0xa0);
        sdmaQueues[j].PlaceAndSubmitPacket(
            SDMACopyDataPacket(sdmaQueues[j].GetFamilyId(), destBuf.As<unsigned int*>(), srcBuf.As<unsigned int*>(), bufSize));
        sdmaQueues[j].PlaceAndSubmitPacket(
            SDMAWriteDataPacket(sdmaQueues[j].GetFamilyId(), destBuf.As<unsigned int*>() + bufSize/4, 0x02020202));

        sdmaQueues[j].Wait4PacketConsumption();

        EXPECT_TRUE_GPU(WaitOnValue(destBuf.As<unsigned int*>() + bufSize/4, 0x02020202), gpuNode);

        EXPECT_SUCCESS_GPU(memcmp(
            destBuf.As<unsigned int*>(), srcBuf.As<unsigned int*>(), bufSize), gpuNode);
    }

    for (j = 0; j < numXgmiSdmaQueues; ++j) {
        destBuf.Fill(0x0);
        srcBuf.Fill(j + 0xa0);
        xgmiSdmaQueues[j].PlaceAndSubmitPacket(
            SDMACopyDataPacket(xgmiSdmaQueues[j].GetFamilyId(),
                    destBuf.As<unsigned int*>(), srcBuf.As<unsigned int*>(), bufSize));
        xgmiSdmaQueues[j].PlaceAndSubmitPacket(
            SDMAWriteDataPacket(xgmiSdmaQueues[j].GetFamilyId(),
                    destBuf.As<unsigned int*>() + bufSize/4, 0x02020202));

        xgmiSdmaQueues[j].Wait4PacketConsumption();

        EXPECT_TRUE_GPU(WaitOnValue(destBuf.As<unsigned int*>() + bufSize/4, 0x02020202), gpuNode);

        EXPECT_SUCCESS_GPU(memcmp(
            destBuf.As<unsigned int*>(), srcBuf.As<unsigned int*>(), bufSize), gpuNode);
    }


    for (i = 0; i < numCpQueues; ++i)
       EXPECT_SUCCESS_GPU(cpQueues[i].Destroy(), gpuNode);

    for (j = 0; j < numSdmaQueues; ++j)
        EXPECT_SUCCESS_GPU(sdmaQueues[j].Destroy(), gpuNode);

    for (j = 0; j < numXgmiSdmaQueues; ++j)
        EXPECT_SUCCESS_GPU(xgmiSdmaQueues[j].Destroy(), gpuNode);

}

TEST_F(KFDQMTest, AllQueues) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(AllQueues));

    TEST_END
}

/* The following test is designed to reproduce an intermittent hang on
 * Fiji and other VI/Polaris GPUs. This test typically hangs in a few
 * seconds. According to analysis done by HW engineers, the culprit
 * seems to be PCIe speed switching. The problem can be worked around
 * by disabling the lowest DPM level on Fiji.
 */
static void SdmaConcurrentCopies(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    int gpuIndex = pKFDQMTest->Get_NodeInfo()->HsaGPUindexFromGpuNode(gpuNode);
    HSAuint32 m_FamilyId = pKFDQMTest->GetFamilyIdFromNodeId(gpuNode);

#define BUFFER_SIZE (64*1024)
#define NPACKETS 1
#define COPY_SIZE (BUFFER_SIZE / NPACKETS)
    HsaMemoryBuffer srcBuf(BUFFER_SIZE, 0, true);
    HsaMemoryBuffer dstBuf(BUFFER_SIZE, gpuNode, false, hsakmt_is_dgpu() ? true : false);

    SDMAQueue queue;

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    std::ostream &log = LOG();
    char progress[] = "-\b";
    log << "Running ... ";

    for (unsigned i = 0; i < 100000; i++) {
        if (i % 1000 == 0) {
            const char progressSteps[4] = {'-', '\\', '|', '/'};
            progress[0] = progressSteps[(i/1000) % 4];
            log << progress;
        }

        for (unsigned j = 0; j < NPACKETS; j++)
            queue.PlacePacket(
                SDMACopyDataPacket(queue.GetFamilyId(), dstBuf.As<char *>()+COPY_SIZE*j,
                                   srcBuf.As<char *>()+COPY_SIZE*j, COPY_SIZE));
        queue.SubmitPacket();

        /* Waste a variable amount of time. Submission timing
         * while SDMA runs concurrently seems to be critical for
         * reproducing the hang
         */
        for (int k = 0; k < (i & 0xfff); k++)
            memcpy(srcBuf.As<char *>()+PAGE_SIZE, srcBuf.As<char *>(), 1024);

        /* Wait for idle every 8 packets to allow the SDMA engine to
         * run concurrently for a bit without getting too far ahead
         */
        if ((i & 0x7) == 0)
            queue.Wait4PacketConsumption();
    }
    log << "Done." << std::endl;

    queue.PlaceAndSubmitPacket(SDMAWriteDataPacket(queue.GetFamilyId(), srcBuf.As<unsigned *>(), 0x02020202));
    queue.Wait4PacketConsumption();
    EXPECT_TRUE_GPU(WaitOnValue(srcBuf.As<unsigned int*>(), 0x02020202), gpuNode);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDQMTest, SdmaConcurrentCopies) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(SdmaConcurrentCopies));

    TEST_END
}

static void DisableCpQueueByUpdateWithNullAddress(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    HsaMemoryBuffer destBuf(PAGE_SIZE, gpuNode, false);

    destBuf.Fill(0xFFFFFFFF);

    PM4Queue queue;

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    HsaEvent *event;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event), gpuNode);

    queue.PlaceAndSubmitPacket(PM4WriteDataPacket(destBuf.As<unsigned int*>(), 0, 0));

    queue.Wait4PacketConsumption(event);

    WaitOnValue(destBuf.As<unsigned int*>(), 0);

    destBuf.Fill(0xFFFFFFFF);

    EXPECT_SUCCESS_GPU(queue.Update(BaseQueue::DEFAULT_QUEUE_PERCENTAGE, BaseQueue::DEFAULT_PRIORITY, true), gpuNode);

    queue.PlaceAndSubmitPacket(PM4WriteDataPacket(destBuf.As<unsigned int*>(), 1, 1));

    // Don't sync since we don't expect rptr to change when the queue is disabled.
    Delay(2000);

    EXPECT_EQ_GPU(destBuf.As<unsigned int*>()[0], 0xFFFFFFFF, gpuNode)
        << "Packet executed even though the queue is supposed to be disabled!";

    EXPECT_SUCCESS_GPU(queue.Update(BaseQueue::DEFAULT_QUEUE_PERCENTAGE, BaseQueue::DEFAULT_PRIORITY, false), gpuNode);

    queue.Wait4PacketConsumption(event);

    WaitOnValue(destBuf.As<unsigned int*>(), 1);

    hsaKmtDestroyEvent(event);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

}

TEST_F(KFDQMTest, DisableCpQueueByUpdateWithNullAddress) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(DisableCpQueueByUpdateWithNullAddress));

    TEST_END
}

static void DisableSdmaQueueByUpdateWithNullAddress(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;

    HsaMemoryBuffer destBuf(PAGE_SIZE, gpuNode, false);

    destBuf.Fill(0xFFFFFFFF);

    SDMAQueue queue;

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    queue.PlaceAndSubmitPacket(SDMAWriteDataPacket(queue.GetFamilyId(), destBuf.As<void*>(), 0));

    WaitOnValue(destBuf.As<unsigned int*>(), 0);

    destBuf.Fill(0xFFFFFFFF);

    EXPECT_SUCCESS_GPU(queue.Update(BaseQueue::DEFAULT_QUEUE_PERCENTAGE, BaseQueue::DEFAULT_PRIORITY, true), gpuNode);

    queue.PlaceAndSubmitPacket(SDMAWriteDataPacket(queue.GetFamilyId(), destBuf.As<void*>(), 0));

    // Don't sync since we don't expect rptr to change when the queue is disabled.
    Delay(2000);

    EXPECT_EQ_GPU(destBuf.As<unsigned int*>()[0], 0xFFFFFFFF, gpuNode)
        << "Packet executed even though the queue is supposed to be disabled!";

    EXPECT_SUCCESS_GPU(queue.Update(BaseQueue::DEFAULT_QUEUE_PERCENTAGE, BaseQueue::DEFAULT_PRIORITY, false), gpuNode);

    queue.Wait4PacketConsumption();

    WaitOnValue(destBuf.As<unsigned int*>(), 0);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);


}
TEST_F(KFDQMTest, DisableSdmaQueueByUpdateWithNullAddress) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(DisableSdmaQueueByUpdateWithNullAddress));

    TEST_END
}

static void DisableCpQueueByUpdateWithZeroPercentage(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;

    HsaMemoryBuffer destBuf(PAGE_SIZE, gpuNode, false);

    destBuf.Fill(0xFFFFFFFF);

    PM4Queue queue;

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    HsaEvent *event;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event), gpuNode);

    PM4WriteDataPacket packet1, packet2;
    packet1.InitPacket(destBuf.As<unsigned int*>(), 0, 0);
    packet2.InitPacket(destBuf.As<unsigned int*>(), 1, 1);

    queue.PlaceAndSubmitPacket(packet1);

    queue.Wait4PacketConsumption(event);

    WaitOnValue(destBuf.As<unsigned int*>(), 0);

    destBuf.Fill(0xFFFFFFFF);

    EXPECT_SUCCESS_GPU(queue.Update(0/*percentage*/, BaseQueue::DEFAULT_PRIORITY, false), gpuNode);

    queue.PlaceAndSubmitPacket(packet2);

    // Don't sync since we don't expect rptr to change when the queue is disabled.
    Delay(2000);

    EXPECT_EQ_GPU(destBuf.As<unsigned int*>()[0], 0xFFFFFFFF, gpuNode)
        << "Packet executed even though the queue is supposed to be disabled!";

    EXPECT_SUCCESS_GPU(queue.Update(BaseQueue::DEFAULT_QUEUE_PERCENTAGE, BaseQueue::DEFAULT_PRIORITY, false), gpuNode);

    queue.Wait4PacketConsumption(event);

    WaitOnValue(destBuf.As<unsigned int*>(), 1);
    hsaKmtDestroyEvent(event);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

}

TEST_F(KFDQMTest, DisableCpQueueByUpdateWithZeroPercentage) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(DisableCpQueueByUpdateWithZeroPercentage));

    TEST_END
}

static void CreateQueueStressSingleThreaded(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;

    static const HSAuint64 TEST_TIME_SEC = 15;

    HSAuint64 initialTime = GetSystemTickCountInMicroSec();

    unsigned int numIter = 0;

    HSAuint64 timePassed = 0;

    do {
        // The following means we'll get the order 0,0 => 0,1 => 1,0 => 1,1 so we cover all options.
        unsigned int firstToCreate = (numIter % 2 != 0) ? 1 : 0;
        unsigned int firstToDestroy = (numIter % 4 > 1) ? 1 : 0;

        unsigned int secondToCreate = (firstToCreate + 1)%2;
        unsigned int secondToDestroy = (firstToDestroy + 1)%2;

        BaseQueue *queues[2] = {new PM4Queue(), new SDMAQueue()};

        ASSERT_SUCCESS_GPU(queues[firstToCreate]->Create(gpuNode), gpuNode);
        ASSERT_SUCCESS_GPU(queues[secondToCreate]->Create(gpuNode),gpuNode);

        EXPECT_SUCCESS_GPU(queues[firstToDestroy]->Destroy(), gpuNode);
        EXPECT_SUCCESS_GPU(queues[secondToDestroy]->Destroy(), gpuNode);

        delete queues[0];
        delete queues[1];
        ++numIter;

        HSAuint64 curTime = GetSystemTickCountInMicroSec();
        timePassed = (curTime - initialTime) / 1000000;
    } while (timePassed < TEST_TIME_SEC);

}

TEST_F(KFDQMTest, CreateQueueStressSingleThreaded) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(CreateQueueStressSingleThreaded));

    TEST_END
}

static void OverSubscribeCpQueues(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;
    const HSAuint32 m_FamilyId = pKFDQMTest->GetFamilyIdFromNodeId(gpuNode);

    if (m_FamilyId == FAMILY_CI || m_FamilyId == FAMILY_KV) {
        LOG() << "Skipping test: CI doesn't have HW scheduling." << std::endl;
        return;
    }

    /* The max queues per process is 1024 limited by
     * KFD, so MAX_CP_QUEUES is needed to adapt it
     * when total queues exceed it.
     */
    static const unsigned int MAX_CP_QUEUES = g_TestGPUsNum > 15 ?
                                              1024 / g_TestGPUsNum :
                                              65;
    static const unsigned int MAX_PACKETS = 100;

    HsaMemoryBuffer destBuf(PAGE_SIZE, gpuNode, false);

    destBuf.Fill(0xFF);

    PM4Queue queues[MAX_CP_QUEUES];

    for (unsigned int qidx = 0; qidx < MAX_CP_QUEUES; ++qidx)
        ASSERT_SUCCESS_GPU(queues[qidx].Create(gpuNode), gpuNode) << " QueueId=" << qidx;

    for (unsigned int qidx = 0; qidx < MAX_CP_QUEUES; ++qidx) {
        unsigned int pktSizeDw = 0;
        for (unsigned int i = 0; i < MAX_PACKETS; i++) {
            PM4WriteDataPacket packet;
            packet.InitPacket(destBuf.As<unsigned int*>()+qidx*2, qidx+i, qidx+i);  // two dwords per packet
            queues[qidx].PlacePacket(packet);
        }
    }

    for (unsigned int qidx = 0; qidx < MAX_CP_QUEUES; ++qidx)
        queues[qidx].SubmitPacket();

    // Delaying for 5 seconds in order to get all the results
    Delay(5000);

    for (unsigned int qidx = 0; qidx < MAX_CP_QUEUES; ++qidx)
        EXPECT_TRUE_GPU(queues[qidx].AllPacketsSubmitted(), gpuNode)<< "QueueId=" << qidx;;

    for (unsigned int qidx = 0; qidx < MAX_CP_QUEUES; ++qidx)
        EXPECT_SUCCESS_GPU(queues[qidx].Destroy(), gpuNode);

}

TEST_F(KFDQMTest, OverSubscribeCpQueues) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(OverSubscribeCpQueues));

    TEST_END
}

HSAint64 KFDQMTest::TimeConsumedwithCUMask(int node, uint32_t* mask, uint32_t mask_count) {
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, node, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer dstBuffer(PAGE_SIZE, node, true, false, false);
    HsaMemoryBuffer ctlBuffer(PAGE_SIZE, node, true, false, false);

    EXPECT_SUCCESS(m_pAsm->RunAssembleBuf(LoopIsa, isaBuffer.As<char*>()));

    Dispatch dispatch(isaBuffer);
    dispatch.SetDim(1024, 16, 16);

    PM4Queue queue;
    EXPECT_SUCCESS(queue.Create(node));
    EXPECT_SUCCESS(queue.SetCUMask(mask, mask_count));
    queue.SetSkipWaitConsump(true);

    HSAuint64 startTime = GetSystemTickCountInMicroSec();
    dispatch.Submit(queue);
    dispatch.Sync();
    HSAuint64 endTime = GetSystemTickCountInMicroSec();

    EXPECT_SUCCESS(queue.Destroy());
    return endTime - startTime;
}

/* To cover for outliers, allow us to get the Average time based on a specified number of iterations */
HSAint64 KFDQMTest::GetAverageTimeConsumedwithCUMask(int node, uint32_t* mask, uint32_t mask_count, int iterations) {
    HSAint64 timeArray[iterations];
    HSAint64 timeTotal = 0;
    if (iterations < 1) {
        LOG() << "ERROR: At least 1 iteration must be performed" << std::endl;
        return 0;
    }

    for (int x = 0; x < iterations; x++) {
        timeArray[x] = TimeConsumedwithCUMask(node, mask, mask_count);
        timeTotal += timeArray[x];
    }

    if (timeTotal == 0) {
        LOG() << "ERROR: Total time reported as 0. Exiting" << std::endl;
        return 0;
    }

    for (int x = 0; x < iterations; x++) {
        HSAint64 variance = timeArray[x] / (timeTotal / iterations);
        if (variance < CuNegVariance || variance > CuPosVariance)
            LOG() << "WARNING: Measurement #" << x << "/" << iterations << " (" << timeArray[x]
                  << ") is at least " << CuVariance*100 << "% away from the mean (" << timeTotal/iterations << ")"
                  << std::endl;
    }

    return timeTotal / iterations;
}

/*
 * Apply CU masking in a linear fashion, adding 1 CU per iteration
 * until all Shader Engines are full
 */
void BasicCuMaskingLinear(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;
    const HSAuint32 m_FamilyId = pKFDQMTest->GetFamilyIdFromNodeId(gpuNode);

    if (m_FamilyId >= FAMILY_VI) {
        const HsaNodeProperties *pNodeProperties = pKFDQMTest->Get_NodeInfo()->GetNodeProperties(gpuNode);
        uint32_t ActiveCU = (pNodeProperties->NumFComputeCores / pNodeProperties->NumSIMDPerCU);
        uint32_t numSEs = pNodeProperties->NumShaderBanks;
        LOG() << std::dec << "# Compute cores: " << pNodeProperties->NumFComputeCores << std::endl;
        LOG() << std::dec << "# SIMDs per CU: " << pNodeProperties->NumSIMDPerCU << std::endl;
        LOG() << std::dec << "# Shader engines: " << numSEs << std::endl;
        LOG() << std::dec << "# Active CUs: " << ActiveCU << std::endl;
        HSAint64 TimewithCU1, TimewithCU;
        uint32_t maskNumDwords = (ActiveCU + 31) / 32; /* Round up to the nearest multiple of 32 */
        uint32_t maskNumBits = maskNumDwords * 32;
        uint32_t mask[maskNumDwords];
        double ratio;

        mask[0] = 0x1;
        for (int i = 1; i < maskNumDwords; i++)
            mask[i] = 0x0;

        /* Execute once to get any HW optimizations out of the way */
        pKFDQMTest->TimeConsumedwithCUMask(gpuNode, mask, maskNumBits);

        LOG() << "Getting baseline performance numbers (CU Mask: 0x1)" << std::endl;
        TimewithCU1 = pKFDQMTest->GetAverageTimeConsumedwithCUMask(gpuNode, mask, maskNumBits, 3);

        for (int nCUs = 2; nCUs <= ActiveCU; nCUs++) {
            int maskIndex = (nCUs - 1) / 32;
            mask[maskIndex] |= 1 << ((nCUs - 1) % 32);

            TimewithCU = pKFDQMTest->TimeConsumedwithCUMask(gpuNode, mask, maskNumBits);
            ratio = (double)(TimewithCU1) / ((double)(TimewithCU) * nCUs);

            LOG() << "Expected performance of " << nCUs << " CUs vs 1 CU:" << std::endl;
            LOG() << std::setprecision(2) << pKFDQMTest->CuNegVariance << " <= " << std::fixed << std::setprecision(8)
                  << ratio << " <= " << std::setprecision(2) << pKFDQMTest->CuPosVariance << std::endl;

            EXPECT_TRUE((ratio >= pKFDQMTest->CuNegVariance) && (ratio <= pKFDQMTest->CuPosVariance));

            RECORD(ratio) << "Ratio-" << nCUs << "-CUs";
        }
    } else {
        LOG() << "Skipping test: Test not supported for family ID 0x" << m_FamilyId << "." << std::endl;
    }
}

TEST_F(KFDQMTest, BasicCuMaskingLinear) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(BasicCuMaskingLinear));

    TEST_END
}


// ====== ExtendedCuMasking Helper Functions ====== //


#define CUMASK_DEBUG 0   // Enable extra output for debugging issues

#if CUMASK_DEBUG
#define DBG_PRINT printf
#else
#define DBG_PRINT
#endif


/*
 * Helper function to print multi-dword mask.
 *
 *   pHeader: A non-NULL pointer to a string to use as the header.
 *     pMask: A pointer to the mask to print out.
 * numDwords: Number of elements in mask array.
 *
 */
static void printMask(const char *pHeader, uint32_t *pMask, uint32_t numDwords) {
    printf("%s0x", pHeader);
    for (int i = numDwords - 1; i >= 0; i--) {
        printf("%08x", pMask[i]);
    }
    printf("\n");
}


/*
 * Set the CU mask for each specified WGPs.
 *
 * Note: The effect is cumulative, function can be called multiple times to
 *       set up additional WGPs in the provided pMask.
 *
 * pMask:      A non-NULL pointer to the CU mask.
 * maskConfig: Information on GPU configuration.
 * seMask:     Specifies SEs that are targetted.
 * saMask:     Specifies SAs that are targetted within the SEs specified.
 * wgpMask:    Specifies WGPs that are targetted within the (SE,SA) specified.
 *
 * For seMask, saMask, and wgpMask:
 *   One bit per SE/SA/WGP, multiple bits can be specified.
 *   Masks cannot be 0 (at least 1 SE, 1 SA and 1 WGP must be specified).
 *   Special value: -1 (specifies ALL)
 *
 */
static bool setCUMask(uint32_t *pMask, mask_config_t maskConfig, uint32_t seMask, uint32_t saMask, uint32_t wgpMask) {

    bool result = true;

    if (pMask) {
        if (seMask && saMask && wgpMask) {   // proceed only with non-zero mask
            for (int i = 0; i < maskConfig.numWGPperSA; i++) {
                if (((wgpMask >> i) & 1)) {
                    for (int j = 0; j < maskConfig.numSAperSE; j++) {
                        if (((saMask >> j) & 1)) {
                            for (int k = 0; k < maskConfig.numSEs; k++) {
                                if (((seMask >> k) & 1)) {
                                    uint32_t insLoc = k * 2 + j * (2 * maskConfig.numSEs) + i * (2 * maskConfig.numSEs * maskConfig.numSAperSE);
                                    pMask[insLoc / 32] |= (0x3 << (insLoc % 32));
                                }
                            }
                        }
                    }
                }
            }
        } else {
            LOG() << "ERROR: SE/SA/WGP mask values must be non-zero!\n";
            result = false;
        }
    } else {
        LOG() << "ERROR: pMask is NULL!\n";
        result = false;
    }

    return result;
}


/*
 * Compute an adjusted CU mask to use when some WGPs are inactive.
 *
 * The adjusted mask takes into account the inactive WGPs by removing their corresponding
 * bits from the mask as these are skipped by KFD.   As bits are removed from the mask,
 * the remaining bit values are shifted right.
 *
 *   pAdjMask: A non-NULL pointer where the adjusted mask will be written.
 *      pMask: A non-NULL pointer to the CU mask.
 * maskConfig: Information on GPU configuration.
 *
 * Returns:
 *      true: If adjusted mask has one or more non-zero bit set.
 *     false: If the adjusted mask is all zeroes (no WGPs left to do work).
 *
 * When false is returned, we should skipped the specific test scenario.
 *
 */
bool adjustMask(uint32_t *pAdjMask, uint32_t *pMask, mask_config_t maskConfig) {
    int wi = 0;
    int totalBits = maskConfig.numBits;
    bool nonZero = false;

    memset(pAdjMask, 0, sizeof(uint32_t) * maskConfig.numDwords);

    for (int ri = 0; ri < totalBits; ri += 2) {

        uint32_t value = (pMask[ri / 32] >> (ri % 32)) & 0x3;

        if ((maskConfig.pInactiveMask[ri / 32] & (0x3 << (ri % 32))) != 0)
        {
            // skip that entry
        }
        else
        {
            uint32_t newValue = value << (wi % 32);
            pAdjMask[wi / 32] |= newValue;
            wi += 2;

            if (value != 0) {
                nonZero = true;
            }
        }
    }

#if CUMASK_DEBUG
    printf("\nAdjusting mask:\n");
    printMask("    mask: ", pMask, maskConfig.numDwords);
    printMask("inactive: ", maskConfig.pInactiveMask, maskConfig.numDwords);
    printMask("adjusted: ", pAdjMask, maskConfig.numDwords);
    printf("\n");
#endif //CUMASK_DEBUG

    return nonZero;
}



/*
 * Validates the result of a test.
 *
 * pMask:        A non-NULL pointer to the CU mask that was used for the test.
 * maskConfig:   Information on GPU configuration.
 * numWorkItems: Number of work items used for shader execution.
 * pOutput:      Pointer to the output array.
 * pResultMask:  If non-NULL, result mask constructed from output is stored at that memory location.
 *
 */
static bool validateTest(uint32_t *pMask, mask_config_t maskConfig, uint32_t numWorkItems, out_data_t *pOutput, uint32_t *pResultMask)
{
    uint32_t resultMask[maskConfig.numDwords];
    bool result = false;

    memset(resultMask, 0, sizeof(resultMask));

    for (int i = 0; i < numWorkItems; i++) {
        DBG_PRINT("=== % 4d: 0x%08x [ se: %2d, sa: %2d, wgp: %2d]\n", i, pOutput[i].data, pOutput[i].se, pOutput[i].sa, pOutput[i].wgp);

        setCUMask(resultMask, maskConfig,
                  1 << pOutput[i].se,
                  1 << pOutput[i].sa,
                  1 << pOutput[i].wgp);
    }

    if (pResultMask) {
        memcpy(pResultMask, resultMask, sizeof(resultMask));
    }

    if (maskConfig.pInactiveMask) {
        // If some WGPs were inactive, compute a verify mask taking into account the inactive WGPs.
        uint32_t verifyMask[maskConfig.numDwords];
        memset(verifyMask, 0, sizeof(verifyMask));

        for (int i = 0; i < maskConfig.numDwords; i++) {
            verifyMask[i] = pMask[i] & ~maskConfig.pInactiveMask[i];
        }

#if CUMASK_DEBUG
        printf("\nValidate test:\n");
        printMask("        mask: ", pMask, maskConfig.numDwords);
        printMask("  resultMask: ", resultMask, maskConfig.numDwords);
        printMask("inactiveMask: ", maskConfig.pInactiveMask, maskConfig.numDwords);
        printMask("  verifyMask: ", verifyMask, maskConfig.numDwords);
#endif //CUMASK_DEBUG

        result = (memcmp(verifyMask, resultMask, sizeof(resultMask)) == 0);
    } else {

#if CUMASK_DEBUG
        printf("\nValidate test:\n");
        printMask("        mask: ", pMask, maskConfig.numDwords);
        printMask("  resultMask: ", resultMask, maskConfig.numDwords);
#endif //CUMASK_DEBUG

        result = (memcmp(pMask, resultMask, sizeof(resultMask)) == 0);
    }

    DBG_PRINT("      Result: %s\n\n", result ? "PASS" : "FAIL");

    return result;
}

/*
 * Set CU Mask, submit the testing shader, and validate the results.
 *
 * gpuNode:       The node to use for the test.
 * pMask:         A non-NULL pointer to the CU mask to use for the test.
 * maskConfig:    Information on GPU configuration.
 * programBuffer: The buffer that contains the shader program.
 * numWorkItems:  The number of work items to use.
 * pOutput:       A non-NULL pointer to the output buffer used by the shader.
 * pResultMask:   If non-NULL, result mask constructed from output is stored at that memory location.
 *
 */
static bool testCUMask(int gpuNode, uint32_t *pMask, mask_config_t maskConfig, HsaMemoryBuffer &programBuffer, uint32_t numWorkItems, out_data_t *pOutput, uint32_t *pResultMask = NULL) {

    PM4Queue queue;
    uint32_t *pAdjMask = NULL;
    uint32_t adjMask[maskConfig.numDwords];

    if (maskConfig.pInactiveMask) {
        if (adjustMask(adjMask, pMask, maskConfig)) {
            pAdjMask = adjMask;
        } else {
            // Adjusted mask is all zeroes, skip test and mark as passing.
            return true;
        }
    } else {
        pAdjMask = pMask;
    }

    Dispatch dispatch(programBuffer);
    dispatch.SetArgs(NULL, pOutput);
    dispatch.SetDim(numWorkItems, 1, 1);

    EXPECT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    EXPECT_SUCCESS_GPU(queue.SetCUMask(pAdjMask, maskConfig.numBits), gpuNode);

    dispatch.Submit(queue);
    dispatch.Sync();
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

    return validateTest(pMask, maskConfig, numWorkItems, pOutput, pResultMask);
}


/*
 * ExtendedCuMasking
 *
 * Newer implementation of CU mask testing that focuses on correctness of masking.
 *
 * Unlike previous implementations, this new implementation does not rely on performance
 * measurements to decide if the masking took place.   Instead, this implementation checks
 * if waves were executed on all the CUs enabled and only the CUs enabled.
 *
 * Implementation does a series of tests, new tests can be easily added as needed.
 *
 * For each test, these steps are performed:
 *
 * 1) Decide the units that are enabled for the test (SEs, SAs, WGPs).
 * 2) Generate a CU mask that specifies the WGPs enabled on each (SE,SA) pairs.
 * 3) Set the mask for the queue and run a special shader.
 * 4) Shader records in a buffer the unit that is used by the wave (SE,SA,WGP).
 * 5) Test program analyses the results and verifies if shader used all and only the
 *    WGP units specified by the mask.
 *
 * Multiple tests are done with different combinations.
 * There are (2^numWGPs - 1) possibilities, not everything can be tested.
 *
 * For each new ASIC supported, the following changes might be required:
 * 1) Minor shader changes to put fill information into buffer.
 * 2) Format of out_data_t struct.
 * 3) Changes to validation code.
 *
 */
static void extendedCuMasking(KFDTEST_PARAMETERS* pTestParameters) {

    int gpuNode = pTestParameters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParameters->pTestObject;
    const HSAuint32 m_FamilyId = pKFDQMTest->GetFamilyIdFromNodeId(gpuNode);

    if (m_FamilyId >= FAMILY_GFX12) {  // Supporting GFX12 and up for now

        // Lock to prevent interleave of logging on multigpu (multithreaded) testing
        static std::mutex logMutex;

        const HsaNodeProperties *pProps = pKFDQMTest->Get_NodeInfo()->GetNodeProperties(gpuNode);
        const uint32_t activeCU = (pProps->NumFComputeCores / pProps->NumSIMDPerCU);
        const uint32_t numSEs = pProps->NumShaderBanks;
        const uint32_t numSAperSE = pProps->NumArrays;
        const uint32_t numWGPperSA = pProps->NumCUPerArray / 2;
        const uint32_t maxCU = numSEs * numSAperSE * numWGPperSA * 2;

        std::ostringstream nodeStream;
        nodeStream << "(Node " << gpuNode << ")";
        const std::string nodeStr = nodeStream.str();

        logMutex.lock();
        LOG() << std::endl;
        LOG() << std::dec << "****** GFX Configuration " << nodeStr << " ******" << std::endl;
        LOG() << std::dec << "  Compute Cores (SIMD): " << std::setw(3) << pProps->NumFComputeCores << std::endl;
        LOG() << std::dec << "          SIMDs per CU: " << std::setw(3) << pProps->NumSIMDPerCU << std::endl;
        LOG() << std::dec << "            Active CUs: " << std::setw(3) << activeCU << std::endl;
        LOG() << std::dec << "               Max CUs: " << std::setw(3) << maxCU << std::endl;
        LOG() << std::dec << "        Shader Engines: " << std::setw(3) << numSEs << std::endl;
        LOG() << std::dec << "            SAs per SE: " << std::setw(3) << numSAperSE << std::endl;
        LOG() << std::dec << "           WGPs per SA: " << std::setw(3) << numWGPperSA << std::endl;
        LOG() << std::dec << "****************************************" << std::endl;
        logMutex.unlock();

        const uint32_t maskNumDwords = (maxCU + 31) / 32; /* Round up to the nearest multiple of 32 */
        const uint32_t maskNumBits = maskNumDwords * 32;


        uint32_t mask[maskNumDwords];
        uint32_t inactiveMask[maskNumDwords];

        mask_config_t maskConfig = { maskNumDwords, maskNumBits, numSEs, numSAperSE, numWGPperSA, NULL };

        /*
         * Note: On system with WGPs, CU bits in the same WGP must be either both set or both unset
         *       i.e. enabling/disabling is on a per-WGP basis.
         *
         * Format of CU Mask array (Assuming 4 SEs)
         *
         * Bit    Value    Masking
         *
         *  0,1    0x03     SE0 SA0 WGP0 (i.e. CU0 and CU1)
         *  2,3    0x0c     SE1 SA0 WGP0
         *  4,5    0x30     SE2 SA0 WGP0
         *  6,7    0xc0     SE3 SA0 WGP0
         *
         *  8,9    0x0300   SE0 SA1 WGP0
         * 10,11   0x0c00   SE1 SA1 WGP0
         * 12,13   0x3000   SE2 SA1 WGP0
         * 14,15   0xc000   SE3 SA1 WGP0
         *
         * 16,17   0x030000 SE0 SA0 WGP1
         * 18,19   0x030000 SE1 SA0 WGP1
         * ...
         * 32,33            SE0 SA0 WGP2
         * ...
         * 48,49            SE0 SA0 WGP3
         * ...
         *
         */

        /*
         * Number of work items needs to be sufficiently large to have enough work items for each WGP enabled.
         *
         * Using total number of WGPs multiplied by 16.
         *
         */
        const uint32_t numWorkItems = 16 * numSEs * numSAperSE * numWGPperSA;

        // Allocate buffers for program and output
        HsaMemoryBuffer programBuffer(PAGE_SIZE, gpuNode, true, false, true);
        HsaMemoryBuffer outputBuffer(((sizeof(out_data_t) * numWorkItems) + (PAGE_SIZE - 1)) & ~(PAGE_SIZE - 1), gpuNode, true, false, false);
        out_data_t *pOutput = outputBuffer.As<out_data_t *>();

        // Assemble shader
        Assembler *pAsm = pKFDQMTest->GetAssemblerFromNodeId(gpuNode);
        ASSERT_NOTNULL_GPU(pAsm, gpuNode);
        ASSERT_SUCCESS_GPU(pAsm->RunAssembleBuf(CheckCuMaskIsa, programBuffer.As<char*>()), gpuNode);


       /*
        * Check and record any inactive WPGs.
        *
        */
        memset(mask, 0, sizeof(mask));
        memset(inactiveMask, 0, sizeof(inactiveMask));

        // Use full mask and collect all active CUs in inactiveMask
        setCUMask(mask, maskConfig, -1, -1, -1);
        if (testCUMask(gpuNode, mask, maskConfig, programBuffer, numWorkItems, pOutput, inactiveMask)) {
            // Using full mask, if all CUs are used, we expect them to be all active.
            EXPECT_TRUE_GPU(activeCU == maxCU, gpuNode);
        } else {
            // Some CUs were not used, generate inactive mask and count inactive CUs.
            uint32_t inactiveCount = 0;

            // Flip bits and count inactive
            for (int i = 0; i < maskNumDwords; i++) {
                inactiveMask[i] = ~inactiveMask[i];
                inactiveCount += __builtin_popcount(inactiveMask[i]);
            }

            // Check if what we detected is consistent with info from KFD
            EXPECT_TRUE_GPU((activeCU + inactiveCount) == maxCU, gpuNode);

            maskConfig.pInactiveMask = inactiveMask;

            std::ostringstream logStr;
            logStr << nodeStr << " Inactive WGP detected: " << inactiveCount << "  0x" << std::hex << std::setw(8);
            for (int i = maskNumDwords - 1; i >= 0; i--) {
                logStr << inactiveMask[i];
            }
            LOG() << logStr.str() << std::endl;
        }


        /*
         * Generate symmetric test configuration for all (SE, SA, WGP) combinations, one level at a time.
         *
         * Other levels fully enabled.
         *
         * Example: If testing SE disablement, all SA/WGP are enabled on the SE that are used.
         *          If testing SA disablement, all SE are used, all WGP are enabled on the SA enabled.
         *
         */
        uint32_t totalConfigTested = 0;

        // All SE combination (0 not allowed, need at least one enabled)
        LOG() << nodeStr << " === Testing SE mask (" << ((1 << numSEs) - 1) << " configs)\n";
        for (int i = 1; i < (1 << numSEs); i++) {
            memset(mask, 0, sizeof(mask));
            DBG_PRINT("SE mask: 0x%x\n", i);
            setCUMask(mask, maskConfig, i, -1, -1);
            EXPECT_TRUE_GPU(testCUMask(gpuNode, mask, maskConfig, programBuffer, numWorkItems, pOutput), gpuNode);
            totalConfigTested++;
        }

        // All SA combinations (0 not allowed, need at least one enabled)
        LOG() << nodeStr << " === Testing SA mask (" << ((1 << numSAperSE) - 1) << " configs)\n";
        for (uint32_t i = 1; i < (1 << numSAperSE); i++) {
            memset(mask, 0, sizeof(mask));

            DBG_PRINT("SA mask: 0x%x\n", i);
            setCUMask(mask, maskConfig, -1, i, -1);
            EXPECT_TRUE_GPU(testCUMask(gpuNode, mask, maskConfig, programBuffer, numWorkItems, pOutput), gpuNode);
            totalConfigTested++;
        }

        // All WGP combinations (0 not allowed, need at least one enabled)
        LOG() << nodeStr << " === Testing WGP mask (" << ((1 << numWGPperSA) - 1) << " configs)\n";
        for (uint32_t i = 1; i < (1 << numWGPperSA); i++) {
            memset(mask, 0, sizeof(mask));

            DBG_PRINT("WGP mask: 0x%x\n", i);
            setCUMask(mask, maskConfig, -1, -1, i);
            EXPECT_TRUE_GPU(testCUMask(gpuNode, mask, maskConfig, programBuffer, numWorkItems, pOutput), gpuNode);
            totalConfigTested++;
        }

        /*
         * Linear Masking
         *
         * Enable one WGP at a time until they are all enabled.
         *
         */
        {
            uint32_t totalWGPs = numSEs * numSAperSE * numWGPperSA;

            LOG() << nodeStr << " === Testing linear mask (" << totalWGPs << " configs)\n";

            memset(mask, 0, sizeof(mask));

            for (int32_t i = 0; i < totalWGPs; i++) {
                mask[i / 16] |= (0x3 << (i * 2));

#if CUMASK_DEBUG
                printMask("  linear mask: ", mask, maskNumDwords);
#endif //CUMASK_DEBUG

                EXPECT_TRUE_GPU(testCUMask(gpuNode, mask, maskConfig, programBuffer, numWorkItems, pOutput), gpuNode);
                totalConfigTested++;
            }
        }

        /*
         * Random asymmetric config.
         *
         * Asymmetric, different WGPs/SAs are enabled/disabled on different SEs.
         *
         */
        {
            uint32_t randomCount = 1000;  // Total number of random test to perform
            uint32_t seed = 1;            // Specifying a seed to have deterministic random sequence

            srand(seed);

            LOG() << nodeStr << " === Testing " << randomCount << " random mask config...\n";

            for (uint32_t i = 0; i < randomCount; i++) {

                memset(mask, 0, sizeof(mask));

                uint32_t wgpLeft = maxCU / 2;   // init to total WGPs
                uint32_t maskIndex = 0;

                while (wgpLeft > 0) {
                    uint32_t wgpBlock = (wgpLeft > 16) ? 16 : wgpLeft;   // max 16 WGPs at a time
                    wgpLeft -= wgpBlock;

                    /*
                     * Pick random number between 0 to (2^wgpBlock - 1) - 1.
                     * Then add 1 to get random number between 1 to (2^wgpBlock - 1).
                     * This ensure that we don't end up with 0 for all the dwords in the mask.
                     */
                    uint32_t wgpMask = (rand() % ((1ULL << wgpBlock) - 1)) + 1;

                    // expand WGP mask to CU mask by doubling each individual bits.
                    uint32_t expandToCUMask = 0;
                    for (uint32_t j = 0; j < wgpBlock; j++) {
                        if (wgpMask & (1 << j)) {
                            expandToCUMask |= (0x3ULL << (j * 2));
                        }
                    }

                    DBG_PRINT("maskIndex: %u  fullWGPMask: 0x%08x  expand: 0x%08x\n", maskIndex, wgpMask, expandToCUMask);

                    mask[maskIndex++] = expandToCUMask;
                }

                EXPECT_TRUE_GPU(testCUMask(gpuNode, mask, maskConfig, programBuffer, numWorkItems, pOutput), gpuNode);
                totalConfigTested++;
            }
        }

        LOG() << std::endl;
        LOG() << nodeStr << " Total config tested: " << totalConfigTested << std::endl;
        LOG() << std::endl;

    } else {
        LOG() << "Skipping test: Test not supported for family ID 0x" << m_FamilyId << "." << std::endl;
    }
}

TEST_F(KFDQMTest, ExtendedCuMasking) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(extendedCuMasking));

    TEST_END
}

#undef CUMASK_DEBUG
#undef DBG_PRINT

// ====== End of ExtendedCUMasking Functions ====== //



/**
 * Apply CU masking where the number of CUs is equal across all Shader Engines
 * This will work due to the HW splitting the workload unevenly across the Shader
 * Engines when ((#ofCUs)/(#ofShaderEngines)) is not a whole number. The tests above
 * will not yield viable results when an uneven distribution of CUs is used over multiple
 * shader engines (e.g. 0x1000100030003), until the HW changes how it schedules work.
 */
void BasicCuMaskingEven(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;
    const HSAuint32 m_FamilyId = pKFDQMTest->GetFamilyIdFromNodeId(gpuNode);

    if (m_FamilyId >= FAMILY_VI) {
        const HsaNodeProperties *pNodeProperties = pKFDQMTest->Get_NodeInfo()->GetNodeProperties(gpuNode);
        uint32_t ActiveCU = (pNodeProperties->NumFComputeCores / pNodeProperties->NumSIMDPerCU);
        uint32_t numShaderEngines = pNodeProperties->NumShaderBanks;
        if (numShaderEngines == 1) {
            LOG() << "Skipping test: Only 1 Shader Engine present." << std::endl;
            return;
        }

        LOG() << std::dec << "# Compute cores: " << pNodeProperties->NumFComputeCores << std::endl;
        LOG() << std::dec << "# SIMDs per CU: " << pNodeProperties->NumSIMDPerCU << std::endl;
        LOG() << std::dec << "# Shader engines: " << numShaderEngines << std::endl;
        LOG() << std::dec << "# Active CUs: " << ActiveCU << std::endl;
        HSAint64 TimewithCU1, TimewithCU;
        uint32_t maskNumDwords = (ActiveCU + 31) / 32; /* Round up to the nearest multiple of 32 */
        uint32_t maskNumBits = maskNumDwords * 32;
        uint32_t mask[maskNumDwords];
        int numCuPerShader = ActiveCU / numShaderEngines;
        double ratio;

        /* In KFD we symmetrically map mask to all SEs:
         * mask[0] bit0 -> se0 cu0;
         * mask[0] bit1 -> se1 cu0;
         * ... (if # SE is 4)
         * mask[0] bit4 -> se0 cu1;
         * ...
         */
        /* Set Mask to 1 CU per SE */
        memset(mask, 0, maskNumDwords * sizeof(uint32_t));
        for (int i = 0; i < numShaderEngines; i++) {
            int maskIndex = (i / 32) % maskNumDwords;
            mask[maskIndex] |= 1 << (i % 32);
        }

        /* Execute once to get any HW optimizations out of the way */
        pKFDQMTest->TimeConsumedwithCUMask(gpuNode, mask, maskNumBits);

        LOG() << "Getting baseline performance numbers (1 CU per SE)" << std::endl;
        TimewithCU1 = pKFDQMTest->GetAverageTimeConsumedwithCUMask(gpuNode, mask, maskNumBits, 3);

        /* Each loop will add 1 more CU per SE. We use the mod and divide to handle
         * when SEs aren't distributed in multiples of 32 (e.g. Tonga)
         * OR the new bit in for simplicity instead of re-creating the mask each iteration
         */
        for (int x = 0; x < numCuPerShader; x++) {
            for (int se = 0; se < numShaderEngines; se++) {
                int offset = x * numShaderEngines + se;
                int maskIndex = (offset / 32) % maskNumDwords;
                mask[maskIndex] |= 1 << (offset % 32);
            }
            int nCUs = x + 1;

            TimewithCU = pKFDQMTest->TimeConsumedwithCUMask(gpuNode, mask, maskNumBits);
            ratio = (double)(TimewithCU1) / ((double)(TimewithCU) * nCUs);

            LOG() << "Expected performance of " << nCUs << " CU(s)/SE vs 1 CU/SE:" << std::endl;
            LOG() << std::setprecision(2) << pKFDQMTest->CuNegVariance << " <= " << std::fixed << std::setprecision(8)
                  << ratio << " <= " << std::setprecision(2) << pKFDQMTest->CuPosVariance << std::endl;

            EXPECT_TRUE_GPU((ratio >= pKFDQMTest->CuNegVariance) && (ratio <= pKFDQMTest->CuPosVariance), gpuNode);

            RECORD(ratio) << "Ratio-" << nCUs << "-CUs";
        }
    } else {
        LOG() << "Skipping test: Test not supported for family ID 0x" << m_FamilyId << "." << std::endl;
    }
}

TEST_F(KFDQMTest, BasicCuMaskingEven) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(BasicCuMaskingEven));

    TEST_END
}

void testQueuePriority(KFDTEST_PARAMETERS* pTestParamters, bool isSamePipe)
{
    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;
    const HSAuint32 m_FamilyId = pKFDQMTest->GetFamilyIdFromNodeId(gpuNode);

    Assembler* m_pAsm;
    m_pAsm = pKFDQMTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    if (m_FamilyId < FAMILY_VI) {
        LOG() << "Skipping test: Shader won't run on CI." << std::endl;
        return;
    }

    // Reduce test case if running on emulator
    // Reduction applies to all 3 dims (effect is cubic)
    const int scaleDown = (g_IsEmuMode ? 4 : 1);

    HsaMemoryBuffer syncBuf(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HSAint32 *syncBuffer = syncBuf.As<HSAint32*>();
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    //ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(LoopIsa, isaBuffer.As<char*>()));
	ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(LoopIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch[2] = {
        Dispatch(isaBuffer, true),
        Dispatch(isaBuffer, true)
    };

    const int queueCount = isSamePipe ? 13 : 2;
    int activeTaskBitmap = 0x3;
    HSAuint64 startTime, endTime[2];
    HsaEvent *pHsaEvent[2];
    int numEvent = 2;
    PM4Queue queue[queueCount];
    HSA_QUEUE_PRIORITY priority[2] = {
        HSA_QUEUE_PRIORITY_LOW,
        HSA_QUEUE_PRIORITY_HIGH
    };
    int i;

    /*
     * For different pipe variation:
     *   Only two queues are created, they should be on two different pipes.
     *
     * For same pipe variation:
     *   queue[2..12] are dummy queues. Create queue in this sequence to
     *   render queue[0] and queue[1] on same pipe with no assumptions
     *   about the number of pipes used by KFD. Queue #12 is a multiple
     *   of 1, 2, 3 and 4, so it falls on pipe 0 for any number of pipes
     */
	EXPECT_SUCCESS_GPU(queue[0].Create(gpuNode), gpuNode);  // Queue 0 is on Pipe 0
    if (isSamePipe) {
        for (i = 2; i < queueCount; i++)
            EXPECT_SUCCESS_GPU(queue[i].Create(gpuNode), gpuNode);
    }
    EXPECT_SUCCESS_GPU(queue[1].Create(gpuNode), gpuNode);

    for (i = 0; i < 2; i++) {
        syncBuffer[i] = -1;
        queue[i].Update(BaseQueue::DEFAULT_QUEUE_PERCENTAGE, priority[i], false);
        pHsaEvent[i] = dispatch[i].GetHsaEvent();
        pHsaEvent[i]->EventData.EventData.SyncVar.SyncVar.UserData = &syncBuffer[i];
        dispatch[i].SetDim(1024 / scaleDown , 16 / scaleDown, 16 / scaleDown);
    }

    startTime = GetSystemTickCountInMicroSec();
    for (i = 0; i < 2; i++)
        dispatch[i].Submit(queue[i]);

    while (activeTaskBitmap > 0) {
        hsaKmtWaitOnMultipleEvents(pHsaEvent, numEvent, false, g_TestTimeOut);
        for (i = 0; i < 2; i++) {
            if ((activeTaskBitmap & (1 << i)) && (syncBuffer[i] == pHsaEvent[i]->EventId)) {
                endTime[i] = GetSystemTickCountInMicroSec();
                activeTaskBitmap &= ~(1 << i);
            }
        }
    }

    for (i = 0; i < 2; i++) {
        int usecs = endTime[i] - startTime;
        LOG() << "Task priority: " << std::dec << priority[i] << "\t";
        LOG() << "Task duration: " << std::dec << std::setw(10) << usecs << " usecs" << std::endl;
    }

    for (i = 0; i < queueCount; i++) {
        EXPECT_SUCCESS_GPU(queue[i].Destroy(), gpuNode);
    }
}

static void QueuePriorityOnDifferentPipe(KFDTEST_PARAMETERS* pTestParamters) {

	testQueuePriority(pTestParamters, false);
}

TEST_F(KFDQMTest, QueuePriorityOnDifferentPipe) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(QueuePriorityOnDifferentPipe));

    TEST_END
}

void QueuePriorityOnSamePipe(KFDTEST_PARAMETERS* pTestParamters) {

    testQueuePriority(pTestParamters, true);
}

TEST_F(KFDQMTest, QueuePriorityOnSamePipe) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(QueuePriorityOnSamePipe));

    TEST_END
}

void KFDQMTest::SyncDispatch(const HsaMemoryBuffer& isaBuffer, void* pSrcBuf, void* pDstBuf, int node) {
    PM4Queue queue;

    if (node == -1)
        node = m_NodeInfo.HsaDefaultGPUNode();

    ASSERT_GE_GPU(node, 0, node) << "failed to get GPU Node";

    Dispatch dispatch(isaBuffer);
    dispatch.SetArgs(pSrcBuf, pDstBuf);
    dispatch.SetDim(1, 1, 1);

    ASSERT_SUCCESS_GPU(queue.Create(node), node);

    dispatch.Submit(queue);
    dispatch.Sync();

    EXPECT_SUCCESS_GPU(queue.Destroy(), node);
}

void EmptyDispatch(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDQMTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(NoopIsa, isaBuffer.As<char*>()), gpuNode);

    pKFDQMTest->SyncDispatch(isaBuffer, NULL, NULL, gpuNode);

}

TEST_F(KFDQMTest, EmptyDispatch) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(EmptyDispatch));

    TEST_END
}

void SimpleWriteDispatch(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDQMTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer srcBuffer(PAGE_SIZE, gpuNode, false);
    HsaMemoryBuffer destBuffer(PAGE_SIZE, gpuNode);

    srcBuffer.Fill(0x01010101);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()),gpuNode);

    pKFDQMTest->SyncDispatch(isaBuffer, srcBuffer.As<void*>(), destBuffer.As<void*>(), gpuNode);

    EXPECT_EQ(destBuffer.As<unsigned int*>()[0], 0x01010101);

}

TEST_F(KFDQMTest, SimpleWriteDispatch) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(SimpleWriteDispatch));

    TEST_END
}

static void MultipleCpQueuesStressDispatch(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDQMTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    static const unsigned int MAX_CP_QUEUES = 16;

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer srcBuffer(PAGE_SIZE, gpuNode, false);
    HsaMemoryBuffer destBuffer(PAGE_SIZE, gpuNode);

    unsigned int* src = srcBuffer.As<unsigned int*>();
    unsigned int* dst = destBuffer.As<unsigned int*>();

    static const HSAuint64 TEST_TIME_SEC = 15;
    HSAuint64 initialTime, curTime;
    unsigned int numIter = 0;
    HSAuint64 timePassed = 0;

    unsigned int i;
    PM4Queue queues[MAX_CP_QUEUES];
    Dispatch* dispatch[MAX_CP_QUEUES];

    destBuffer.Fill(0xFF);

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    for (i = 0; i < MAX_CP_QUEUES; ++i)
        ASSERT_SUCCESS_GPU(queues[i].Create(gpuNode), gpuNode) << " QueueId=" << i;

    initialTime = GetSystemTickCountInMicroSec();

    do {
        for (i = 0; i < MAX_CP_QUEUES; ++i) {
            dispatch[i] = new Dispatch(isaBuffer);
            src[i] = numIter;
            dst[i] = 0xff;
            dispatch[i]->SetArgs(&src[i], &dst[i]);
            dispatch[i]->SetDim(1, 1, 1);
            dispatch[i]->Submit(queues[i]);
        }
        for (i = 0; i < MAX_CP_QUEUES; ++i) {
            dispatch[i]->Sync();
            EXPECT_EQ_GPU(dst[i], src[i], gpuNode);
            delete dispatch[i];
        }
        ++numIter;
        curTime = GetSystemTickCountInMicroSec();
        timePassed = (curTime - initialTime) / 1000000;
    } while (timePassed < TEST_TIME_SEC);

    LOG() << "Total iterated : " << std::dec << numIter << std::endl;

    for (i = 0; i < MAX_CP_QUEUES; ++i)
       EXPECT_SUCCESS_GPU(queues[i].Destroy(), gpuNode);


}

TEST_F(KFDQMTest, MultipleCpQueuesStressDispatch) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(MultipleCpQueuesStressDispatch));

    TEST_END
}

static void CpuWriteCoherence(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    PM4Queue queue;

    HsaMemoryBuffer destBuf(PAGE_SIZE, gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    HsaEvent *event;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event), gpuNode);

    /* The queue might be full and we fail to submit. There is always one word space unused in queue.
     * So let rptr one step ahead then we continually submit packet.
     */
    queue.PlaceAndSubmitPacket(PM4NopPacket());
    queue.Wait4PacketConsumption();
    EXPECT_EQ(1, queue.Rptr());

    do {
        queue.PlaceAndSubmitPacket(PM4NopPacket());
    } while (queue.Wptr() != 0);

    queue.Wait4PacketConsumption();

    EXPECT_EQ_GPU(0, queue.Rptr(), gpuNode);

    /* Now that the GPU has cached the PQ contents, we modify them in CPU cache and
     * ensure that the GPU sees the updated value:
     */
    queue.PlaceAndSubmitPacket(PM4WriteDataPacket(destBuf.As<unsigned int*>(), 0x42, 0x42));

    queue.Wait4PacketConsumption(event);

    WaitOnValue(destBuf.As<unsigned int*>(), 0x42);

    hsaKmtDestroyEvent(event);
}

TEST_F(KFDQMTest, CpuWriteCoherence) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(CpuWriteCoherence));

    TEST_END
}

static void CreateAqlCpQueue(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    AqlQueue queue;

    HsaMemoryBuffer pointers(PAGE_SIZE, gpuNode, /*zero*/true, /*local*/false, /*exec*/false, /*isScratch */false, /* isReadOnly */false, /* isUncached */false, /* NonPaged */g_baseTest->NeedNonPagedWptr(gpuNode));

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode, PAGE_SIZE, pointers.As<HSAuint64 *>()), gpuNode);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDQMTest, CreateAqlCpQueue) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(CreateAqlCpQueue));

    TEST_END
}

static void QueueLatency(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;
    HSAuint32 m_FamilyId = pKFDQMTest->GetFamilyIdFromNodeId(gpuNode);

    PM4Queue queue;
    const int queueSize = PAGE_SIZE * 2;
    const int packetSize = PM4ReleaseMemoryPacket(m_FamilyId, 0, 0, 0, 0, 0).SizeInBytes();
    /* We always leave one NOP(dword) empty after packet which is required by ring itself.
     * We also place NOPs when queue wraparound to avoid crossing buffer end. See PlacePacket().
     * So the worst case is that we need two packetSize space to place one packet.
     * Like below, N=NOP,E=Empty,P=Packet.
     * |E|E|E|E|E|E|E|rptr...wptr|E|E|E|E|E| ---> |P|P|P|P|P|P|E|rptr...wptr|N|N|N|N|N|
     * So to respect that, we reserve packetSize space for these additional NOPs.
     * Also we reserve the remainder of the division by packetSize explicitly.
     * Reserve another packetSize for event-based wait which uses a releseMemory packet.
     */
    const int reservedSpace = packetSize + queueSize % packetSize + packetSize;
    const int slots = (queueSize - reservedSpace) / packetSize;
    HSAint64 queue_latency_avg = 0, queue_latency_min, queue_latency_max, queue_latency_med;
    HSAint64 overhead, workload;
    HSAint64 *queue_latency_arr = reinterpret_cast<HSAint64*>(calloc(slots, sizeof(HSAint64)));
    const int skip = 2;
    const char *fs[skip] = {"1st", "2nd"};
    HsaClockCounters *ts;
    HSAuint64 *qts;
    int i = 0;

    ASSERT_NE_GPU((HSAuint64)queue_latency_arr, 0, gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode, queueSize), gpuNode);

    LOG() << std::dec << "Queue Submit NanoSeconds (" << slots << " Packets)" << std::endl;

    HsaMemoryBuffer buf(ALIGN_UP(slots * sizeof(HsaClockCounters), PAGE_SIZE), 0);
    ts = buf.As<HsaClockCounters*>();

    HsaMemoryBuffer qbuf(ALIGN_UP(slots * sizeof(HSAuint64), PAGE_SIZE), 0);
    qts = qbuf.As<HSAuint64*>();

    HsaEvent *event;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event), gpuNode);

    /* GpuCounter overhead*/
    do {
        hsaKmtGetClockCounters(gpuNode, &ts[i]);
    } while (++i < slots);
    overhead = ts[slots-1].GPUClockCounter - ts[0].GPUClockCounter;
    overhead /= 2 * (slots - 1);

    /* Submit packets serially*/
    i = 0;
    do {
        queue.PlacePacket(PM4ReleaseMemoryPacket(m_FamilyId, true,
                    (HSAuint64)&qts[i],
                    0,
                    true,
                    1));
        hsaKmtGetClockCounters(gpuNode, &ts[i]);
        queue.SubmitPacket();
        queue.Wait4PacketConsumption(event);
    } while (++i < slots);

    /* Calculate timing which includes workload and overhead*/
    i = 0;
    do {
        HSAint64 queue_latency = qts[i] - ts[i].GPUClockCounter;

        EXPECT_GE_GPU(queue_latency, 0, gpuNode);

        queue_latency_arr[i] = queue_latency;
        if (i >= skip)
            queue_latency_avg += queue_latency;
    } while (++i < slots);
    /* Calculate avg from packet[skip, slots-1] */
    queue_latency_avg /= (slots - skip);

    /* Workload of queue packet itself */
    i = 0;
    do {
        queue.PlacePacket(PM4ReleaseMemoryPacket(m_FamilyId, true,
                    (HSAuint64)&qts[i],
                    0,
                    true,
                    1));
    } while (++i < slots);
    queue.SubmitPacket();
    queue.Wait4PacketConsumption(event);

    hsaKmtDestroyEvent(event);
    /* qts[i] records the timestamp of the end of packet[i] which is
     * approximate that of the beginging of packet[i+1].
     * The workload total is [0, skip], [skip+1, slots-1].
     * And We ignore [0, skip], that means we ignore (skip+1) packets.
     */
    workload = qts[slots - 1] - qts[skip];
    workload /= (slots - 1 - skip);

    EXPECT_GE_GPU(workload, 0, gpuNode);

    i = 0;
    do {
        /* The queue_latency is not that correct as the workload and overhead are average*/
        queue_latency_arr[i] -= workload + overhead;
        /* The First submit takes an HSAint64 time*/
        if (i < skip)
            LOG() << "Queue Latency " << fs[i] << ": \t" << CounterToNanoSec(queue_latency_arr[i]) << std::endl;
    } while (++i < slots);

    std::sort(queue_latency_arr + skip, queue_latency_arr + slots);

    queue_latency_min = queue_latency_arr[skip];
    queue_latency_med = queue_latency_arr[(slots+skip)/2];
    queue_latency_max = queue_latency_arr[slots-1];

    LOG() << "Queue Latency Avg:     \t" << CounterToNanoSec(queue_latency_avg) << std::endl;
    LOG() << "Queue Latency Min:     \t" << CounterToNanoSec(queue_latency_min) << std::endl;
    LOG() << "Queue Latency Median:  \t" << CounterToNanoSec(queue_latency_med) << std::endl;
    LOG() << "Queue Latency Max:     \t" << CounterToNanoSec(queue_latency_max) << std::endl;
    LOG() << "Queue Packet Workload: \t" << CounterToNanoSec(workload) << std::endl;
    LOG() << "Get GpuCounter Overhead: \t" << CounterToNanoSec(overhead) << std::endl;

    RECORD(CounterToNanoSec(queue_latency_avg)) << "Queue-Latency-Avg";
    RECORD(CounterToNanoSec(queue_latency_min)) << "Queue-Latency-Min";
    RECORD(CounterToNanoSec(queue_latency_med)) << "Queue-Latency-Med";
    RECORD(CounterToNanoSec(queue_latency_max)) << "Queue-Latency-Max";
    RECORD(CounterToNanoSec(workload)) << "Queue-Packet-Workload";
    RECORD(CounterToNanoSec(overhead)) << "GpuCounter-Overhead";

}

TEST_F(KFDQMTest, QueueLatency) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(QueueLatency));

    TEST_END
}

static void CpQueueWraparound(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    PM4Queue queue;

    HsaMemoryBuffer destBuf(PAGE_SIZE, gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    HsaEvent *event;
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event), gpuNode);

    for (unsigned int pktIdx = 0; pktIdx <= PAGE_SIZE/sizeof(PM4WRITE_DATA_CI); ++pktIdx) {
        queue.PlaceAndSubmitPacket(PM4WriteDataPacket(destBuf.As<unsigned int*>(), pktIdx, pktIdx));
        queue.Wait4PacketConsumption(event);
        WaitOnValue(destBuf.As<unsigned int*>(), pktIdx);
    }

    for (unsigned int pktIdx = 0; pktIdx <= PAGE_SIZE/sizeof(PM4WRITE_DATA_CI); ++pktIdx) {
        queue.PlaceAndSubmitPacket(PM4WriteDataPacket(destBuf.As<unsigned int*>(), pktIdx, pktIdx));
        queue.Wait4PacketConsumption(event);
        WaitOnValue(destBuf.As<unsigned int*>(), pktIdx);
    }

    hsaKmtDestroyEvent(event);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

}

TEST_F(KFDQMTest, CpQueueWraparound) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(CpQueueWraparound));

    TEST_END
}

static void SdmaQueueWraparound(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    int bufSize = PAGE_SIZE;

    SDMAQueue queue;

    HsaMemoryBuffer destBuf(bufSize << 1, gpuNode, false);
    HsaMemoryBuffer srcBuf(bufSize, gpuNode, false);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    for (unsigned int pktIdx = 0;  pktIdx <= queue.Size()/sizeof(SDMA_PKT_COPY_LINEAR); ++pktIdx) {
        destBuf.Fill(0x0);
        srcBuf.Fill(pktIdx);
        queue.PlaceAndSubmitPacket(
                SDMACopyDataPacket(queue.GetFamilyId(), destBuf.As<unsigned int*>(), srcBuf.As<unsigned int*>(), bufSize));
        queue.PlaceAndSubmitPacket(
                SDMAWriteDataPacket(queue.GetFamilyId(), destBuf.As<unsigned int*>() + bufSize/4, 0x02020202));
        queue.Wait4PacketConsumption();

        EXPECT_TRUE_GPU(WaitOnValue(destBuf.As<unsigned int*>() + bufSize/4, 0x02020202), gpuNode);

        EXPECT_SUCCESS_GPU(memcmp(
                destBuf.As<unsigned int*>(), srcBuf.As<unsigned int*>(), bufSize), gpuNode);
    }

    for (unsigned int pktIdx = 0; pktIdx <= queue.Size()/sizeof(SDMA_PKT_WRITE_UNTILED); ++pktIdx) {
        queue.PlaceAndSubmitPacket(SDMAWriteDataPacket(queue.GetFamilyId(), destBuf.As<unsigned int*>(), pktIdx));
        queue.Wait4PacketConsumption();
        WaitOnValue(destBuf.As<unsigned int*>(), pktIdx);
    }

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDQMTest, SdmaQueueWraparound) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(SdmaQueueWraparound));

    TEST_END
}

struct AtomicIncThreadParams {
    HSAint64* pDest;
    volatile unsigned int count;
    volatile bool loop;
};

unsigned int AtomicIncThread(void* pCtx) {
    AtomicIncThreadParams* pArgs = reinterpret_cast<AtomicIncThreadParams*>(pCtx);

    while (pArgs->loop) {
        AtomicInc(pArgs->pDest);
        ++pArgs->count;
    }

    LOG() << "CPU atomic increments finished" << std::endl;

    return 0;
}

static void Atomics(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    Assembler* m_pAsm;
    m_pAsm = pKFDQMTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    if (!hasPciAtomicsSupport(gpuNode)) {
        LOG() << "Skipping test: Node doesn't support Atomics." << std::endl;
        return;
    }

    HsaMemoryBuffer isaBuf(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer destBuf(PAGE_SIZE, gpuNode);

    PM4Queue queue;

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(AtomicIncIsa, isaBuf.As<char*>()), gpuNode);

    Dispatch dispatch(isaBuf);
    dispatch.SetArgs(destBuf.As<void*>(), NULL);
    dispatch.SetDim(1024, 1, 1);

    hsaKmtSetMemoryPolicy(gpuNode, HSA_CACHING_CACHED, HSA_CACHING_CACHED, NULL, 0);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    AtomicIncThreadParams params;
    params.pDest = destBuf.As<HSAint64*>();
    params.loop = true;
    params.count = 0;

    uint64_t threadId;

    ASSERT_EQ_GPU(true, StartThread(&AtomicIncThread, &params, threadId), gpuNode);

    LOG() << "Waiting for CPU to atomic increment 1000 times" << std::endl;

    while (params.count < 1000)
        {}

    LOG() << "Submitting the GPU atomic increment shader" << std::endl;

    dispatch.Submit(queue);
    dispatch.Sync();

    params.loop = false;

    WaitForThread(threadId);

    EXPECT_EQ_GPU(destBuf.As<unsigned int*>()[0], 1024 + params.count, gpuNode);

    LOG() << "GPU increments: 1024, CPU increments: " << std::dec
            << params.count << std::endl;

    queue.Destroy();
}

TEST_F(KFDQMTest, Atomics) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(Atomics));

    TEST_END
}

TEST_F(KFDQMTest, mGPUShareBO) {
    TEST_START(TESTPROFILE_RUNALL);

    unsigned int src_node = 2;
    unsigned int dst_node = 1;

    if (g_TestDstNodeId != -1 && g_TestNodeId != -1) {
        src_node = g_TestNodeId;
        dst_node = g_TestDstNodeId;
    }

    HsaMemoryBuffer shared_addr(PAGE_SIZE, dst_node, true, false, false, false);

    HsaMemoryBuffer srcNodeMem(PAGE_SIZE, src_node);
    HsaMemoryBuffer dstNodeMem(PAGE_SIZE, dst_node);

    /* Handle ISA to write to local memory BO */
    HsaMemoryBuffer isaBufferSrc(PAGE_SIZE, src_node, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer isaBufferDst(PAGE_SIZE, dst_node, true/*zero*/, false/*local*/, true/*exec*/);

    srcNodeMem.Fill(0x05050505);

    ASSERT_SUCCESS(m_pAsm->RunAssemble(CopyDwordIsa));

    m_pAsm->CopyInstrStream(isaBufferSrc.As<char*>());
    SyncDispatch(isaBufferSrc, srcNodeMem.As<void*>(), shared_addr.As<void *>(), src_node);

    m_pAsm->CopyInstrStream(isaBufferDst.As<char*>());
    SyncDispatch(isaBufferDst, shared_addr.As<void *>(), dstNodeMem.As<void*>(), dst_node);

    EXPECT_EQ(dstNodeMem.As<unsigned int*>()[0], 0x05050505);

    EXPECT_SUCCESS(shared_addr.UnmapMemToNodes(&dst_node, 1));

    TEST_END
}

static void
sdma_copy(HSAuint32 node, void *src, void *const dst[], int n, HSAuint64 size) {
    SDMAQueue sdmaQueue;
    HsaEvent *event;
    ASSERT_SUCCESS(CreateQueueTypeEvent(false, false, node, &event));
    ASSERT_SUCCESS(sdmaQueue.Create(node));
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(), dst, src, n, size));
    sdmaQueue.Wait4PacketConsumption(event);
    EXPECT_SUCCESS(sdmaQueue.Destroy());
    hsaKmtDestroyEvent(event);
}

static void
sdma_fill(HSAint32 node, void *dst, unsigned int data, HSAuint64 size) {
    SDMAQueue sdmaQueue;
    HsaEvent *event;
    ASSERT_SUCCESS(CreateQueueTypeEvent(false, false, node, &event));
    ASSERT_SUCCESS(sdmaQueue.Create(node));
    sdmaQueue.PlaceAndSubmitPacket(SDMAFillDataPacket(sdmaQueue.GetFamilyId(), dst, data, size));
    sdmaQueue.Wait4PacketConsumption(event);
    EXPECT_SUCCESS(sdmaQueue.Destroy());
    hsaKmtDestroyEvent(event);
}

TEST_F(KFDQMTest, P2PTest) {
    TEST_START(TESTPROFILE_RUNALL);
    if (!hsakmt_is_dgpu()) {
        LOG() << "Skipping test: Two GPUs are required, but no dGPUs are present." << std::endl;
        return;
    }

    const std::vector<int> gpuNodes = m_NodeInfo.GetNodesWithGPU();
    if (gpuNodes.size() < 2) {
        LOG() << "Skipping test: At least two GPUs are required." << std::endl;
        return;
    }
    std::vector<int> nodes;

    /* This test simulates RT team's P2P part in IPCtest:
     *
     * +------------------------------------------------+
     * |         gpu1           gpu2           gpuX     |
     * |gpu1 mem ----> gpu2 mem ----> gpuX mem          |
     * |        \               \               \  mGPUShareBO     |
     * |         \               \               \      |
     * |    system buffer   system buffer  system buffer|
     * +------------------------------------------------+
     *
     * Copy data from current GPU memory to next GPU memory and system memory
     * Using current GPU, aka p2p push.
     * Verify the system buffer has the expected content after each push.
     */

    /* Users can use "--node=gpu1 --dst_node=gpu2" to specify devices */
    if (g_TestDstNodeId != -1 && g_TestNodeId != -1) {
        nodes.push_back(g_TestNodeId);
        nodes.push_back(g_TestDstNodeId);

        if (!m_NodeInfo.IsPeerAccessibleByNode(g_TestNodeId, g_TestDstNodeId)) {
            LOG() << "Skipping test: Dst GPU specified is not peer-accessible." << std::endl;
            return;
        }
        if (nodes[0] == nodes[1]) {
            LOG() << "Skipping test: Different GPUs must be specified (2 GPUs required)." << std::endl;
            return;
        }
    } else {
        nodes = m_NodeInfo.GetNodesWithGPU();
        if (nodes.size() < 2) {
            LOG() << "Skipping test: Test requires at least one large bar GPU." << std::endl;
            LOG() << "               or two GPUs are XGMI connected." << std::endl;
            return;
        }
    }

    HSAuint32 *sysBuf;
    HSAuint32 size = 16ULL<<20;  // bigger than 16MB to test non-contiguous memory
    HsaMemFlags memFlags = {0};
    HsaMemMapFlags mapFlags = {0};
    memFlags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    memFlags.ui32.HostAccess = 0;
    memFlags.ui32.NonPaged = 1;
    memFlags.ui32.NoNUMABind = 1;
    unsigned int end = size / sizeof(HSAuint32) - 1;

    /* 1. Allocate a system buffer and allow the access to GPUs */
    EXPECT_SUCCESS(hsaKmtAllocMemory(0, size, m_MemoryFlags,
                                     reinterpret_cast<void **>(&sysBuf)));
    EXPECT_SUCCESS(hsaKmtMapMemoryToGPUNodes(sysBuf, size, NULL,
                                             mapFlags, nodes.size(), (HSAuint32 *)&nodes[0]));
#define MAGIC_NUM 0xdeadbeaf

    /* First GPU fills mem with MAGIC_NUM */
    void *src, *dst;
    HSAuint32 cur = nodes[0], next;
    ASSERT_SUCCESS(hsaKmtAllocMemory(cur, size, memFlags, reinterpret_cast<void**>(&src)));
    ASSERT_SUCCESS(hsaKmtMapMemoryToGPU(src, size, NULL));
    sdma_fill(cur, src, MAGIC_NUM, size);

    for (unsigned i = 1; i <= nodes.size(); i++) {
        int n;
        memset(sysBuf, 0, size);

        /* Last GPU just copy mem to sysBuf*/
        if (i == nodes.size()) {
               n = 1;
               next = 0;/*system memory node*/
               dst = 0;
        } else {
            n = 2;
            next = nodes[i];

            /* check if cur access next node */
            if (!m_NodeInfo.IsPeerAccessibleByNode(next, cur))
                continue;

            ASSERT_SUCCESS(hsaKmtAllocMemory(next, size, memFlags, reinterpret_cast<void**>(&dst)));
            ASSERT_SUCCESS(hsaKmtMapMemoryToGPU(dst, size, NULL));
        }

        LOG() << "Test " << cur << " -> " << next << std::endl;
        /* Copy to sysBuf and next GPU*/
        void *dst_array[] = {sysBuf, dst};
        sdma_copy(cur, src, dst_array, n, size);

        /* Verify the data*/
        EXPECT_EQ(sysBuf[0], MAGIC_NUM);
        EXPECT_EQ(sysBuf[end], MAGIC_NUM);

        LOG() << "PASS " << cur << " -> " << next << std::endl;

        EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(src));
        EXPECT_SUCCESS(hsaKmtFreeMemory(src, size));

        cur = next;
        src = dst;
    }

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(sysBuf));
    EXPECT_SUCCESS(hsaKmtFreeMemory(sysBuf, size));

    TEST_END
}

static void PM4EventInterrupt(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;
    HSAuint32 m_FamilyId = pKFDQMTest->GetFamilyIdFromNodeId(gpuNode);

    const HSAuint64 bufSize = PAGE_SIZE;
    const int packetCount = bufSize / sizeof(unsigned int);
    const int totalPacketSize = packetCount * PM4WriteDataPacket(0, 0).SizeInBytes() +
                                                PM4ReleaseMemoryPacket(m_FamilyId, 0, 0, 0).SizeInBytes();
    const int queueSize = RoundToPowerOf2(totalPacketSize);

    /* Reduce number of iteration if running with emulator. */
    const int numIter = (g_IsEmuMode ? 32 : 1024);

    /* 4 PM4 queues will be running at same time.*/
    const int numPM4Queue = 4;
    HsaEvent *event[numPM4Queue];
    PM4Queue queue[numPM4Queue];
    HsaMemoryBuffer *destBuf[numPM4Queue];
    unsigned int *buf[numPM4Queue];

    for (int i = 0; i < numPM4Queue; i++) {
        destBuf[i] = new HsaMemoryBuffer(bufSize, gpuNode, true, false); // System memory
        buf[i] = destBuf[i]->As<unsigned int *>();
    }

    /* A simple loop here to give more pressure.*/
    for (int test_count = 0; test_count < numIter; test_count++) {
        for (int i = 0; i < numPM4Queue; i++) {
            ASSERT_SUCCESS_GPU(queue[i].Create(gpuNode, queueSize), gpuNode);
            ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event[i]), gpuNode);

            /* Let CP have some workload first.*/
            for(int index = 0; index < packetCount; index++)
                queue[i].PlacePacket(PM4WriteDataPacket(buf[i] + index, 0xdeadbeaf));

            /* releaseMemory packet makes sure all previous written data is visible.*/
            queue[i].PlacePacket(PM4ReleaseMemoryPacket(m_FamilyId, 0,
                        reinterpret_cast<HSAuint64>(event[i]->EventData.HWData2),
                        event[i]->EventId,
                        true));
        }

        for (int i = 0; i < numPM4Queue; i++)
            queue[i].SubmitPacket();

        for (int i = 0; i < numPM4Queue; i++) {
            EXPECT_SUCCESS_GPU(hsaKmtWaitOnEvent(event[i], g_TestTimeOut), gpuNode);
            EXPECT_EQ_GPU(buf[i][0], 0xdeadbeaf, gpuNode);
            EXPECT_EQ_GPU(buf[i][packetCount - 1], 0xdeadbeaf, gpuNode);
            memset(buf[i], 0, bufSize);
        }

        for (int i = 0; i < numPM4Queue; i++) {
            EXPECT_SUCCESS_GPU(queue[i].Destroy(), gpuNode);
            EXPECT_SUCCESS_GPU(hsaKmtDestroyEvent(event[i]), gpuNode);
        }
    }

    for (int i = 0; i < numPM4Queue; i++)
        delete destBuf[i];
}

TEST_F(KFDQMTest, PM4EventInterrupt) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(PM4EventInterrupt));

    TEST_END
}

#include "KFDTestUtilQueue.hpp"
static void SdmaEventInterrupt(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;

    const HSAuint64 bufSize = 4 << 20;
    HsaMemoryBuffer srcBuf(bufSize, 0); // System memory.

    HSAuint64 *src = srcBuf.As<HSAuint64*>();
    TimeStamp *tsbuf = srcBuf.As<TimeStamp*>();
    tsbuf = reinterpret_cast<TimeStamp *>ALIGN_UP(tsbuf, sizeof(TimeStamp));

    /* Have 3 queues created for test.*/
    const int numSDMAQueue = 3;
    HsaEvent *event[numSDMAQueue];
    SDMAQueue queue[numSDMAQueue];
    HsaMemoryBuffer *destBuf[numSDMAQueue];
    HSAuint64 *dst[numSDMAQueue];

    for (int i = 0; i < numSDMAQueue; i++) {
        destBuf[i] = new HsaMemoryBuffer(bufSize, gpuNode, true, false); // System memory
        dst[i] = destBuf[i]->As<HSAuint64*>();
    }

    /* Test 1 queue, 2 queues, 3 queues running at same time one by one.*/
    for (int testSDMAQueue = 1; testSDMAQueue <= numSDMAQueue; testSDMAQueue++)
        /* A simple loop here to give more pressure.*/
        for (int test_count = 0; test_count < 2048; test_count++) {
            for (int i = 0; i < testSDMAQueue; i++) {
                TimeStamp *ts = tsbuf + i * 32;
                ASSERT_SUCCESS_GPU(queue[i].Create(gpuNode), gpuNode);
                /* FIXME
                 * We create event every time along with queue.
                 * However that will significantly enhance the failure of sdma event timeout.
                 */
                ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &event[i]), gpuNode);

                /* Get the timestamp directly. The first member of HsaClockCounters and TimeStamp is GPU clock counter.*/
                hsaKmtGetClockCounters(gpuNode, reinterpret_cast<HsaClockCounters*>(&ts[0]));
                /* Let sDMA have some workload first.*/
                queue[i].PlacePacket(SDMATimePacket(&ts[1]));
                queue[i].PlacePacket(
                        SDMACopyDataPacket(queue[i].GetFamilyId(), dst[i], src, bufSize));
                queue[i].PlacePacket(SDMATimePacket(&ts[2]));
                queue[i].PlacePacket(
                        SDMAFencePacket(queue[i].GetFamilyId(),
                                reinterpret_cast<void*>(event[i]->EventData.HWData2), event[i]->EventId));
                queue[i].PlacePacket(SDMATimePacket(&ts[3]));
                queue[i].PlacePacket(SDMATrapPacket(event[i]->EventId));
                queue[i].PlacePacket(SDMATimePacket(&ts[4]));

                /* Will verify the value of srcBuf and destBuf later. Give it a different value each time.*/
                src[0] = ts[0].timestamp;
            }

            for (int i = 0; i < testSDMAQueue; i++)
                queue[i].SubmitPacket();

            for (int i = 0; i < testSDMAQueue; i++) {
                TimeStamp *ts = tsbuf + i * 32;
                HSAKMT_STATUS ret = hsaKmtWaitOnEvent(event[i], g_TestTimeOut);

                if (dst[i][0] != src[0])
                    WARN() << "SDMACopyData FAIL! " << std::dec
                        << dst[i][0] << " VS " << src[0] << std::endl;

                if (ret == HSAKMT_STATUS_SUCCESS) {
                    for (int i = 1; i <= 4; i++)
                        /* Is queue latency too big? The workload is really small.*/
                        if (CounterToNanoSec(ts[i].timestamp - ts[i - 1].timestamp) > 1000000000)
                            WARN() << "SDMA queue latency is bigger than 1s!" << std::endl;
                } else {
                    WARN() << "Event On Queue " << testSDMAQueue << ":" << i
                        << " Timeout, try to resubmit packets!" << std::endl;

                    queue[i].SubmitPacket();

                    if (hsaKmtWaitOnEvent(event[i], g_TestTimeOut) == HSAKMT_STATUS_SUCCESS)
                        WARN() << "The timeout event is signaled!" << std::endl;
                    else
                        WARN() << "The timeout event is lost after resubmit!" << std::endl;

                    LOG() << "Time Consumption (ns)" << std::endl;
                    for (int i = 1; i <= 4; i++)
                        LOG() << std::dec << i << ": "
                            << CounterToNanoSec(ts[i].timestamp - ts[i - 1].timestamp) << std::endl;
                }

                EXPECT_SUCCESS_GPU(ret, gpuNode);
            }

            for (int i = 0; i < testSDMAQueue; i++) {
                EXPECT_SUCCESS_GPU(queue[i].Destroy(), gpuNode);
                EXPECT_SUCCESS_GPU(hsaKmtDestroyEvent(event[i]), gpuNode);
            }
        }

    for (int i = 0; i < numSDMAQueue; i++)
        delete destBuf[i];

}

TEST_F(KFDQMTest, SdmaEventInterrupt) {
    TEST_START(TESTPROFILE_RUNALL)

     ASSERT_SUCCESS(KFDTest_Launch(SdmaEventInterrupt));

    TEST_END
}

#define DOORBELL_WRITE_USE_SDMA
static void GPUDoorbellWrite(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDQMTest* pKFDQMTest = (KFDQMTest*)pTestParamters->pTestObject;
    HSAuint32 m_FamilyId = pKFDQMTest->GetFamilyIdFromNodeId(gpuNode);

    HsaMemoryBuffer destBuf(PAGE_SIZE, 0, true);
    PM4Queue pm4Queue;
#ifdef DOORBELL_WRITE_USE_SDMA
    SDMAQueue otherQueue;
#else
    PM4Queue otherQueue;
#endif

    ASSERT_SUCCESS_GPU(pm4Queue.Create(gpuNode), gpuNode);
    ASSERT_SUCCESS_GPU(otherQueue.Create(gpuNode), gpuNode);

    /* Place PM4 packet in the queue, but don't submit it */
    pm4Queue.PlacePacket(PM4WriteDataPacket(destBuf.As<unsigned int*>(), 0x12345678, 0x87654321));

    HsaQueueResource *qRes = pm4Queue.GetResource();

    if (m_FamilyId < FAMILY_AI) {
        unsigned int pendingWptr = pm4Queue.GetPendingWptr();

#ifdef DOORBELL_WRITE_USE_SDMA
        /* Write the wptr and doorbell update using the GPU's SDMA
         * engine. This should submit the PM4 packet on the first
         * queue.
         */
        otherQueue.PlacePacket(SDMAWriteDataPacket(otherQueue.GetFamilyId(), qRes->Queue_write_ptr,
                                                   pendingWptr));
        otherQueue.PlacePacket(SDMAWriteDataPacket(otherQueue.GetFamilyId(), qRes->Queue_DoorBell,
                                                   pendingWptr));
#else
        /* Write the wptr and doorbell update using WRITE_DATA packets
         * on a second PM4 queue. This should submit the PM4 packet on
         * the first queue.
         */
        otherQueue.PlacePacket(
            PM4ReleaseMemoryPacket(m_FamilyId, true, (HSAuint64)qRes->Queue_write_ptr,
                                   pendingWptr, false));
        otherQueue.PlacePacket(
            PM4ReleaseMemoryPacket(m_FamilyId, true, (HSAuint64)qRes->Queue_DoorBell,
                                   pendingWptr, false));
#endif

        otherQueue.SubmitPacket();
    } else {
        HSAuint64 pendingWptr64 = pm4Queue.GetPendingWptr64();

#ifdef DOORBELL_WRITE_USE_SDMA
        /* Write the wptr and doorbell update using the GPU's SDMA
         * engine. This should submit the PM4 packet on the first
         * queue.
         */
        otherQueue.PlacePacket(SDMAWriteDataPacket(otherQueue.GetFamilyId(), qRes->Queue_write_ptr,
                                                   2, &pendingWptr64));
        otherQueue.PlacePacket(SDMAWriteDataPacket(otherQueue.GetFamilyId(), qRes->Queue_DoorBell,
                                                   2, &pendingWptr64));
#else
        /* Write the 64-bit wptr and doorbell update using RELEASE_MEM
         * packets without IRQs on a second PM4 queue. RELEASE_MEM
         * should perform one atomic 64-bit access. This should submit
         * the PM4 packet on the first queue.
         */
        otherQueue.PlacePacket(
            PM4ReleaseMemoryPacket(m_FamilyId, true, (HSAuint64)qRes->Queue_write_ptr,
                                   pendingWptr64, true));
        otherQueue.PlacePacket(
            PM4ReleaseMemoryPacket(m_FamilyId, true, (HSAuint64)qRes->Queue_DoorBell,
                                   pendingWptr64, true));
#endif

        otherQueue.SubmitPacket();
    }

    /* Check that the PM4 packet has been executed */
    EXPECT_TRUE_GPU(WaitOnValue(destBuf.As<unsigned int *>(), 0x12345678), gpuNode);
    EXPECT_TRUE_GPU(WaitOnValue(destBuf.As<unsigned int *>()+1, 0x87654321), gpuNode);

    EXPECT_SUCCESS_GPU(pm4Queue.Destroy(), gpuNode);
    EXPECT_SUCCESS_GPU(otherQueue.Destroy(), gpuNode);

}

TEST_F(KFDQMTest, GPUDoorbellWrite) {
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(GPUDoorbellWrite));

    TEST_END
}

TEST_F(KFDQMTest, UserQueueBufValidation) {
    TEST_START(TESTPROFILE_RUNALL)

    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    HsaQueueResource QueueResources;
    HsaMemoryBuffer *QueueBuf;
    HSAKMT_STATUS status;

    memset(&QueueResources, 0, sizeof(QueueResources));

    // System memory mapping on GPU
    QueueBuf = new HsaMemoryBuffer(PAGE_SIZE, defaultGPUNode);

    EXPECT_SUCCESS(hsaKmtCreateQueue(defaultGPUNode,
                               HSA_QUEUE_COMPUTE,
                               100,
                               HSA_QUEUE_PRIORITY_NORMAL,
                               QueueBuf->As<unsigned int*>(),
                               PAGE_SIZE,
                               NULL,
                               &QueueResources));
    EXPECT_SUCCESS(hsaKmtDestroyQueue(QueueResources.QueueId));

    // CP Queue creation should fail using wrong ring buffer size
    EXPECT_SUCCESS(!hsaKmtCreateQueue(defaultGPUNode,
                               HSA_QUEUE_COMPUTE,
                               100,
                               HSA_QUEUE_PRIORITY_NORMAL,
                               QueueBuf->As<unsigned int*>(),
                               PAGE_SIZE * 2,
                               NULL,
                               &QueueResources));

    // SDMA queue create should fail using wrong ring buffer size
    EXPECT_SUCCESS(!hsaKmtCreateQueue(defaultGPUNode,
                               HSA_QUEUE_SDMA,
                               100,
                               HSA_QUEUE_PRIORITY_NORMAL,
                               QueueBuf->As<unsigned int*>(),
                               PAGE_SIZE * 2,
                               NULL,
                               &QueueResources));

    // CP queue create should fail using NULL ring buffer
    EXPECT_SUCCESS(!hsaKmtCreateQueue(defaultGPUNode,
                               HSA_QUEUE_COMPUTE,
                               100,
                               HSA_QUEUE_PRIORITY_NORMAL,
                               NULL,
                               PAGE_SIZE,
                               NULL,
                               &QueueResources));

    // SDMA queue create should fail using NULL ring buffer
    EXPECT_SUCCESS(!hsaKmtCreateQueue(defaultGPUNode,
                               HSA_QUEUE_SDMA,
                               100,
                               HSA_QUEUE_PRIORITY_NORMAL,
                               NULL,
                               PAGE_SIZE,
                               NULL,
                               &QueueResources));

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(QueueBuf->As<unsigned int*>()));
    EXPECT_SUCCESS(hsaKmtFreeMemory(QueueBuf->As<unsigned int*>(), PAGE_SIZE));

    //
    // This following negative test will evict user queues, must execute in child process,
    // because parent process is allowed to create queue to run the remaining tests.
    //
    pid_t childPid = fork();

    if (childPid == 0) { /* Child process */
        void *cwsr_addr;
        int exit_code = 1;

        TearDown();
        SetUp();

        // System memory mapping on GPU
        QueueBuf = new HsaMemoryBuffer(PAGE_SIZE, defaultGPUNode);
        memset(&QueueResources, 0, sizeof(QueueResources));

        status = hsaKmtCreateQueue(defaultGPUNode,
                               HSA_QUEUE_COMPUTE,
                               100,
                               HSA_QUEUE_PRIORITY_NORMAL,
                               QueueBuf->As<unsigned int*>(),
                               PAGE_SIZE,
                               NULL,
                               &QueueResources);
        if (status != HSAKMT_STATUS_SUCCESS) {
            LOG() << "create queue failed." << std::endl;
            goto free_exit;
        }

        // Update queue percentage 0 to set queue inactive in order to get queue info CWSR area
        status = hsaKmtUpdateQueue(QueueResources.QueueId, 0, HSA_QUEUE_PRIORITY_NORMAL,
                                     QueueBuf->As<unsigned int*>(), PAGE_SIZE, NULL);
        if (status != HSAKMT_STATUS_SUCCESS) {
            LOG() << "update queue failed." << std::endl;
            goto err_exit;
        }

        HsaQueueInfo QueueInfo;
        status = hsaKmtGetQueueInfo(QueueResources.QueueId, &QueueInfo);
        if (status != HSAKMT_STATUS_SUCCESS) {
            LOG() << "get queue info failed." << std::endl;
            goto err_exit;
        }

        // unmap CWSR buffer will evict queue before queue is destroyed
        cwsr_addr = QueueInfo.UserContextSaveArea;
        munmap(cwsr_addr, PAGE_SIZE);

        // unmap and free queue ring buffer should fail before the queue is destroyed
        status = hsaKmtFreeMemory(QueueBuf->As<unsigned int*>(), PAGE_SIZE);
        if (status == HSAKMT_STATUS_SUCCESS) {
            LOG() << "free queue buf should fail." << std::endl;
            goto err_exit;
        }

        status = hsaKmtUnmapMemoryToGPU(QueueBuf->As<unsigned int*>());
        if (status == HSAKMT_STATUS_SUCCESS) {
            LOG() << "unmap queue buf should fail." << std::endl;
            goto err_exit;
        }

        exit_code = 0;

err_exit:
        status = hsaKmtDestroyQueue(QueueResources.QueueId);
        if (status != HSAKMT_STATUS_SUCCESS) {
            LOG() << "destroy queue failed." << std::endl;
            exit_code = 1;
        }
free_exit:
        status = hsaKmtUnmapMemoryToGPU(QueueBuf->As<unsigned int*>());
        if (status != HSAKMT_STATUS_SUCCESS) {
            LOG() << "unmap queue buf failed." << std::endl;
            exit_code = 1;
        }

        status = hsaKmtFreeMemory(QueueBuf->As<unsigned int*>(), PAGE_SIZE);
        if (status != HSAKMT_STATUS_SUCCESS) {
            LOG() << "free queue buf failed." << std::endl;
            exit_code = 1;
        }

        exit(exit_code);
    } else {
        int childStatus;

        waitpid(childPid, &childStatus, 0);
        EXPECT_EQ(true, WIFEXITED(childStatus));
        EXPECT_EQ(0, WEXITSTATUS(childStatus));
    }

    TEST_END
}
