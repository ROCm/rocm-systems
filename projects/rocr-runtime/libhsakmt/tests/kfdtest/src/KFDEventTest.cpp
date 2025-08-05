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

#include <math.h>
#include <limits.h>

#include "KFDEventTest.hpp"
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"


void KFDEventTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    for (int i = 0; i < MAX_GPU; i++)
        m_pHsaEventGPU[i] = NULL;

    ROUTINE_END
}

void KFDEventTest::TearDown() {
    ROUTINE_START

    // Not all tests create an event, destroy only if there is one
    for (int i = 0; i < MAX_GPU; i++) {
        if (m_pHsaEventGPU[i] != NULL) {
            // hsaKmtDestroyEvent moved to TearDown to make sure it is being called
            EXPECT_SUCCESS(hsaKmtDestroyEvent(m_pHsaEventGPU[i]));
        }
    }

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

static void CreateDestroyEvent(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDEventTest* pKFDEventTest = (KFDEventTest*)pTestParamters->pTestObject;

    int gpuIndex = pKFDEventTest->Get_NodeInfo()->HsaGPUindexFromGpuNode(gpuNode);
    HsaEvent* m_pHsaEvent = pKFDEventTest->m_pHsaEventGPU[gpuIndex];

    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &m_pHsaEvent), gpuNode);
    EXPECT_NE_GPU(0, m_pHsaEvent->EventData.HWData2, gpuNode);

}

TEST_F(KFDEventTest, CreateDestroyEvent) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(CreateDestroyEvent));

    // Destroy event is being called in test TearDown
    TEST_END;
}


static void CreateMaxEvents(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDEventTest* pKFDEventTest = (KFDEventTest*)pTestParamters->pTestObject;

    static const unsigned int MAX_EVENT_NUMBER = 256;

    HsaEvent* pHsaEvent[MAX_EVENT_NUMBER];

    unsigned int i = 0;

    for (i = 0; i < MAX_EVENT_NUMBER; i++) {
        pHsaEvent[i] = NULL;
        ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &pHsaEvent[i]), gpuNode);
    }

    for (i = 0; i < MAX_EVENT_NUMBER; i++) {
        EXPECT_SUCCESS_GPU(hsaKmtDestroyEvent(pHsaEvent[i]), gpuNode);
    }
}

TEST_F(KFDEventTest, CreateMaxEvents) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(CreateMaxEvents));

    TEST_END;
}

static void SignalEvent(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDEventTest* pKFDEventTest = (KFDEventTest*)pTestParamters->pTestObject;

    int gpuIndex = pKFDEventTest->Get_NodeInfo()->HsaGPUindexFromGpuNode(gpuNode);
    HsaEvent* m_pHsaEvent = pKFDEventTest->m_pHsaEventGPU[gpuIndex];
    HSAuint32 m_FamilyId = pKFDEventTest->GetFamilyIdFromNodeId(gpuNode);

    PM4Queue queue;
    HsaEvent *tmp_event;

    ASSERT_SUCCESS(CreateQueueTypeEvent(false, false, gpuNode, &tmp_event));

    /* Intentionally let event id for m_pHsaEvent be non zero */
    ASSERT_SUCCESS(CreateQueueTypeEvent(false, false, gpuNode, &m_pHsaEvent));
    ASSERT_NE(0, m_pHsaEvent->EventData.HWData2);

    ASSERT_SUCCESS(queue.Create(gpuNode));

    /* From gfx9 onward, m_pHsaEvent->EventId will also be passed to int_ctxid in
     * the Release Mem packet, which is used as context id in ISR.
     */
    queue.PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, false,
                    m_pHsaEvent->EventData.HWData2, m_pHsaEvent->EventId));

    queue.Wait4PacketConsumption();

    EXPECT_SUCCESS_GPU(hsaKmtWaitOnEvent(m_pHsaEvent, g_TestTimeOut), gpuNode);

    EXPECT_SUCCESS_GPU(hsaKmtDestroyEvent(tmp_event), gpuNode);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
}

TEST_F(KFDEventTest, SignalEvent) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(SignalEvent));

    TEST_END;
}

/* test event signaling with event age enabled wait */
static void SignalEventExt(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDEventTest* pKFDEventTest = (KFDEventTest*)pTestParamters->pTestObject;

    int gpuIndex = pKFDEventTest->Get_NodeInfo()->HsaGPUindexFromGpuNode(gpuNode);
    HsaEvent* m_pHsaEvent = pKFDEventTest->m_pHsaEventGPU[gpuIndex];
    HSAuint32 m_FamilyId = pKFDEventTest->GetFamilyIdFromNodeId(gpuNode);

    PM4Queue queue;
    HsaEvent *tmp_event;
    uint64_t event_age;

    if (pKFDEventTest->Get_Version()->KernelInterfaceMajorVersion == 1 &&
        pKFDEventTest->Get_Version()->KernelInterfaceMinorVersion < 14) {
        LOG() << "event age tracking isn't supported in KFD. Exiting." << std::endl;
        return;
    }

    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &tmp_event), gpuNode);

    /* Intentionally let event id for m_pHsaEvent be non zero */
    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &m_pHsaEvent), gpuNode);
    ASSERT_NE_GPU(0, m_pHsaEvent->EventData.HWData2, gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    /* 1. event_age gets incremented every time when the event signals */
    event_age = 1;
    queue.PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, false,
                    m_pHsaEvent->EventData.HWData2, m_pHsaEvent->EventId));
    EXPECT_SUCCESS_GPU(hsaKmtWaitOnEvent_Ext(m_pHsaEvent, g_TestTimeOut, &event_age), gpuNode);
    ASSERT_EQ_GPU(event_age, 2, gpuNode);
    queue.PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, false,
                    m_pHsaEvent->EventData.HWData2, m_pHsaEvent->EventId));
    EXPECT_SUCCESS_GPU(hsaKmtWaitOnEvent_Ext(m_pHsaEvent, g_TestTimeOut, &event_age), gpuNode);
    ASSERT_EQ_GPU(event_age, 3, gpuNode);

    /* 2. event wait return without sleep after the event signals */
    queue.PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, false,
                    m_pHsaEvent->EventData.HWData2, m_pHsaEvent->EventId));
    sleep(1); /* wait for event signaling */
    EXPECT_SUCCESS_GPU(hsaKmtWaitOnEvent_Ext(m_pHsaEvent, g_TestTimeOut, &event_age), gpuNode);
    ASSERT_EQ_GPU(event_age, 4, gpuNode);

    /* 3. signaling from CPU */
    hsaKmtSetEvent(m_pHsaEvent);
    EXPECT_SUCCESS_GPU(hsaKmtWaitOnEvent_Ext(m_pHsaEvent, g_TestTimeOut, &event_age), gpuNode);
    ASSERT_EQ_GPU(event_age, 5, gpuNode);

    /* 4. when event_age is 0, hsaKmtWaitOnEvent_Ext always sleeps */
    event_age = 0;
    ASSERT_EQ_GPU(HSAKMT_STATUS_WAIT_TIMEOUT, hsaKmtWaitOnEvent_Ext(m_pHsaEvent, g_TestTimeOut, &event_age), gpuNode);

    /* 5. when event_age is 0, it always stays 0 after the event signals */
    queue.PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, false,
                    m_pHsaEvent->EventData.HWData2, m_pHsaEvent->EventId));
    EXPECT_SUCCESS(hsaKmtWaitOnEvent_Ext(m_pHsaEvent, g_TestTimeOut, &event_age));
    ASSERT_EQ(event_age, 0);

    EXPECT_SUCCESS(hsaKmtDestroyEvent(tmp_event));

    EXPECT_SUCCESS(queue.Destroy());

}

TEST_F(KFDEventTest, SignalEventExt) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(SignalEventExt));

    TEST_END;
}

static uint64_t gettime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((int64_t)ts.tv_sec) * 1000 * 1000 * 1000 + ts.tv_nsec;
}

static inline double pow2_round_up(int num) {
    return pow(2, ceil(log(num)/log(2)));
}

class QueueAndSignalBenchmark {
 private:
    static const int HISTORY_SIZE = 100;

    int mNumEvents;
    int mHistorySlot;
    uint64_t mTimeHistory[HISTORY_SIZE];
    uint64_t mLatHistory[HISTORY_SIZE];

 public:
    QueueAndSignalBenchmark(int events) : mNumEvents(events), mHistorySlot(0) {
        memset(mTimeHistory, 0, sizeof(mTimeHistory));
        memset(mLatHistory, 0, sizeof(mLatHistory));
    }

    int queueAndSignalEvents(int node, int eventCount, uint64_t &time, uint64_t &latency) {
        int r;
        uint64_t startTime;
        PM4Queue queue;

        unsigned int familyId = g_baseTest->GetFamilyIdFromNodeId(node);
        HsaEvent** pHsaEvent = reinterpret_cast<HsaEvent**>(calloc(eventCount, sizeof(HsaEvent*)));
        size_t packetSize = PM4ReleaseMemoryPacket(familyId, false, 0, 0).SizeInBytes();
        int qSize = fmax(PAGE_SIZE, pow2_round_up(packetSize*eventCount + 1));

        time = 0;

        r = queue.Create(node, qSize);
        if (r != HSAKMT_STATUS_SUCCESS)
            goto exit;

        for (int i = 0; i < eventCount; i++) {
            r = CreateQueueTypeEvent(false, false, node, &pHsaEvent[i]);
            if (r != HSAKMT_STATUS_SUCCESS)
                goto exit;

            queue.PlacePacket(PM4ReleaseMemoryPacket(familyId, false, pHsaEvent[i]->EventData.HWData2, pHsaEvent[i]->EventId));
        }

        startTime = gettime();
        queue.SubmitPacket();
        for (int i = 0; i < eventCount; i++) {
            r = hsaKmtWaitOnEvent(pHsaEvent[i], g_TestTimeOut);

            if (r != HSAKMT_STATUS_SUCCESS)
                goto exit;

            if (i == 0)
                latency = gettime() - startTime;
        }
        time = gettime() - startTime;

exit:
        for (int i = 0; i < eventCount; i++) {
            if (pHsaEvent[i])
                hsaKmtDestroyEvent(pHsaEvent[i]);
        }
        queue.Destroy();

        return r;
    }

    void run(int node) {
        int r = 0;
        uint64_t time = 0, latency = 0;
        uint64_t avgLat = 0, avgTime = 0;
        uint64_t minTime = ULONG_MAX, maxTime = 0;
        uint64_t minLat = ULONG_MAX, maxLat = 0;

        ASSERT_EQ(queueAndSignalEvents(node, mNumEvents, time, latency), HSAKMT_STATUS_SUCCESS);

        mTimeHistory[mHistorySlot%HISTORY_SIZE] = time;
        mLatHistory[mHistorySlot%HISTORY_SIZE] = latency;

        for (int i = 0; i < HISTORY_SIZE; i++) {
            minTime = mTimeHistory[i] < minTime ? mTimeHistory[i] : minTime;
            maxTime = mTimeHistory[i] > maxTime ? mTimeHistory[i] : maxTime;
            avgTime += mTimeHistory[i];

            minLat = mLatHistory[i] < minLat ? mLatHistory[i] : minLat;
            maxLat = mLatHistory[i] > maxLat ? mLatHistory[i] : maxLat;
            avgLat += mLatHistory[i];
        }

        avgTime /= HISTORY_SIZE;
        avgLat /= HISTORY_SIZE;
        mHistorySlot++;

        printf("\033[KEvents: %d History: %d/%d\n", mNumEvents, mHistorySlot, HISTORY_SIZE);
        printf("\033[KMin Latency: %f ms\n", (float)minLat/1000000);
        printf("\033[KMax Latency: %f ms\n", (float)maxLat/1000000);
        printf("\033[KAvg Latency: %f ms\n", (float)avgLat/1000000);
        printf("\033[K   Min Rate: %f IH/ms\n", ((float)mNumEvents)/maxTime*1000000);
        printf("\033[K   Max Rate: %f IH/ms\n", ((float)mNumEvents)/minTime*1000000);
        printf("\033[K   Avg Rate: %f IH/ms\n", ((float)mNumEvents)/avgTime*1000000);
    }
};

TEST_F(KFDEventTest, DISABLED_MeasureInterruptConsumption) {
    TEST_START(TESTPROFILE_RUNALL);
    QueueAndSignalBenchmark latencyBench(128);
    QueueAndSignalBenchmark sustainedBench(4095);

    printf("\033[2J");
    while (true) {
        printf("\033[H");
        printf("--------------------------\n");
        latencyBench.run(m_NodeInfo.HsaDefaultGPUNode());
        printf("--------------------------\n");
        sustainedBench.run(m_NodeInfo.HsaDefaultGPUNode());
        printf("--------------------------\n");
    }

    TEST_END;
}

static void SignalMaxEvents(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDEventTest* pKFDEventTest = (KFDEventTest*)pTestParamters->pTestObject;

    static const unsigned int MAX_EVENT_NUMBER = 4095;
    uint64_t time, latency;

    QueueAndSignalBenchmark maxEventTest(MAX_EVENT_NUMBER);
    maxEventTest.queueAndSignalEvents(gpuNode, MAX_EVENT_NUMBER,
            time, latency);
}

TEST_F(KFDEventTest, SignalMaxEvents) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(SignalMaxEvents));

    TEST_END;
}

static void SignalMultipleEventsWaitForAll(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDEventTest* pKFDEventTest = (KFDEventTest*)pTestParamters->pTestObject;

    int gpuIndex = pKFDEventTest->Get_NodeInfo()->HsaGPUindexFromGpuNode(gpuNode);
    HSAuint32 m_FamilyId = pKFDEventTest->GetFamilyIdFromNodeId(gpuNode);

    static const unsigned int EVENT_NUMBER = 64;  // 64 is the maximum for hsaKmtWaitOnMultipleEvents
    static const unsigned int WAIT_BETWEEN_SUBMISSIONS_MS = 50;

    HsaEvent* pHsaEvent[EVENT_NUMBER];
    unsigned int i = 0;

    for (i = 0; i < EVENT_NUMBER; i++) {
        pHsaEvent[i] = NULL;
        ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &pHsaEvent[i]), gpuNode);
    }

    PM4Queue queue;

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    unsigned int pktSizeDwords = 0;
    for (i = 0; i < EVENT_NUMBER; i++) {
        queue.PlaceAndSubmitPacket(PM4ReleaseMemoryPacket(m_FamilyId, false, pHsaEvent[i]->EventData.HWData2,
                                   pHsaEvent[i]->EventId));
        queue.Wait4PacketConsumption();

        Delay(WAIT_BETWEEN_SUBMISSIONS_MS);
    }

    EXPECT_SUCCESS_GPU(hsaKmtWaitOnMultipleEvents(pHsaEvent, EVENT_NUMBER, true, g_TestTimeOut), gpuNode);

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

    for (i = 0; i < EVENT_NUMBER; i++)
        EXPECT_SUCCESS_GPU(hsaKmtDestroyEvent(pHsaEvent[i]), gpuNode);
}

TEST_F(KFDEventTest, SignalMultipleEventsWaitForAll) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(SignalMultipleEventsWaitForAll));

    TEST_END;
}

/* Send an event interrupt with 0 context ID. Test that KFD handles it
 * gracefully and with good performance. On current GPUs and firmware it
 * should be handled on a fast path.
 */
static void SignalInvalidEvent(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDEventTest* pKFDEventTest = (KFDEventTest*)pTestParamters->pTestObject;

    int gpuIndex = pKFDEventTest->Get_NodeInfo()->HsaGPUindexFromGpuNode(gpuNode);
    HsaEvent* m_pHsaEvent = pKFDEventTest->m_pHsaEventGPU[gpuIndex];
    HSAuint32 m_FamilyId = pKFDEventTest->GetFamilyIdFromNodeId(gpuNode);

    PM4Queue queue;

    // Create some dummy events, to make the slow path a bit slower
    static const unsigned int EVENT_NUMBER = 64;//4094;
    HsaEvent* pHsaEvent[EVENT_NUMBER];
    for (int i = 0; i < EVENT_NUMBER; i++) {
        pHsaEvent[i] = NULL;
        ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &pHsaEvent[i]), gpuNode);
    }

    ASSERT_SUCCESS_GPU(CreateQueueTypeEvent(false, false, gpuNode, &m_pHsaEvent), gpuNode);
    ASSERT_NE_GPU(0, m_pHsaEvent->EventData.HWData2, gpuNode);

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

    static const unsigned int REPS = 2000;
    HSAuint64 duration[REPS];
    HSAuint64 total = 0, min = 1000000, max = 0;
    for (int i = 0; i < REPS; i++) {
        // Invalid signal packet
        queue.PlacePacket(PM4ReleaseMemoryPacket(m_FamilyId, false, 0, 0));
        // Submit valid signal packet
        queue.PlacePacket(PM4ReleaseMemoryPacket(m_FamilyId, false,
                        m_pHsaEvent->EventData.HWData2, m_pHsaEvent->EventId));

        HSAuint64 startTime = GetSystemTickCountInMicroSec();
        queue.SubmitPacket();

        EXPECT_SUCCESS_GPU(hsaKmtWaitOnEvent(m_pHsaEvent, g_TestTimeOut), gpuNode);

        duration[i] = GetSystemTickCountInMicroSec() - startTime;
        total += duration[i];
        if (duration[i] < min)
            min = duration[i];
        if (duration[i] > max)
            max = duration[i];
    }

    double mean = (double)(total - min - max) / (REPS - 2);
    double variance = 0;
    bool skippedMin = false, skippedMax = false;
    HSAuint64 newMin = max, newMax = min;
    for (int i = 0; i < REPS; i++) {
        if (!skippedMin && duration[i] == min) {
            skippedMin = true;
            continue;
        }
        if (!skippedMax && duration[i] == max) {
            skippedMax = true;
            continue;
        }
        if (duration[i] < newMin)
            newMin = duration[i];
        if (duration[i] > newMax)
            newMax = duration[i];
        double diff = mean - duration[i];
        variance += diff*diff;
    }
    variance /= REPS - 2;
    double stdDev = sqrt(variance);

    LOG() << "Time for event handling (min/avg/max [std.dev] in us) " << std::dec
          << newMin << "/" << mean << "/" << newMax << " [" << stdDev << "]\n";

    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

    for (int i = 0; i < EVENT_NUMBER; i++)
        EXPECT_SUCCESS_GPU(hsaKmtDestroyEvent(pHsaEvent[i]), gpuNode);

}

TEST_F(KFDEventTest, SignalInvalidEvent) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(SignalInvalidEvent));

    TEST_END;
}
