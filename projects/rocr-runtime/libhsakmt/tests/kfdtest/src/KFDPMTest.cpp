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

#include "KFDPMTest.hpp"
#include "KFDTestUtil.hpp"
#include "PM4Packet.hpp"
#include "PM4Queue.hpp"
#include "hsakmt/hsakmt.h"

void KFDPMTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDPMTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

TEST_F(KFDPMTest, SuspendWithActiveProcess) {
    TEST_START(TESTPROFILE_RUNALL)

    EXPECT_EQ(true, SuspendAndWakeUp());

    TEST_END
}

TEST_F(KFDPMTest, SuspendWithIdleQueue) {
    TEST_START(TESTPROFILE_RUNALL)

    PM4Queue queue;
    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    ASSERT_SUCCESS(queue.Create(defaultGPUNode));

    EXPECT_EQ(true, SuspendAndWakeUp());

    EXPECT_SUCCESS(queue.Destroy());

    TEST_END
}

TEST_F(KFDPMTest, SuspendWithIdleQueueAfterWork) {
    TEST_START(TESTPROFILE_RUNALL)

    PM4Queue queue;
    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    HsaMemoryBuffer destBuffer(PAGE_SIZE, defaultGPUNode);

    ASSERT_SUCCESS(queue.Create(defaultGPUNode));

    HsaEvent *event;
    ASSERT_SUCCESS(CreateQueueTypeEvent(false, false, defaultGPUNode, &event));

    queue.PlaceAndSubmitPacket(PM4WriteDataPacket(destBuffer.As<unsigned int*>(), 0x1, 0x2));
    queue.Wait4PacketConsumption(event);
    WaitOnValue(&(destBuffer.As<unsigned int*>()[0]), 0x1);
    WaitOnValue(&(destBuffer.As<unsigned int*>()[1]), 0x2);

    destBuffer.Fill(0);

    EXPECT_EQ(true, SuspendAndWakeUp());

    queue.PlaceAndSubmitPacket(PM4WriteDataPacket(&(destBuffer.As<unsigned int*>()[2]), 0x3, 0x4));
    queue.Wait4PacketConsumption(event);

    EXPECT_EQ(destBuffer.As<unsigned int*>()[0], 0);
    EXPECT_EQ(destBuffer.As<unsigned int*>()[1], 0);

    WaitOnValue(&(destBuffer.As<unsigned int*>()[2]), 0x3);
    WaitOnValue(&(destBuffer.As<unsigned int*>()[3]), 0x4);

    hsaKmtDestroyEvent(event);
    EXPECT_SUCCESS(queue.Destroy());

    TEST_END
}

// TODO: Suspend while workload is being executed by a queue
