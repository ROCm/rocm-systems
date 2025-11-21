/*
 * Copyright (C) 2014-2019 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "KFDGWSTest.hpp"
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "Dispatch.hpp"

void KFDGWSTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDGWSTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

static void Allocate(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDGWSTest* pKFDGWSTest = (KFDGWSTest*)pTestParamters->pTestObject;

    HSAuint32 firstGWS;
    PM4Queue queue;
    HsaNodeInfo* m_NodeInfo = pKFDGWSTest->Get_NodeInfo();
    const HsaNodeProperties *pNodeProperties = m_NodeInfo->GetNodeProperties(gpuNode);

    if (!pNodeProperties || !pNodeProperties->NumGws) {
        LOG() << "Skip test: GPU node doesn't support GWS" << std::endl;
        return;
    }

    ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
    ASSERT_SUCCESS_GPU(hsaKmtAllocQueueGWS(queue.GetResource()->QueueId,
                       pNodeProperties->NumGws,&firstGWS), gpuNode);
    EXPECT_EQ_GPU(0, firstGWS, gpuNode);
    EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);

}
TEST_F(KFDGWSTest, Allocate) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(Allocate));

    TEST_END
}

static void Semaphore(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDGWSTest* pKFDGWSTest = (KFDGWSTest*)pTestParamters->pTestObject;

    HsaNodeInfo* m_NodeInfo = pKFDGWSTest->Get_NodeInfo();
    const HsaNodeProperties *pNodeProperties = m_NodeInfo->GetNodeProperties(gpuNode);

    HSAuint32 firstGWS;
    HSAuint32 numResources = 1;
    PM4Queue queue;

    if (!pNodeProperties || !pNodeProperties->NumGws) {
        LOG() << "Skip test: GPU node doesn't support GWS" << std::endl;
        return;
    }

    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HsaMemoryBuffer buffer(PAGE_SIZE, gpuNode, true, false, false);
    ASSERT_SUCCESS(queue.Create(gpuNode));
    ASSERT_SUCCESS_GPU(hsaKmtAllocQueueGWS(queue.GetResource()->QueueId,
                       pNodeProperties->NumGws,&firstGWS), gpuNode);
    EXPECT_EQ_GPU(0, firstGWS, gpuNode);

    Assembler* m_pAsm;
    m_pAsm = pKFDGWSTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);
    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(GwsInitIsa, isaBuffer.As<char*>()), gpuNode);

    Dispatch dispatch0(isaBuffer);
    buffer.Fill(numResources, 0, 4);
    dispatch0.SetArgs(buffer.As<void*>(), NULL);
    dispatch0.Submit(queue);
    dispatch0.Sync();

    ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(GwsAtomicIncreaseIsa, isaBuffer.As<char*>()),gpuNode);

    Dispatch dispatch(isaBuffer);
    dispatch.SetArgs(buffer.As<void*>(), NULL);
    dispatch.SetDim(1024, 16, 16);

    dispatch.Submit(queue);
    dispatch.Sync();

    EXPECT_EQ_GPU(1024*16*16+1, *buffer.As<uint32_t *>(), gpuNode);
    EXPECT_SUCCESS_GPU(queue.Destroy(),gpuNode);

}

TEST_F(KFDGWSTest, Semaphore) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(Semaphore));

    TEST_END
}

