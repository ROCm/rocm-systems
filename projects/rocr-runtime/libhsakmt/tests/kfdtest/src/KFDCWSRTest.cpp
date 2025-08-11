/*
 * Copyright (C) 2015-2018 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <tuple>
#include "KFDCWSRTest.hpp"
#include "Dispatch.hpp"

void KFDCWSRTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDCWSRTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

static inline uint32_t checkCWSREnabled() {
    uint32_t cwsr_enable = 0;

    fscanf_dec("/sys/module/amdgpu/parameters/cwsr_enable", &cwsr_enable);

    return cwsr_enable;
}

/**
 * KFDCWSRTest.BasicTest
 *
 * This test dispatches the PersistentIterateIsa shader, which continuously increments a vgpr for
 * (num_witems / WAVE_SIZE) waves. While this shader is running, dequeue/requeue requests
 * are sent in a loop to trigger CWSRs.
 *
 * This is a paremeterized test. See the INSTANTIATE_TEST_CASE_P below for an explanation
 * on the parameters.
 *
 * This test defines a CWSR threshold. The shader will continuously loop until inputBuf is
 * filled with the known stop value, which occurs once cwsr_thresh CWSRs have been
 * successfully triggered.
 *
 * 4 parameterized tests are defined:
 *
 * KFDCWSRTest.BasicTest/0
 * KFDCWSRTest.BasicTest/1
 * KFDCWSRTest.BasicTest/2
 * KFDCWSRTest.BasicTest/3
 *
 * 0: 1 work-item, CWSR threshold of 10
 * 1: 256 work-items (multi-wave), CWSR threshold of 50
 * 2: 512 work-items (multi-wave), CWSR threshold of 100
 * 3: 1024 work-items (multi-wave), CWSR threshold of 1000
 */

static void BasicTest(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDCWSRTest* pKFDCWSRTest = (KFDCWSRTest*)pTestParamters->pTestObject;

    const HSAuint32 m_FamilyId = pKFDCWSRTest->GetFamilyIdFromNodeId(gpuNode);

    Assembler* m_pAsm;
    m_pAsm = pKFDCWSRTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    int num_witems = std::get<0>(pKFDCWSRTest->GetParam());
    int cwsr_thresh = std::get<1>(pKFDCWSRTest->GetParam());
    // Increase delay on emulator by this factor.
    const int delayMult = (g_IsEmuMode ? 20 : 1);

    if ((m_FamilyId >= FAMILY_VI) && (checkCWSREnabled())) {
        HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true, false, true);
        ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(PersistentIterateIsa, isaBuffer.As<char*>()), gpuNode);

        unsigned stopval = 0x1234'5678;
        unsigned outval  = 0x8765'4321;

        // 4B per work-item ==> 1 page per 1024 work-items (take ceiling)
        unsigned bufSize = PAGE_SIZE * ((num_witems / 1024) + (num_witems % 1024 != 0));

        HsaMemoryBuffer inputBuf(bufSize, gpuNode, true, false, false);
        HsaMemoryBuffer outputBuf(bufSize, gpuNode, true, false, false);
        unsigned int* input = inputBuf.As<unsigned int*>();
        unsigned int* output = outputBuf.As<unsigned int*>();
        inputBuf.Fill(0);
        outputBuf.Fill(outval);

        PM4Queue queue;
        ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);

        Dispatch dispatch(isaBuffer);
        dispatch.SetArgs(input, output);
        dispatch.SetDim(num_witems, 1, 1);
        dispatch.Submit(queue);

        Delay(5 * delayMult);

        LOG() << "Starting iteration for " << std::dec << num_witems
              << " work items(s) (targeting " << std::dec << cwsr_thresh
              << " CWSRs)" << std::endl;

        for (int num_cwsrs = 0; num_cwsrs < cwsr_thresh; num_cwsrs++) {

            // Send dequeue request
            EXPECT_SUCCESS_GPU(queue.Update(0, BaseQueue::DEFAULT_PRIORITY, false), gpuNode);

            Delay(5 * delayMult);

            // Send requeue request
            EXPECT_SUCCESS_GPU(queue.Update(100, BaseQueue::DEFAULT_PRIORITY, false), gpuNode);

            Delay(50 * delayMult);

            // Check for reg mangling
            for (int i = 0; i < num_witems; i++) {
                EXPECT_EQ_GPU(outval, output[i], gpuNode);
            }
        }

        LOG() << "Successful completion for " << std::dec << num_witems
              << " work item(s) (CWSRs triggered: " << std::dec << cwsr_thresh
              << ")" << std::endl;
        LOG() << "Signalling shader stop..." << std::endl;

        inputBuf.Fill(stopval);

        // Wait for shader to finish or timeout if shader has vm page fault
        EXPECT_EQ_GPU(0, dispatch.SyncWithStatus(180000), gpuNode);
        EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
    } else {
        LOG() << "Skipping test: No CWSR present for family ID 0x" << m_FamilyId << "." << std::endl;
    }

}

TEST_P(KFDCWSRTest, BasicTest) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(BasicTest));

    TEST_END
}

/**
 * Instantiates various KFDCWSRTest.BasicTest parameterizations
 * Tuple Format: (num_witems, cwsr_thresh)
 *
 * num_witems:    Defines the number of work-items.
 * cwsr_thresh:   Defines the number of CWSRs to trigger.
 */
INSTANTIATE_TEST_CASE_P(
    , KFDCWSRTest,
    ::testing::Values(
            std::make_tuple(1, 10),     /* Single Wave Test,  10 CWSR Triggers */
            std::make_tuple(256, 50),   /* Multi Wave Test,   50 CWSR Triggers */
            std::make_tuple(512, 100),  /* Multi Wave Test,  100 CWSR Triggers */
            std::make_tuple(1024, 1000) /* Multi Wave Test, 1000 CWSR Triggers */
    )
);

/**
 * KFDCWSRTest.InterruptRestore
 *
 * This test verifies that CP can preempt an HQD while it is restoring a dispatch.
 * Create queue 1.
 * Start a dispatch on queue 1 which runs indefinitely and fills all CU wave slots.
 * Create queue 2, triggering context save on queue 1.
 * Start a dispatch on queue 2 which runs indefinitely and fills all CU wave slots.
 * Create queue 3, triggering context save and restore on queues 1 and 2.
 * Preempt runlist. One or both queues must interrupt context restore to preempt.
 */

static void InterruptRestore(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDCWSRTest* pKFDCWSRTest = (KFDCWSRTest*)pTestParamters->pTestObject;

    const HSAuint32 m_FamilyId = pKFDCWSRTest->GetFamilyIdFromNodeId(gpuNode);

    Assembler* m_pAsm;
    m_pAsm = pKFDCWSRTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

   if ((m_FamilyId >= FAMILY_VI) && (checkCWSREnabled())) {
        HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

        ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(InfiniteLoopIsa, isaBuffer.As<char*>()), gpuNode);

        PM4Queue queue1, queue2, queue3;

        ASSERT_SUCCESS_GPU(queue1.Create(gpuNode), gpuNode);

        Dispatch *dispatch1, *dispatch2;

        dispatch1 = new Dispatch(isaBuffer);
        dispatch2 = new Dispatch(isaBuffer);

        dispatch1->SetDim(0x10000, 1, 1);
        dispatch2->SetDim(0x10000, 1, 1);

        dispatch1->Submit(queue1);

        ASSERT_SUCCESS_GPU(queue2.Create(gpuNode), gpuNode);

        dispatch2->Submit(queue2);

        // Give waves time to launch.
        Delay(1);

        ASSERT_SUCCESS_GPU(queue3.Create(gpuNode), gpuNode);

        EXPECT_SUCCESS_GPU(queue1.Destroy(), gpuNode);
        EXPECT_SUCCESS_GPU(queue2.Destroy(), gpuNode);
        EXPECT_SUCCESS_GPU(queue3.Destroy(), gpuNode);

        delete dispatch1;
        delete dispatch2;

    } else {
        LOG() << "Skipping test: No CWSR present for family ID 0x" << m_FamilyId << "." << std::endl;
    }
}

TEST_F(KFDCWSRTest, InterruptRestore) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(InterruptRestore));

    TEST_END
}

#define CWSR_WAVE_MEM_TYPE_VGPR		0
#define CWSR_WAVE_MEM_TYPE_SGPR		1
#define CWSR_WAVE_MEM_TYPE_LDS		2
#define CWSR_WAVE_MEM_TYPE_EXEC		3
#define CWSR_WAVE_MEM_TYPE_PKT_LO	4
// Helper for SaveAreCheck test to match seeded wave memory words
// wordAddr auto increments on word search
static int CheckWaveMemBlockWords(int memType, int waveSize, uint32_t **wordAddr, int numWords,
                                  uint32_t mask) {
    int wordCount = 0, searchStart = 0;
    bool wordFound = false;

    if (memType == CWSR_WAVE_MEM_TYPE_PKT_LO)
        mask &= 0x1ffffff; // saved as 25 bits in ttmp8

    for (int i = 0; i < numWords; i++) {
        uint32_t word =  (*wordAddr)[i], lane = word & 0xffff;
        bool match = memType == CWSR_WAVE_MEM_TYPE_VGPR ?
                     (word & mask) == mask && ((i - searchStart) % waveSize == lane) :
                     word == mask;
	
        if (match) {
            wordFound = true;
            wordCount++;
   	} else {
            if (wordFound)
                break;
            searchStart++;
            continue;
	}

    }

    *wordAddr += (searchStart + wordCount);

    return wordCount;
}	

enum SaveAreaTestCase {
    SAVE_AREA_TEST_WAVE_MEM = 0,
    SAVE_AREA_TEST_WAVE_BLOCKS,
    SAVE_AREA_TEST_RAND_MEM_SIZE,
    SAVE_AREA_TEST_RAND_WAVE_BLOCKS,
    SAVE_AREA_TEST_WAVE_MEM_64,
    SAVE_AREA_TEST_WAVE_BLOCKS_64,
    SAVE_AREA_TEST_RAND_MEM_SIZE_64,
    SAVE_AREA_TEST_RAND_WAVE_BLOCKS_64,
    SAVE_AREA_TEST_MAX
};

/**
 * KFDCWSRTest.SaveAreaCheck
 *
 * Shader/Dispatch seeds wave memory with the following:
 * VGPR word = 0xffff00<ll> where ll = lane ID
 * SGPR word = 0x87654321
 * LDS word = 0x12345678
 * EXEC LO = 0x1010101, EXEC_HI = 0x2020202
 * Dispatch Packet ADDR Lo = 0x1abcabc
 *
 * After dispatch, test checks 2 buffers for initialization status.
 * initBuf[waveId] => Shader writes non-zero to each waveId to signal
 *                    wave memory initialization is ready
 * execBuf[laneId] => All waves vector write non-zero to laneId to
 *                    hint EXEC mask after initBuf signaled
 *
 * Test launch 10 CWSRs then crawls through wave memory to check
 * seeded values.
 *
 * After check, test signals shader termination.
 * Shader rechecks reads on initBuf & execBuf as buffer address are
 * saved in v0-v3 & s0-s3.
 *
 * Test is run in both Wave32/64 as well as fixed and random
 * VGPR & LDS allocations.
 * SGPR allocation is fixed in HW.
 */
static void SaveAreaCheck(KFDTEST_PARAMETERS* pTestParamters) {
    int gpuNode = pTestParamters->gpuNode;
    KFDCWSRTest* pKFDCWSRTest = (KFDCWSRTest*)pTestParamters->pTestObject;

    const HSAuint32 m_FamilyId = pKFDCWSRTest->GetFamilyIdFromNodeId(gpuNode);

    Assembler* m_pAsm;
    m_pAsm = pKFDCWSRTest->GetAssemblerFromNodeId(gpuNode);
    ASSERT_NOTNULL_GPU(m_pAsm, gpuNode);

    if ((m_FamilyId >= FAMILY_GFX12) && (checkCWSREnabled())) {
        HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);

        ASSERT_SUCCESS_GPU(m_pAsm->RunAssembleBuf(InitWaveMemoryIsa, isaBuffer.As<char*>()), gpuNode);

        HsaMemoryBuffer execBuf(PAGE_SIZE, gpuNode, true, false, false);
        HsaMemoryBuffer initBuf(PAGE_SIZE * 2, gpuNode, true, false, false);
        unsigned int* execVal = execBuf.As<unsigned int*>();
        unsigned int* initVal = initBuf.As<unsigned int*>();
        uint32_t ldsMask = 0x12345678, sgprMask = 0x87654321, vgprMask = 0xffff0000;
        uint32_t pktLoMask = 0x1abcabc;
        uint32_t vgprMax, vgprBlockSize;
        size_t ldsMax, ldsBlockSize;
        srand((unsigned) time(NULL)); // seed for random allocation tests

        for (int tcase = SAVE_AREA_TEST_WAVE_MEM; tcase < SAVE_AREA_TEST_MAX; tcase++) {

            // reinit memory
            execBuf.Fill(0);
            initBuf.Fill(0);

            PM4Queue queue;
            ASSERT_SUCCESS_GPU(queue.Create(gpuNode), gpuNode);
            Dispatch *dispatch = new Dispatch(isaBuffer);
            if (tcase > SAVE_AREA_TEST_RAND_WAVE_BLOCKS)
                dispatch->SetWave32(false); // Wave64
            dispatch->SetPacketAddrLo(pktLoMask);
            Dispatch::WaveMemInfo waveInfo = dispatch->GetWaveMemInfo();
            int numWaves = 1;
            unsigned int numVgpr = waveInfo.maxNumVgpr;
            size_t ldsSize = waveInfo.maxBytesLds;
            int vgprSteps = ((waveInfo.maxNumVgpr - waveInfo.vgprBlockSize)/
                            waveInfo.vgprBlockSize) + 1;
            int ldsSteps = ((waveInfo.maxBytesLds - waveInfo.ldsBlockSize)/
                           waveInfo.ldsBlockSize) + 1;
            std::string testName = "MAX WAVE MEMORY TEST";

            switch (tcase) {
            case SAVE_AREA_TEST_WAVE_BLOCKS:
            case SAVE_AREA_TEST_WAVE_BLOCKS_64:
                testName = "MULTI-WAVE BLOCKS TEST";
                numWaves = 1024;
                numVgpr = waveInfo.vgprBlockSize * 2;
                ldsSize = waveInfo.ldsBlockSize * 2;
                break;
            case SAVE_AREA_TEST_RAND_MEM_SIZE:
            case SAVE_AREA_TEST_RAND_MEM_SIZE_64:
                testName = "RANDOM MEM SIZE TEST";
                numWaves = 1;
                numVgpr = ((rand() % vgprSteps) * waveInfo.vgprBlockSize) +
                          waveInfo.vgprBlockSize;
                ldsSize = ((rand() % ldsSteps) * waveInfo.ldsBlockSize) +
                          waveInfo.ldsBlockSize;
                break;
            case SAVE_AREA_TEST_RAND_WAVE_BLOCKS:
            case SAVE_AREA_TEST_RAND_WAVE_BLOCKS_64:
                testName = "RANDOM NUM WAVES TEST";
                numWaves = 1 + (rand() % 1024);
                numVgpr = waveInfo.vgprBlockSize * 2;
                ldsSize = waveInfo.ldsBlockSize * 2;
                break;
             default:
                break;
             }

             std::string waveString = waveInfo.isWave32 ? "32" : "64";
             LOG() << testName << " (Wave" << waveString << ")" << std::dec << "\t => Waves: " << numWaves
                   << "\tVGPRs per Wave: " << numVgpr
                   << "\tLDS per Wave(Bytes): " << ldsSize << std::endl;	

             ASSERT_SUCCESS_GPU(dispatch->SetNumVgprs(numVgpr, &vgprMax, &vgprBlockSize), gpuNode);
             ASSERT_SUCCESS_GPU(dispatch->SetLdsSize(ldsSize, &ldsMax, &ldsBlockSize), gpuNode);
             waveInfo = dispatch->GetWaveMemInfo();
             dispatch->SetArgs(execVal, initVal);
             dispatch->SetDim(numWaves, 1, 1);
             dispatch->Submit(queue);

             // Wait for all waves to finish wave memory initialization
             for (int i = 0; i < numWaves; i++) {
                 unsigned int timeoutMsec = 10000, timeIntervalMsec = 50, timeElapsedMsec = 0;
                     while (timeElapsedMsec < timeoutMsec && !initVal[i]) {
                         timeElapsedMsec += timeIntervalMsec;
                         Delay(timeIntervalMsec);
                     }
                     ASSERT_NE_GPU(timeElapsedMsec, timeoutMsec, gpuNode);
             }

             // Infer EXEC mask from per-lane write buffer
             uint32_t waveSize = waveInfo.isWave32 ? 32 : 64;
             uint32_t execLo = 0, execHi = 0;
             for (int i = 0; i < waveSize; i++) {
                 if (execVal[i]) {
                     if (i < 32)
                         execLo |= 1 << i;
                     else
                         execHi |= 1 << i;
                 }
             }

             // Do a bunch of save/restores
             for (int i = 0; i < 10; i++) {
                 EXPECT_SUCCESS_GPU(queue.Update(100, BaseQueue::DEFAULT_PRIORITY, false), gpuNode);
                 EXPECT_SUCCESS_GPU(queue.Update(0, BaseQueue::DEFAULT_PRIORITY, false), gpuNode);
             }

             HsaQueueResource *qResource = queue.GetResource();
             HsaQueueInfo qinfo;
             ASSERT_SUCCESS_GPU(hsaKmtGetQueueInfo(qResource->QueueId, &qinfo), gpuNode);

             const int numWaveWords = (qinfo.SaveAreaHeader->WaveStateSize/numWaves)/4;
             const int numArgGprs = 4, numUserSgprWords = waveInfo.numUserSgpr - numArgGprs;

             // Crawl the context save area to check seeded memory
             for (int i = 0; i < numWaves; i++) {
                 unsigned int vgprWordCount = numArgGprs * waveSize;
                 unsigned int sgprWordCount = numArgGprs;
                 unsigned int execWordCount =0, ldsWordCount = 0, pktLoWordCount = 0;
                 uint32_t *addr = &qinfo.UserContextSaveArea[numWaveWords * i];

                 // VGPRs
                 unsigned int numWords = (waveInfo.numVgpr * waveSize);
                 vgprWordCount += CheckWaveMemBlockWords(CWSR_WAVE_MEM_TYPE_VGPR, waveSize,
                                                         &addr, numWords, vgprMask);

                 // SGPRs
                 numWords = numUserSgprWords;
                 addr += numArgGprs;
                 sgprWordCount += CheckWaveMemBlockWords(CWSR_WAVE_MEM_TYPE_SGPR, waveSize,
                                                         &addr, numWords, sgprMask);

                 // EXEC mask
                 unsigned int totalFoundWords = vgprWordCount + sgprWordCount;
                 numWords = numWaveWords - totalFoundWords - waveInfo.numBytesLds/4;
                 execWordCount = CheckWaveMemBlockWords(CWSR_WAVE_MEM_TYPE_EXEC, waveSize,
                                                        &addr, numWords, execLo);

                 if (execHi)
                     execWordCount += CheckWaveMemBlockWords(CWSR_WAVE_MEM_TYPE_EXEC, waveSize,
                                                             &addr, numWords, execHi);

                 // Dispatch packet ADDR LO
                 totalFoundWords += execWordCount;
                 numWords = numWaveWords - totalFoundWords;
                 pktLoWordCount = CheckWaveMemBlockWords(CWSR_WAVE_MEM_TYPE_PKT_LO, waveSize,
                                                         &addr, numWords, pktLoMask);

                 // LDS
                 if (waveInfo.numBytesLds) {
                     numWords = numWaveWords - vgprWordCount - sgprWordCount;
                     ldsWordCount = CheckWaveMemBlockWords(CWSR_WAVE_MEM_TYPE_LDS, waveSize,
                                                           &addr, numWords, ldsMask);
                 }

                 EXPECT_EQ_GPU(vgprWordCount, waveInfo.numVgpr * waveSize, gpuNode);
                 EXPECT_EQ_GPU(sgprWordCount, waveInfo.numUserSgpr, gpuNode);
                 EXPECT_EQ_GPU(ldsWordCount, waveInfo.numBytesLds/4, gpuNode);
                 EXPECT_EQ_GPU(execWordCount, waveInfo.isWave32 ? 1 : 2, gpuNode);
                 EXPECT_EQ_GPU(pktLoWordCount, 1, gpuNode);
            }

            // Send requeue request
            EXPECT_SUCCESS_GPU(queue.Update(100, BaseQueue::DEFAULT_PRIORITY, false), gpuNode);

            // Trigger shader end (shader implicitly rechecks v0-v3 & s0-s3) and clean up
            execBuf.Fill(0);
            dispatch->Sync();
            EXPECT_SUCCESS_GPU(queue.Destroy(), gpuNode);
            delete dispatch;
        }
    } else {
        LOG() << "Skipping test: No CWSR present for family ID 0x" << m_FamilyId << "." << std::endl;
    }
}

TEST_F(KFDCWSRTest, SaveAreaCheck) {
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(SaveAreaCheck));

    TEST_END
}
