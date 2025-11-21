/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "KFDPCSamplingTest.hpp"
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <numa.h>
#include <vector>
#include "Dispatch.hpp"
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "SDMAQueue.hpp"
#include "SDMAPacket.hpp"
#include "hsakmt/linux/kfd_ioctl.h"

#define N_PROCESSES             (2)     /* Number of processes running in parallel, must be at least 2 */

/* Captures user specified time (seconds) to sleep */
extern unsigned int g_SleepTime;

void KFDPCSamplingTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDPCSamplingTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

TEST_F(KFDPCSamplingTest, BasicTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (hsaKmtPcSamplingSupport() != HSAKMT_STATUS_SUCCESS)
        return;

    HSAuint32 num_sample_info = 0;
    HSAuint32 return_num_sample_info = 0;

    HSAuint32 defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "Failed to get default GPU Node.";

    /* 1. get pc sampling format numbe of entry */
    HSAKMT_STATUS ret = hsaKmtPcSamplingQueryCapabilities(defaultGPUNode, NULL,
                                         num_sample_info, &return_num_sample_info);
    if (ret == HSAKMT_STATUS_NOT_SUPPORTED) {
        LOG() << "Skipping test: This GPU does not support PC Sampling." << std::endl;
        return;
    }
    ASSERT_GE(return_num_sample_info, 1);

    num_sample_info = return_num_sample_info;
    void *info_buf = calloc(num_sample_info, sizeof(HsaPcSamplingInfo));

    ASSERT_SUCCESS(hsaKmtPcSamplingQueryCapabilities(defaultGPUNode, info_buf,
                                         num_sample_info, &return_num_sample_info));

    HsaPcSamplingInfo *samples = (HsaPcSamplingInfo*) info_buf;
    HsaPcSamplingTraceId traceId1, traceId2;

    samples[0].value = 0x100000; /* 1,048,576 usec */

    /* 1. Failed to start uncreated pc sampling ID */
    ASSERT_SUCCESS(!hsaKmtPcSamplingStart(defaultGPUNode, 12345));

    /* 2. Failed to stop uncreated pc sampling ID */
    ASSERT_SUCCESS(!hsaKmtPcSamplingStop(defaultGPUNode, 12345));

    /* 3. Failed to destroy uncreated pc sampling ID */
    ASSERT_SUCCESS(!hsaKmtPcSamplingDestroy(defaultGPUNode, 12345));

    /* 4. create pc sampling */
    ASSERT_SUCCESS(hsaKmtPcSamplingCreate(defaultGPUNode, &samples[0], &traceId1));
    ASSERT_SUCCESS(hsaKmtPcSamplingDestroy(defaultGPUNode, traceId1));

    /* 5. create twice in the same process with pc sampling activated */
    ASSERT_SUCCESS(hsaKmtPcSamplingCreate(defaultGPUNode, &samples[0], &traceId2));
    ASSERT_SUCCESS(hsaKmtPcSamplingStart(defaultGPUNode, traceId2));
          /* Creat and start 2nd session pc sampling */
    ASSERT_SUCCESS(hsaKmtPcSamplingCreate(defaultGPUNode, &samples[0], &traceId1));
    ASSERT_SUCCESS(hsaKmtPcSamplingStart(defaultGPUNode, traceId1));
    sleep(2);
          /* Stop its own pc sampling session, but another session still alive */
    ASSERT_SUCCESS(hsaKmtPcSamplingStop(defaultGPUNode, traceId2));
          /* Destroy its own pc sampling session when it is de-activated */
    ASSERT_SUCCESS(hsaKmtPcSamplingDestroy(defaultGPUNode, traceId2));
    sleep(1);
    ASSERT_SUCCESS(hsaKmtPcSamplingDestroy(defaultGPUNode, traceId1));

    free(info_buf);
    TEST_END
}

struct ThreadParams {
    int test_num;
    HSAuint32 GPUNode;
    HsaPcSamplingInfo *samples;
};

static unsigned int PCSamplingThread(void* p) {
    struct ThreadParams* pArgs = reinterpret_cast<struct ThreadParams*>(p);

    LOG() << "PCSamplingThread #" << pArgs->test_num << " start." << std::endl;
    HsaPcSamplingTraceId traceId;

    EXPECT_SUCCESS(hsaKmtPcSamplingCreate(pArgs->GPUNode, pArgs->samples, &traceId));
    EXPECT_SUCCESS(hsaKmtPcSamplingStart(pArgs->GPUNode, traceId));
    sleep(3);

    LOG() << "PCSamplingThread #" << pArgs->test_num << " stop." << std::endl;
    EXPECT_SUCCESS(hsaKmtPcSamplingStop(pArgs->GPUNode, traceId));
    EXPECT_SUCCESS(hsaKmtPcSamplingDestroy(pArgs->GPUNode, traceId));

    return 0;
}

TEST_F(KFDPCSamplingTest, MultiThreadPcSamplingTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (hsaKmtPcSamplingSupport() != HSAKMT_STATUS_SUCCESS)
        return;

    HSAuint64 threadId[2];
    struct ThreadParams params[2];
    HSAuint32 num_sample_info = 0;
    HSAuint32 return_num_sample_info = 0;

    HSAuint32 defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "Failed to get default GPU Node";

    HSAKMT_STATUS ret = hsaKmtPcSamplingQueryCapabilities(defaultGPUNode, NULL,
                                         num_sample_info, &return_num_sample_info);
    if (ret == HSAKMT_STATUS_NOT_SUPPORTED) {
        LOG() << "Skipping test: This GPU does not support PC Sampling." << std::endl;
        return;
    }
    ASSERT_GE(return_num_sample_info, 1);

    num_sample_info = return_num_sample_info;
    void *info_buf = calloc(num_sample_info, sizeof(HsaPcSamplingInfo));

    ASSERT_SUCCESS(hsaKmtPcSamplingQueryCapabilities(defaultGPUNode, info_buf,
                                         num_sample_info, &return_num_sample_info));
    HsaPcSamplingInfo *samples = (HsaPcSamplingInfo*) info_buf;

    samples[0].value = 0x100000; /* 1,048,576 usec */

    params[0].test_num = 1;
    params[1].test_num = 2;
    params[0].GPUNode = defaultGPUNode;
    params[1].GPUNode = defaultGPUNode;
    params[0].samples = samples;
    params[1].samples = samples;

    ASSERT_EQ(true, StartThread(&PCSamplingThread, &params[0], threadId[0]));
    sleep(1);
    /* start 2nd thread after 1 sec */
    ASSERT_EQ(true, StartThread(&PCSamplingThread, &params[1], threadId[1]));

    WaitForThread(threadId[0]);
    WaitForThread(threadId[1]);

    free(info_buf);

    TEST_END;
}

struct ProcParams {
    std::string test_name;
    HSAuint32 GPUNode;
    HsaPcSamplingInfo *samples;
};

static unsigned int PCSamplingProcRun(void* p) {
    struct ProcParams* pArgs = reinterpret_cast<struct ProcParams*>(p);
    bool process1_flag = !pArgs->test_name.compare("Test process 1 ");
    int start_delay;

    if (process1_flag)
        start_delay = 0;
    else
        start_delay = 1;

    LOG() << "PCSamplingProc <" << pArgs->test_name <<
                 "> starting after 0x" <<  start_delay  << " secs" << std::endl;
    sleep(start_delay);

    HsaPcSamplingTraceId traceId = start_delay;

    EXPECT_SUCCESS(hsaKmtPcSamplingCreate(pArgs->GPUNode, pArgs->samples, &traceId));
    EXPECT_SUCCESS(hsaKmtPcSamplingStart(pArgs->GPUNode, traceId));
    sleep(3);

    LOG() << "PCSamplingProc <" << pArgs->test_name << "> stop" << std::endl;
    EXPECT_SUCCESS(hsaKmtPcSamplingStop(pArgs->GPUNode, traceId));
    EXPECT_SUCCESS(hsaKmtPcSamplingDestroy(pArgs->GPUNode, traceId));
    LOG() << "PCSamplingProc <" << pArgs->test_name << "> done" << std::endl;

    return 0;
}

TEST_F(KFDPCSamplingTest, MultiProcPcSamplingTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (hsaKmtPcSamplingSupport() != HSAKMT_STATUS_SUCCESS)
        return;

    HSAuint32 defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "Failed to get default GPU Node";

    HSAuint32 num_sample_info = 0;
    HSAuint32 return_num_sample_info = 0;
    struct ProcParams params;

    params.GPUNode = defaultGPUNode;

    HSAKMT_STATUS ret = hsaKmtPcSamplingQueryCapabilities(defaultGPUNode, NULL,
                                         num_sample_info, &return_num_sample_info);
    if (ret == HSAKMT_STATUS_NOT_SUPPORTED) {
        LOG() << "Skipping test: This GPU does not support PC Sampling." << std::endl;
        return;
    }
    ASSERT_GE(return_num_sample_info, 1);

    num_sample_info = return_num_sample_info;
    void *info_buf = calloc(num_sample_info, sizeof(HsaPcSamplingInfo));
    ASSERT_SUCCESS(hsaKmtPcSamplingQueryCapabilities(defaultGPUNode, info_buf,
                                         num_sample_info, &return_num_sample_info));

    HsaPcSamplingInfo *samples = (HsaPcSamplingInfo*) info_buf;

    samples[0].value = 0x100000; /* 1,048,576 usec */

    /* Fork the child processes */
    ForkChildProcesses(defaultGPUNode, N_PROCESSES);

    int rn = FindDRMRenderNode(defaultGPUNode);
    if (rn < 0) {
        LOG() << "Skipping test: Could not find render node for default GPU." << std::endl;
        WaitChildProcesses(defaultGPUNode);
        return;
    }

    params.samples = samples;

    int gpuIndex = m_NodeInfo.HsaGPUindexFromGpuNode(defaultGPUNode);
    params.test_name = m_psName[gpuIndex];

    PCSamplingProcRun(&params);

    WaitChildProcesses(defaultGPUNode);

    if (info_buf)
        free(info_buf);
    TEST_END
}

/* Manully run multiple KFDPCSamplingTest.MultiProcPcSamplingTestM */
TEST_F(KFDPCSamplingTest, MultiProcPcSamplingTestM) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (hsaKmtPcSamplingSupport() != HSAKMT_STATUS_SUCCESS)
        return;

    HSAuint32 num_sample_info = 0;
    HSAuint32 return_num_sample_info = 0;

    HSAuint32 defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "Failed to get default GPU Node";

    HSAKMT_STATUS ret = hsaKmtPcSamplingQueryCapabilities(defaultGPUNode, NULL,
                                         num_sample_info, &return_num_sample_info);
    if (ret == HSAKMT_STATUS_NOT_SUPPORTED) {
        LOG() << "Skipping test: This GPU does not support PC Sampling." << std::endl;
        return;
    }
    ASSERT_GE(return_num_sample_info, 1);

    num_sample_info = return_num_sample_info;
    void *info_buf = calloc(num_sample_info, sizeof(HsaPcSamplingInfo));
    ASSERT_SUCCESS(hsaKmtPcSamplingQueryCapabilities(defaultGPUNode, info_buf,
                                         num_sample_info, &return_num_sample_info));

    HsaPcSamplingInfo *samples = (HsaPcSamplingInfo*) info_buf;
    HsaPcSamplingTraceId traceId;

    samples[0].value = 0x100000; /* 1,048,576 usec */
    ASSERT_SUCCESS(hsaKmtPcSamplingCreate(defaultGPUNode, &samples[0], &traceId));

    ASSERT_SUCCESS(hsaKmtPcSamplingStart(defaultGPUNode, traceId));
    sleep(3);
    ASSERT_SUCCESS(hsaKmtPcSamplingStop(defaultGPUNode, traceId));
    ASSERT_SUCCESS(hsaKmtPcSamplingDestroy(defaultGPUNode, traceId));

    free(info_buf);
    TEST_END
}
