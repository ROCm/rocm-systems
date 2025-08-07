/*
 * Copyright (C) 2017-2018 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "KFDIPCTest.hpp"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <vector>
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "SDMAQueue.hpp"
#include "SDMAPacket.hpp"

void KFDIPCTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDIPCTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

KFDIPCTest::~KFDIPCTest(void) {
    /* exit() is necessary for the child process. Otherwise when the
     * child process finishes, gtest assumes the test has finished and
     * starts the next test while the parent is still active.
     */
    if (m_ChildPid == 0)
        exit(::testing::UnitTest::GetInstance()->current_test_info()->result()->Failed());
}

/* Import shared Local Memory from parent process. Check for the pattern
 * filled in by the parent process. Then fill a new pattern.
 *
 * Check import handle has same HsaMemFlags as export handle to verify thunk and KFD
 * import export handle ioctl pass HsaMemFlags correctly.
 */
void KFDIPCTest::BasicTestChildProcess(int defaultGPUNode, int *pipefd, HsaMemFlags mflags) {
    /* Open KFD device for child process. This needs to called before
     * any memory definitions
     */
    TearDown();
    SetUp();

    SDMAQueue sdmaQueue;
    HsaSharedMemoryHandle sharedHandleLM;
    HSAuint64 size = PAGE_SIZE, sharedSize;
    HsaMemoryBuffer tempSysBuffer(size, defaultGPUNode, false);
    HSAuint32 *sharedLocalBuffer = NULL;
    HsaMemMapFlags mapFlags = {0};

    /* Read from Pipe the shared Handle. Import shared Local Memory */
    ASSERT_GE(read(pipefd[0], reinterpret_cast<void*>(&sharedHandleLM), sizeof(sharedHandleLM)), 0);

    ASSERT_SUCCESS(hsaKmtRegisterSharedHandle(&sharedHandleLM,
                  reinterpret_cast<void**>(&sharedLocalBuffer), &sharedSize));
    ASSERT_SUCCESS(hsaKmtMapMemoryToGPUNodes(sharedLocalBuffer, sharedSize, NULL,
                  mapFlags, 1, reinterpret_cast<HSAuint32 *>(&defaultGPUNode)));

    /* Check for pattern in the shared Local Memory */
    ASSERT_SUCCESS(sdmaQueue.Create(defaultGPUNode));
    size = size < sharedSize ? size : sharedSize;
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(), tempSysBuffer.As<HSAuint32*>(),
        sharedLocalBuffer, size));
    sdmaQueue.Wait4PacketConsumption();
    EXPECT_TRUE(WaitOnValue(tempSysBuffer.As<HSAuint32*>(), 0xAAAAAAAA));

    /* Fill in the Local Memory with different pattern */
    sdmaQueue.PlaceAndSubmitPacket(SDMAWriteDataPacket(sdmaQueue.GetFamilyId(), sharedLocalBuffer, 0xBBBBBBBB));
    sdmaQueue.Wait4PacketConsumption();

    HsaPointerInfo ptrInfo;
    EXPECT_SUCCESS(hsaKmtQueryPointerInfo(sharedLocalBuffer, &ptrInfo));
    EXPECT_EQ(ptrInfo.Type, HSA_POINTER_REGISTERED_SHARED);
    EXPECT_EQ(ptrInfo.Node, (HSAuint32)defaultGPUNode);
    EXPECT_EQ(ptrInfo.GPUAddress, (HSAuint64)sharedLocalBuffer);
    EXPECT_EQ(ptrInfo.SizeInBytes, sharedSize);
    EXPECT_EQ(ptrInfo.MemFlags.Value, mflags.Value);

    /* Clean up */
    EXPECT_SUCCESS(sdmaQueue.Destroy());
    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(sharedLocalBuffer));
    EXPECT_SUCCESS(hsaKmtDeregisterMemory(sharedLocalBuffer));
}

/* Fill a pattern into Local Memory and share with the child process.
 * Then wait until Child process to exit and check for the new pattern
 * filled in by the child process.
 */

void KFDIPCTest::BasicTestParentProcess(int defaultGPUNode, pid_t cpid, int *pipefd, HsaMemFlags mflags) {
    HSAuint64 size = PAGE_SIZE, sharedSize;
    int status;
    HSAuint64 AlternateVAGPU;
    void *toShareLocalBuffer;
    HsaMemoryBuffer tempSysBuffer(PAGE_SIZE, defaultGPUNode, false);
    SDMAQueue sdmaQueue;
    HsaSharedMemoryHandle sharedHandleLM;
    HsaMemMapFlags mapFlags = {0};

    ASSERT_SUCCESS(hsaKmtAllocMemory(defaultGPUNode, size, mflags, &toShareLocalBuffer));
    /* Fill a Local Buffer with a pattern */
    ASSERT_SUCCESS(hsaKmtMapMemoryToGPUNodes(toShareLocalBuffer, size, &AlternateVAGPU,
                       mapFlags, 1, reinterpret_cast<HSAuint32 *>(&defaultGPUNode)));
    tempSysBuffer.Fill(0xAAAAAAAA);

    /* Copy pattern in Local Memory before sharing it */
    ASSERT_SUCCESS(sdmaQueue.Create(defaultGPUNode));
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(), toShareLocalBuffer,
        tempSysBuffer.As<HSAuint32*>(), size));
    sdmaQueue.Wait4PacketConsumption();

    /* Share it with the child process */
    ASSERT_SUCCESS(hsaKmtShareMemory(toShareLocalBuffer, size, &sharedHandleLM));

    ASSERT_GE(write(pipefd[1], reinterpret_cast<void*>(&sharedHandleLM), sizeof(sharedHandleLM)), 0);

    /* Wait for the child to finish */
    waitpid(cpid, &status, 0);

    EXPECT_EQ(WIFEXITED(status), 1);
    EXPECT_EQ(WEXITSTATUS(status), 0);

    /* Check for the new pattern filled in by child process */
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(), tempSysBuffer.As<HSAuint32*>(),
        toShareLocalBuffer, size));
    sdmaQueue.Wait4PacketConsumption();
    EXPECT_TRUE(WaitOnValue(tempSysBuffer.As<HSAuint32*>(), 0xBBBBBBBB));

    /* Clean up */
    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(toShareLocalBuffer));
    EXPECT_SUCCESS(sdmaQueue.Destroy());
}

/* Test IPC memory.
 * 1. Parent Process [Create/Fill] LocalMemory (LM) --share--> Child Process
 * 2. Child Process import LM and check for the pattern.
 * 3. Child Process fill in a new pattern and quit.
 * 4. Parent Process wait for the Child process to finish and then check for
 * the new pattern in LM
 *
 * IPC support is limited to Local Memory.
 */

TEST_F(KFDIPCTest, BasicTest) {
    TEST_START(TESTPROFILE_RUNALL)

    const std::vector<int>& GpuNodes = m_NodeInfo.GetNodesWithGPU();
    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    int pipefd[2];
    HsaMemFlags mflags = {0};

    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    if (!GetVramSize(defaultGPUNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    /* Test libhsakmt fork() clean up by defining some buffers. These
     * buffers gets duplicated in the child process but not are not valid
     * as it doesn't have proper mapping in GPU. The clean up code in libhsakmt
     * should handle it
     */
    volatile HSAuint32 stackData[1];
    HsaMemoryBuffer tmpSysBuffer(PAGE_SIZE, defaultGPUNode, false);
    HsaMemoryBuffer tmpUserptrBuffer((void *)&stackData[0], sizeof(HSAuint32));

    /* Create Pipes for communicating shared handles */
    ASSERT_EQ(pipe(pipefd), 0);

    /* Create a child process and share the above Local Memory with it */
    mflags.ui32.NonPaged = 1;
    mflags.ui32.CoarseGrain = 1;

    m_ChildPid = fork();
    if (m_ChildPid == 0)
        BasicTestChildProcess(defaultGPUNode, pipefd, mflags); /* Child Process */
    else
        BasicTestParentProcess(defaultGPUNode, m_ChildPid, pipefd, mflags); /* Parent proces */

    /* Code path executed by both parent and child with respective fds */
    close(pipefd[1]);
    close(pipefd[0]);

    TEST_END
}
