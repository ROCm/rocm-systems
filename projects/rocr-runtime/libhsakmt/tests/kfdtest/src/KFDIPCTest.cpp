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

#define SV_SOCK_PATH "/tmp/kfdipctest_socket"
#define BUF_SIZE 10
#define BACKLOG 10

void KFDIPCTest::SendDmabufFDOverSocket(int dmabuf_fd) {
    struct sockaddr_un addr;

    int sfd = socket(AF_UNIX, SOCK_STREAM, 0);

    remove(SV_SOCK_PATH);
    memset(&addr, 0, sizeof(struct sockaddr_un));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SV_SOCK_PATH, sizeof(addr.sun_path) - 1);

    bind(sfd, (struct sockaddr *) &addr, sizeof(struct sockaddr_un));

    listen(sfd, BACKLOG);

    int cfd = accept(sfd, NULL, NULL);

    char iov_buf[1];
    struct msghdr msg = {0};
    char buf[CMSG_SPACE(sizeof(dmabuf_fd))];

    memset(buf, 0, sizeof(buf));
    memset(iov_buf, 0, sizeof(iov_buf));
    iov_buf[0] = 'y';

    struct iovec io = {.iov_base = iov_buf, .iov_len = 1};

    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(dmabuf_fd));

    memcpy(CMSG_DATA(cmsg), &dmabuf_fd, sizeof(dmabuf_fd));

    msg.msg_controllen = CMSG_SPACE(sizeof(dmabuf_fd));

    size_t sent = sendmsg(cfd, &msg, 0);
}

int KFDIPCTest::RecvDmabufFDOverSocket(void) {
    struct sockaddr_un addr;
    int dambuf_fd;

    Delay(200);

    int sfd = socket(AF_UNIX, SOCK_STREAM, 0);

    memset(&addr, 0, sizeof(struct sockaddr_un));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SV_SOCK_PATH, sizeof(addr.sun_path) - 1);

    connect(sfd, (struct sockaddr *) &addr, sizeof(struct sockaddr_un));

    struct msghdr msg = {0};

    char m_buffer[1];
    struct iovec io = {.iov_base = m_buffer, .iov_len = sizeof(m_buffer)};
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;

    char c_buffer[256];
    msg.msg_control = c_buffer;
    msg.msg_controllen = sizeof(c_buffer);

    size_t rcv = recvmsg(sfd, &msg, MSG_WAITALL);
    if (rcv < 0) return -1;

    while (!rcv)
    rcv = recvmsg(sfd, &msg, MSG_WAITALL);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);

    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));

    return fd;

}

#define DMABUF_TEST_BUFFER_SIZE PAGE_SIZE * 10

void KFDIPCTest::DmabufTestChildProcess(int defaultGPUNode, int *pipefd, int delayPlace) {
    TearDown();
    SetUp();

    int dmabuf_fd;
    struct amdgpu_bo_import_result import;
    int rn = FindDRMRenderNode(defaultGPUNode);
    amdgpu_bo_handle handle;
    HsaMemoryBuffer tempSysBuffer(DMABUF_TEST_BUFFER_SIZE, defaultGPUNode, false);
    SDMAQueue sdmaQueue, sdmaQueue2;
    void *localBuffer;

    ASSERT_GE(read(pipefd[0], reinterpret_cast<void*>(&dmabuf_fd), sizeof(dmabuf_fd)), 0);
    dmabuf_fd = RecvDmabufFDOverSocket();

    if (delayPlace == 1)
        Delay(10 * 1000);
    import.alloc_size = DMABUF_TEST_BUFFER_SIZE;
    amdgpu_bo_import(m_RenderNodes[rn].device_handle, amdgpu_bo_handle_type_dma_buf_fd, dmabuf_fd, &import);

    close(dmabuf_fd);

    handle = import.buf_handle;
    amdgpu_bo_cpu_map(handle, &localBuffer);

    amdgpu_bo_va_op(handle, 0, DMABUF_TEST_BUFFER_SIZE,
                            reinterpret_cast<uint64_t>(localBuffer), 0, AMDGPU_VA_OP_MAP);

    if (delayPlace == 0)
        Delay(10 * 1000);

    ASSERT_SUCCESS(sdmaQueue.Create(defaultGPUNode));
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(), tempSysBuffer.As<HSAuint32*>(),
        localBuffer, DMABUF_TEST_BUFFER_SIZE));
    sdmaQueue.Wait4PacketConsumption();
    EXPECT_TRUE(WaitOnValue(tempSysBuffer.As<HSAuint32*>(), 0xAAAAAAAA));
    EXPECT_SUCCESS(sdmaQueue.Destroy());

    LOG() << "Child Local buffer: " << (void const *)localBuffer << std::endl;
    LOG() << "Child Temp buffer: " << (void const *)(tempSysBuffer.As<HSAuint32*>()) << std::endl;

    ASSERT_SUCCESS(sdmaQueue2.Create(defaultGPUNode));
    sdmaQueue2.PlaceAndSubmitPacket(SDMAWriteDataPacket(sdmaQueue2.GetFamilyId(), localBuffer, 0xBBBBBBBB));
    sdmaQueue2.Wait4PacketConsumption();
    EXPECT_SUCCESS(sdmaQueue2.Destroy());

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(localBuffer));
    EXPECT_SUCCESS(hsaKmtDeregisterMemory(localBuffer));

}

void KFDIPCTest::DmabufTestParentProcess(int defaultGPUNode, pid_t cpid, int *pipefd, int delayPlace) {
    void *localBuffer;
    HsaMemFlags mflags = {0};
    int dmabuf_fd;
    HSAuint64 offset;
    int status;
    HsaMemoryBuffer tempSysBuffer(DMABUF_TEST_BUFFER_SIZE, defaultGPUNode, false);
    SDMAQueue sdmaQueue, sdmaQueue2;
    HSAuint64 AlternateVAGPU;
    HsaMemMapFlags mapFlags = {0};

    mflags.ui32.NonPaged = 1;
    mflags.ui32.CoarseGrain = 1;

    ASSERT_SUCCESS(hsaKmtAllocMemory(defaultGPUNode, DMABUF_TEST_BUFFER_SIZE, mflags, &localBuffer));
    ASSERT_SUCCESS(hsaKmtMapMemoryToGPUNodes(localBuffer, DMABUF_TEST_BUFFER_SIZE, &AlternateVAGPU,
                       mapFlags, 1, reinterpret_cast<HSAuint32 *>(&defaultGPUNode)));
    tempSysBuffer.Fill(0xAAAAAAAA);

    LOG() << "Parent Local buffer: " << (void const *)localBuffer << std::endl;
    LOG() << "Parent Temp buffer: " << (void const *)(tempSysBuffer.As<HSAuint32*>()) << std::endl;

    /* Copy pattern in Local Memory before sharing it */
    ASSERT_SUCCESS(sdmaQueue.Create(defaultGPUNode));
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(), localBuffer,
        tempSysBuffer.As<HSAuint32*>(), DMABUF_TEST_BUFFER_SIZE));
    sdmaQueue.Wait4PacketConsumption();

    hsaKmtExportDMABufHandle(localBuffer, DMABUF_TEST_BUFFER_SIZE, &dmabuf_fd, &offset);

    ASSERT_GE(write(pipefd[1], reinterpret_cast<void*>(&dmabuf_fd), sizeof(dmabuf_fd)), 0);

    SendDmabufFDOverSocket(dmabuf_fd);
    close(dmabuf_fd);
    EXPECT_SUCCESS(sdmaQueue.Destroy());

    /* Wait for the child to finish */
    waitpid(cpid, &status, 0);

    LOG() << "Parent finished waiting" << std::endl;

    ASSERT_SUCCESS(sdmaQueue2.Create(defaultGPUNode));
    sdmaQueue2.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue2.GetFamilyId(), tempSysBuffer.As<HSAuint32*>(),
        localBuffer, DMABUF_TEST_BUFFER_SIZE));
    sdmaQueue2.Wait4PacketConsumption();
    EXPECT_TRUE(WaitOnValue(tempSysBuffer.As<HSAuint32*>(), 0xBBBBBBBB));
    EXPECT_SUCCESS(sdmaQueue2.Destroy());

    LOG() << "Parent found BBBBBBBB" << std::endl;

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(localBuffer));
}

TEST_F(KFDIPCTest, DmabufTest) {
    TEST_START(TESTPROFILE_RUNALL)

    const std::vector<int>& GpuNodes = m_NodeInfo.GetNodesWithGPU();
    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    int pipefd[2];

    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    if (!GetVramSize(defaultGPUNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    /* Create Pipes for communicating shared handles */
    ASSERT_EQ(pipe(pipefd), 0);

    m_ChildPid = fork();
    if (m_ChildPid == 0)
        DmabufTestChildProcess(defaultGPUNode, pipefd, 0); /* Child Process */
    else
        DmabufTestParentProcess(defaultGPUNode, m_ChildPid, pipefd, 0); /* Parent proces */

    /* Code path executed by both parent and child with respective fds */
    close(pipefd[1]);
    close(pipefd[0]);

    TEST_END
}

TEST_F(KFDIPCTest, DmabufFdTest) {
    TEST_START(TESTPROFILE_RUNALL)

    const std::vector<int>& GpuNodes = m_NodeInfo.GetNodesWithGPU();
    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    int pipefd[2];

    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    if (!GetVramSize(defaultGPUNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    /* Create Pipes for communicating shared handles */
    ASSERT_EQ(pipe(pipefd), 0);

    m_ChildPid = fork();
    if (m_ChildPid == 0)
        DmabufTestChildProcess(defaultGPUNode, pipefd, 1); /* Child Process */
    else
        DmabufTestParentProcess(defaultGPUNode, m_ChildPid, pipefd, 1); /* Parent proces */

    /* Code path executed by both parent and child with respective fds */
    close(pipefd[1]);
    close(pipefd[0]);

    TEST_END
}


void KFDIPCTest::ParentImportTestParentProcess(int defaultGPUNode, int cpid, int *pipefd) {
    int dmabuf_fd;
    struct amdgpu_bo_import_result import;
    int rn = FindDRMRenderNode(defaultGPUNode);
    amdgpu_bo_handle handle;
    int status;
    HsaMemoryBuffer tempSysBuffer(PAGE_SIZE, defaultGPUNode, false);
    SDMAQueue sdmaQueue, sdmaQueue2;
    void *localBuffer;

    ASSERT_GE(read(pipefd[0], reinterpret_cast<void*>(&dmabuf_fd), sizeof(dmabuf_fd)), 0);
    dmabuf_fd = RecvDmabufFDOverSocket();

    import.alloc_size = PAGE_SIZE;
    amdgpu_bo_import(m_RenderNodes[rn].device_handle, amdgpu_bo_handle_type_dma_buf_fd, dmabuf_fd, &import);
    close(dmabuf_fd);

    handle = import.buf_handle;
    amdgpu_bo_cpu_map(handle, &localBuffer);

    amdgpu_bo_va_op(handle, 0, PAGE_SIZE,
                            reinterpret_cast<uint64_t>(localBuffer), 0, AMDGPU_VA_OP_MAP);

    ASSERT_SUCCESS(sdmaQueue.Create(defaultGPUNode));
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(), tempSysBuffer.As<HSAuint32*>(),
        localBuffer, PAGE_SIZE));
    sdmaQueue.Wait4PacketConsumption();
    EXPECT_TRUE(WaitOnValue(tempSysBuffer.As<HSAuint32*>(), 0xAAAAAAAA));
    EXPECT_SUCCESS(sdmaQueue.Destroy());

    LOG() << "Parent Local buffer: " << (void const *)localBuffer << std::endl;
    LOG() << "Parent Temp buffer: " << (void const *)(tempSysBuffer.As<HSAuint32*>()) << std::endl;

    ASSERT_SUCCESS(sdmaQueue2.Create(defaultGPUNode));
    sdmaQueue2.PlaceAndSubmitPacket(SDMAWriteDataPacket(sdmaQueue2.GetFamilyId(), localBuffer, 0xBBBBBBBB));
    sdmaQueue2.Wait4PacketConsumption();
    EXPECT_SUCCESS(sdmaQueue2.Destroy());

    /* Wait for the child to finish */
    waitpid(cpid, &status, 0);

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(localBuffer));
    EXPECT_SUCCESS(hsaKmtDeregisterMemory(localBuffer));

}

void KFDIPCTest::ParentImportTestChildProcess(int defaultGPUNode, int *pipefd) {
    TearDown();
    SetUp();

    void *localBuffer;
    HsaMemFlags mflags = {0};
    int dmabuf_fd;
    HSAuint64 offset;
    HsaMemoryBuffer tempSysBuffer(PAGE_SIZE, defaultGPUNode, false);
    SDMAQueue sdmaQueue, sdmaQueue2;
    HSAuint64 AlternateVAGPU;
    HsaMemMapFlags mapFlags = {0};

    mflags.ui32.NonPaged = 1;
    mflags.ui32.CoarseGrain = 1;
    ASSERT_SUCCESS(hsaKmtAllocMemory(defaultGPUNode, PAGE_SIZE, mflags, &localBuffer));
    ASSERT_SUCCESS(hsaKmtMapMemoryToGPUNodes(localBuffer, PAGE_SIZE, &AlternateVAGPU,
                       mapFlags, 1, reinterpret_cast<HSAuint32 *>(&defaultGPUNode)));
    tempSysBuffer.Fill(0xAAAAAAAA);

    LOG() << "Child Local buffer: " << (void const *)localBuffer << std::endl;
    LOG() << "Child Temp buffer: " << (void const *)(tempSysBuffer.As<HSAuint32*>()) << std::endl;

    /* Copy pattern in Local Memory before sharing it */
    ASSERT_SUCCESS(sdmaQueue.Create(defaultGPUNode));
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(), localBuffer,
        tempSysBuffer.As<HSAuint32*>(), PAGE_SIZE));
    sdmaQueue.Wait4PacketConsumption();

    hsaKmtExportDMABufHandle(localBuffer, PAGE_SIZE, &dmabuf_fd, &offset);

    ASSERT_GE(write(pipefd[1], reinterpret_cast<void*>(&dmabuf_fd), sizeof(dmabuf_fd)), 0);
    SendDmabufFDOverSocket(dmabuf_fd);
    close(dmabuf_fd);
    EXPECT_SUCCESS(sdmaQueue.Destroy());

    Delay(10 * 1000);

    ASSERT_SUCCESS(sdmaQueue2.Create(defaultGPUNode));
    sdmaQueue2.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue2.GetFamilyId(), tempSysBuffer.As<HSAuint32*>(),
        localBuffer, PAGE_SIZE));
    sdmaQueue2.Wait4PacketConsumption();
    EXPECT_TRUE(WaitOnValue(tempSysBuffer.As<HSAuint32*>(), 0xBBBBBBBB));
    EXPECT_SUCCESS(sdmaQueue2.Destroy());

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(localBuffer));
}

TEST_F(KFDIPCTest, DmabufParentImportTest) {
    TEST_START(TESTPROFILE_RUNALL)

    const std::vector<int>& GpuNodes = m_NodeInfo.GetNodesWithGPU();
    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    int pipefd[2];

    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    if (!GetVramSize(defaultGPUNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    /* Create Pipes for communicating shared handles */
    ASSERT_EQ(pipe(pipefd), 0);

    m_ChildPid = fork();
    if (m_ChildPid == 0)
        ParentImportTestChildProcess(defaultGPUNode, pipefd); /* Child Process */
    else
        ParentImportTestParentProcess(defaultGPUNode, m_ChildPid, pipefd); /* Parent proces */

    /* Code path executed by both parent and child with respective fds */
    close(pipefd[1]);
    close(pipefd[0]);

    TEST_END
}

void KFDIPCTest::MultipleTestChildProcess(int defaultGPUNode, int *pipefd) {
    TearDown();
    SetUp();

    int dmabuf_fd_1, dmabuf_fd_2, dmabuf_fd_3;
    struct amdgpu_bo_import_result import;
    int rn = FindDRMRenderNode(defaultGPUNode);
    amdgpu_bo_handle handle;
    HsaMemoryBuffer tempSysBuffer(PAGE_SIZE, defaultGPUNode, false);
    SDMAQueue sdmaQueue, sdmaQueue2;
    void *localBuffer_1, *localBuffer_2, *localBuffer_3;

    ASSERT_GE(read(pipefd[0], reinterpret_cast<void*>(&dmabuf_fd_1), sizeof(dmabuf_fd_1)), 0);
    dmabuf_fd_1 = RecvDmabufFDOverSocket();

    import.alloc_size = PAGE_SIZE;
    amdgpu_bo_import(m_RenderNodes[rn].device_handle, amdgpu_bo_handle_type_dma_buf_fd, dmabuf_fd_1, &import);
    close(dmabuf_fd_1);

    handle = import.buf_handle;
    amdgpu_bo_cpu_map(handle, &localBuffer_1);

    amdgpu_bo_va_op(handle, 0, PAGE_SIZE,
                            reinterpret_cast<uint64_t>(localBuffer_1), 0, AMDGPU_VA_OP_MAP);

    ASSERT_GE(read(pipefd[0], reinterpret_cast<void*>(&dmabuf_fd_2), sizeof(dmabuf_fd_2)), 0);
    dmabuf_fd_2 = RecvDmabufFDOverSocket();

    import.alloc_size = PAGE_SIZE;
    amdgpu_bo_import(m_RenderNodes[rn].device_handle, amdgpu_bo_handle_type_dma_buf_fd, dmabuf_fd_2, &import);
    close(dmabuf_fd_2);

    handle = import.buf_handle;
    amdgpu_bo_cpu_map(handle, &localBuffer_2);

    amdgpu_bo_va_op(handle, 0, PAGE_SIZE,
                            reinterpret_cast<uint64_t>(localBuffer_2), 0, AMDGPU_VA_OP_MAP);

    ASSERT_GE(read(pipefd[0], reinterpret_cast<void*>(&dmabuf_fd_3), sizeof(dmabuf_fd_3)), 0);
    dmabuf_fd_3 = RecvDmabufFDOverSocket();

    import.alloc_size = PAGE_SIZE;
    amdgpu_bo_import(m_RenderNodes[rn].device_handle, amdgpu_bo_handle_type_dma_buf_fd, dmabuf_fd_3, &import);
    close(dmabuf_fd_3);

    handle = import.buf_handle;
    amdgpu_bo_cpu_map(handle, &localBuffer_3);


    amdgpu_bo_va_op(handle, 0, PAGE_SIZE,
                            reinterpret_cast<uint64_t>(localBuffer_3), 0, AMDGPU_VA_OP_MAP);

    Delay(10 * 1000);

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(localBuffer_1));
    EXPECT_SUCCESS(hsaKmtDeregisterMemory(localBuffer_1));

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(localBuffer_2));
    EXPECT_SUCCESS(hsaKmtDeregisterMemory(localBuffer_2));

    EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(localBuffer_3));
    EXPECT_SUCCESS(hsaKmtDeregisterMemory(localBuffer_3));

}

void KFDIPCTest::MultipleTestParentProcess(int defaultGPUNode, pid_t cpid, int *pipefd) {
    void *localBuffer_1, *localBuffer_2, *localBuffer_3;
    HsaMemFlags mflags = {0};
    int dmabuf_fd_1, dmabuf_fd_2, dmabuf_fd_3;
    HSAuint64 offset;
    int status;
    HSAuint64 AlternateVAGPU;
    HsaMemMapFlags mapFlags = {0};

    mflags.ui32.NonPaged = 1;
    mflags.ui32.CoarseGrain = 1;
    ASSERT_SUCCESS(hsaKmtAllocMemory(defaultGPUNode, PAGE_SIZE, mflags, &localBuffer_1));
    ASSERT_SUCCESS(hsaKmtMapMemoryToGPUNodes(localBuffer_1, PAGE_SIZE, &AlternateVAGPU,
                       mapFlags, 1, reinterpret_cast<HSAuint32 *>(&defaultGPUNode)));

    hsaKmtExportDMABufHandle(localBuffer_1, PAGE_SIZE, &dmabuf_fd_1, &offset);

    ASSERT_GE(write(pipefd[1], reinterpret_cast<void*>(&dmabuf_fd_1), sizeof(dmabuf_fd_1)), 0);
    SendDmabufFDOverSocket(dmabuf_fd_1);
    close(dmabuf_fd_1);

    ASSERT_SUCCESS(hsaKmtAllocMemory(defaultGPUNode, PAGE_SIZE, mflags, &localBuffer_2));
    ASSERT_SUCCESS(hsaKmtMapMemoryToGPUNodes(localBuffer_2, PAGE_SIZE, &AlternateVAGPU,
                       mapFlags, 1, reinterpret_cast<HSAuint32 *>(&defaultGPUNode)));

    hsaKmtExportDMABufHandle(localBuffer_2, PAGE_SIZE, &dmabuf_fd_2, &offset);

    ASSERT_GE(write(pipefd[1], reinterpret_cast<void*>(&dmabuf_fd_2), sizeof(dmabuf_fd_2)), 0);
    SendDmabufFDOverSocket(dmabuf_fd_2);
    close(dmabuf_fd_2);

    ASSERT_SUCCESS(hsaKmtAllocMemory(defaultGPUNode, PAGE_SIZE, mflags, &localBuffer_3));
    ASSERT_SUCCESS(hsaKmtMapMemoryToGPUNodes(localBuffer_3, PAGE_SIZE, &AlternateVAGPU,
                       mapFlags, 1, reinterpret_cast<HSAuint32 *>(&defaultGPUNode)));

    hsaKmtExportDMABufHandle(localBuffer_3, PAGE_SIZE, &dmabuf_fd_3, &offset);

    ASSERT_GE(write(pipefd[1], reinterpret_cast<void*>(&dmabuf_fd_3), sizeof(dmabuf_fd_3)), 0);
    SendDmabufFDOverSocket(dmabuf_fd_3);
    close(dmabuf_fd_3);

    /* Wait for the child to finish */
    waitpid(cpid, &status, 0);
}

TEST_F(KFDIPCTest, DmabufMultipleTest) {
    TEST_START(TESTPROFILE_RUNALL)

    const std::vector<int>& GpuNodes = m_NodeInfo.GetNodesWithGPU();
    int defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    int pipefd[2];

    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    if (!GetVramSize(defaultGPUNode)) {
        LOG() << "Skipping test: No VRAM found." << std::endl;
        return;
    }

    /* Create Pipes for communicating shared handles */
    ASSERT_EQ(pipe(pipefd), 0);

    m_ChildPid = fork();
    if (m_ChildPid == 0)
        MultipleTestChildProcess(defaultGPUNode, pipefd); /* Child Process */
    else
        MultipleTestParentProcess(defaultGPUNode, m_ChildPid, pipefd); /* Parent proces */

    /* Code path executed by both parent and child with respective fds */
    close(pipefd[1]);
    close(pipefd[0]);

    TEST_END
}
