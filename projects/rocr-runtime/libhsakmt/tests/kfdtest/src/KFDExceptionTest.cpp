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

#include "KFDExceptionTest.hpp"
#include "PM4Queue.hpp"
#include "PM4Packet.hpp"
#include "SDMAPacket.hpp"
#include "SDMAQueue.hpp"
#include "Dispatch.hpp"
#include <sys/mman.h>

void KFDExceptionTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    LOG() << "This Exception test might cause expected page fault "
             "error logs at kernel level." << std::endl;

    ROUTINE_END
}

void KFDExceptionTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    // WORKAROUND: This needs to be fixed in the kernel
    // Wait 500ms for the kernel to process any fault storms before the
    // next test to avoid reporting incorrect faults in the next test.
    Delay(500);

    ROUTINE_END
}

/* Test for memory exception. The function expects a Memory Fault to be
 * triggered by the GPU when it tries to copy dword from pSrc to pDst.
 * Should be called from a Child Process since the Memory Fault causes
 * all the queues to be halted.
*/
void KFDExceptionTest::TestMemoryException(int gpuNode, HSAuint64 pSrc,
                                           HSAuint64 pDst, unsigned int dimX,
                                           unsigned int dimY, unsigned int dimZ) {
    PM4Queue queue;
    HsaEvent *vmFaultEvent;
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    HSAuint64 faultAddress, page_mask = ~((HSAuint64)PAGE_SIZE - 1);
    Dispatch dispatch(isaBuffer, false);

    HsaEventDescriptor eventDesc;
    eventDesc.EventType = HSA_EVENTTYPE_MEMORY;
    eventDesc.NodeId = gpuNode;
    eventDesc.SyncVar.SyncVar.UserData = NULL;
    eventDesc.SyncVar.SyncVarSize = 0;

    ASSERT_SUCCESS_GPU(GetAssemblerFromNodeId(
       gpuNode)->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()), gpuNode);

    m_ChildStatus = queue.Create(gpuNode);
    if (m_ChildStatus != HSAKMT_STATUS_SUCCESS) {
        WARN() << "Queue create failed, on gpuNode: " << gpuNode << std::endl;
        return;
    }
    m_ChildStatus = hsaKmtCreateEvent(&eventDesc, true, false, &vmFaultEvent);
    if (m_ChildStatus != HSAKMT_STATUS_SUCCESS) {
        WARN() << "Event create failed on gpuNode: " << gpuNode << std::endl;
        goto queuefail;
    }

    dispatch.SetDim(dimX, dimY, dimZ);
    dispatch.SetArgs(reinterpret_cast<void *>(pSrc), reinterpret_cast<void *>(pDst));
    dispatch.Submit(queue);

    m_ChildStatus = hsaKmtWaitOnEvent(vmFaultEvent, g_TestTimeOut);
    if (m_ChildStatus != HSAKMT_STATUS_SUCCESS) {
        WARN() << "Wait failed. No Exception triggered on gpuNode: " << gpuNode << std::endl;
        goto eventfail;
    }

    if (vmFaultEvent->EventData.EventType != HSA_EVENTTYPE_MEMORY) {
        WARN() << "Unexpected Event Received on gpuNode: " << gpuNode << vmFaultEvent->EventData.EventType
               << std::endl;
        m_ChildStatus = HSAKMT_STATUS_ERROR;
        goto eventfail;
    }
    faultAddress = vmFaultEvent->EventData.EventData.MemoryAccessFault.VirtualAddress;
    if (faultAddress != (pSrc & page_mask) &&
        faultAddress != (pDst & page_mask) ) {
        WARN() << "gpuNode: " << gpuNode << " Unexpected Fault Address " << faultAddress
               << " expected " << (pSrc & page_mask) << " or "
               << (pDst & page_mask) << std::endl;
        m_ChildStatus = HSAKMT_STATUS_ERROR;
    }

eventfail:
    hsaKmtDestroyEvent(vmFaultEvent);
queuefail:
    queue.Destroy();
}

void KFDExceptionTest::TestSdmaException(int gpuNode, void *pDst) {
    SDMAQueue queue;
    HsaEvent *vmFaultEvent;
    HSAuint64 faultAddress, page_mask = ~((HSAuint64)PAGE_SIZE - 1);


    HsaEventDescriptor eventDesc;
    eventDesc.EventType = HSA_EVENTTYPE_MEMORY;
    eventDesc.NodeId = gpuNode;
    eventDesc.SyncVar.SyncVar.UserData = NULL;
    eventDesc.SyncVar.SyncVarSize = 0;

    m_ChildStatus = queue.Create(gpuNode);
    if (m_ChildStatus != HSAKMT_STATUS_SUCCESS) {
        WARN() << "Queue create failed on gpuNode: " << gpuNode << std::endl;
        return;
    }

    m_ChildStatus = hsaKmtCreateEvent(&eventDesc, true, false, &vmFaultEvent);
    if (m_ChildStatus != HSAKMT_STATUS_SUCCESS) {
        WARN() << "Event create failed on gpuNode: " << gpuNode << std::endl;
        goto queuefail;
    }

    queue.PlaceAndSubmitPacket(SDMAWriteDataPacket(queue.GetFamilyId(),
                                                   reinterpret_cast<void *>(pDst),
                                                   0x02020202));

    m_ChildStatus = hsaKmtWaitOnEvent(vmFaultEvent, g_TestTimeOut);
    if (m_ChildStatus != HSAKMT_STATUS_SUCCESS) {
        WARN() << "Wait failed. No Exception triggered on gpuNode: " << gpuNode << std::endl;
        goto eventfail;
    }

    if (vmFaultEvent->EventData.EventType != HSA_EVENTTYPE_MEMORY) {
        WARN() << "Unexpected Event Received " << vmFaultEvent->EventData.EventType
               << std::endl;
        m_ChildStatus = HSAKMT_STATUS_ERROR;
        goto eventfail;
    }
    faultAddress = vmFaultEvent->EventData.EventData.MemoryAccessFault.VirtualAddress;
    if (faultAddress != ((HSAuint64)pDst & page_mask) ) {
        WARN() << "gpuNode: " << gpuNode << "Unexpected Fault Address " << faultAddress
               << " expected " << ((HSAuint64)pDst & page_mask) << std::endl;
        m_ChildStatus = HSAKMT_STATUS_ERROR;
    }

eventfail:
    hsaKmtDestroyEvent(vmFaultEvent);
queuefail:
    queue.Destroy();
}

void AddressFault(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDExceptionTest* pKFDExceptionTest = (KFDExceptionTest*)pTestParamters->pTestObject;

    const HSAuint32 m_FamilyId = pKFDExceptionTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId == FAMILY_RV) {
        LOG() << "Skipping test: IOMMU issues on Raven." << std::endl;
        return;
    }

    pid_t m_ChildPid = fork();
    if (m_ChildPid == 0) {
        pKFDExceptionTest->TearDown();
        pKFDExceptionTest->SetUp();

        HsaMemoryBuffer srcBuffer(PAGE_SIZE, gpuNode, false);

        srcBuffer.Fill(0xAA55AA55);
        pKFDExceptionTest->TestMemoryException(gpuNode, srcBuffer.As<HSAuint64>(),
                                               0x12345678ULL);
        exit(0);

	} else {
        int childStatus;

        waitpid(m_ChildPid, &childStatus, 0);
        if (hsakmt_is_dgpu()) {
            EXPECT_EQ_GPU(WIFEXITED(childStatus), true, gpuNode);
            EXPECT_EQ_GPU(WEXITSTATUS(childStatus), HSAKMT_STATUS_SUCCESS, gpuNode);
        } else {
            EXPECT_EQ_GPU(WIFSIGNALED(childStatus), true, gpuNode);
            EXPECT_EQ_GPU(WTERMSIG(childStatus), SIGSEGV, gpuNode);
        }
   }
}

/* Test Bad Address access in a child process */
TEST_F(KFDExceptionTest, AddressFault) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTest_Launch(AddressFault));

    TEST_END
}

/* Allocate Read Only buffer. Test Memory Exception failure by
 * attempting to write to that buffer in the child process.
 */
void PermissionFault(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDExceptionTest* pKFDExceptionTest = (KFDExceptionTest*)pTestParamters->pTestObject;

    const HSAuint32 m_FamilyId = pKFDExceptionTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId == FAMILY_RV) {
        LOG() << "Skipping test: IOMMU issues on Raven." << std::endl;
        return;
    }

    pid_t m_ChildPid = fork();
    if (m_ChildPid == 0) {
        pKFDExceptionTest->TearDown();
        pKFDExceptionTest->SetUp();

        HsaMemoryBuffer readOnlyBuffer(PAGE_SIZE, gpuNode, false /*zero*/,
                                       false /*isLocal*/, true /*isExec*/,
                                       false /*isScratch*/, true /*isReadOnly*/);
        HsaMemoryBuffer srcSysBuffer(PAGE_SIZE, gpuNode, false);

        srcSysBuffer.Fill(0xAA55AA55);

        pKFDExceptionTest->TestMemoryException(gpuNode, srcSysBuffer.As<HSAuint64>(),
                            readOnlyBuffer.As<HSAuint64>());

        exit(0);
    } else {
        int childStatus;

        waitpid(m_ChildPid, &childStatus, 0);
        if (hsakmt_is_dgpu()) {
            EXPECT_EQ(WIFEXITED(childStatus), true);
            EXPECT_EQ(WEXITSTATUS(childStatus), HSAKMT_STATUS_SUCCESS);
        } else {
            EXPECT_EQ(WIFSIGNALED(childStatus), true);
            EXPECT_EQ(WTERMSIG(childStatus), SIGSEGV);
        }
    }

}

TEST_F(KFDExceptionTest, PermissionFault) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(PermissionFault));

    TEST_END
}

/* Allocate Read Only user pointer buffer. Test Memory Exception failure by
 * attempting to write to that buffer in the child process.
 */
void PermissionFaultUserPointer(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDExceptionTest* pKFDExceptionTest = (KFDExceptionTest*)pTestParamters->pTestObject;

    const HSAuint32 m_FamilyId = pKFDExceptionTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId == FAMILY_RV) {
        LOG() << "Skipping test: IOMMU issues on Raven." << std::endl;
        return;
    }

    pid_t m_ChildPid = fork();
    if (m_ChildPid == 0) {
        pKFDExceptionTest->TearDown();
        pKFDExceptionTest->SetUp();

         void *pBuf = mmap(NULL, PAGE_SIZE, PROT_READ,
                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
         ASSERT_NE(pBuf, MAP_FAILED);
         EXPECT_SUCCESS(hsaKmtRegisterMemory(pBuf, PAGE_SIZE));
         EXPECT_SUCCESS(hsaKmtMapMemoryToGPU(pBuf, PAGE_SIZE, NULL));
         HsaMemoryBuffer srcSysBuffer(PAGE_SIZE, gpuNode, false);

         srcSysBuffer.Fill(0xAA55AA55);

         pKFDExceptionTest->TestMemoryException(gpuNode, srcSysBuffer.As<HSAuint64>(),
                                                (HSAuint64)pBuf);

        exit(0);
    } else {
        int childStatus;

        waitpid(m_ChildPid, &childStatus, 0);
        if (hsakmt_is_dgpu()) {
            EXPECT_EQ(WIFEXITED(childStatus), true);
            EXPECT_EQ(WEXITSTATUS(childStatus), HSAKMT_STATUS_SUCCESS);
        } else {
            EXPECT_EQ(WIFSIGNALED(childStatus), true);
            EXPECT_EQ(WTERMSIG(childStatus), SIGSEGV);
        }
   }

}

TEST_F(KFDExceptionTest, PermissionFaultUserPointer) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(PermissionFault));

    TEST_END
}

/* Test VM fault storm handling by copying to/from invalid pointers
 * with lots of work items at the same time
 */
void FaultStorm(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDExceptionTest* pKFDExceptionTest = (KFDExceptionTest*)pTestParamters->pTestObject;

    const HSAuint32 m_FamilyId = pKFDExceptionTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId == FAMILY_RV) {
        LOG() << "Skipping test: IOMMU issues on Raven." << std::endl;
        return;
    }

    HSAKMT_STATUS status;

    pid_t m_ChildPid = fork();
    if (m_ChildPid == 0) {
        pKFDExceptionTest->TearDown();
        pKFDExceptionTest->SetUp();

        pKFDExceptionTest->TestMemoryException(gpuNode, 0x12345678, 0x76543210, 1024, 1024, 1);

        exit(0);
    } else {
        int childStatus;

        waitpid(m_ChildPid, &childStatus, 0);
        if (hsakmt_is_dgpu()) {
            EXPECT_EQ_GPU(WIFEXITED(childStatus), true, gpuNode);
            EXPECT_EQ_GPU(WEXITSTATUS(childStatus), HSAKMT_STATUS_SUCCESS, gpuNode);
        } else {
            EXPECT_EQ_GPU(WIFSIGNALED(childStatus), true, gpuNode);
            EXPECT_EQ_GPU(WTERMSIG(childStatus), SIGSEGV, gpuNode);
        }
    }

}

TEST_F(KFDExceptionTest, FaultStorm) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(FaultStorm));

    TEST_END
}

/*
 */
void SdmaQueueException(KFDTEST_PARAMETERS* pTestParamters) {

    int gpuNode = pTestParamters->gpuNode;
    KFDExceptionTest* pKFDExceptionTest = (KFDExceptionTest*)pTestParamters->pTestObject;

    const HSAuint32 m_FamilyId = pKFDExceptionTest->GetFamilyIdFromNodeId(gpuNode);
    if (m_FamilyId == FAMILY_RV) {
        LOG() << "Skipping test: IOMMU issues on Raven." << std::endl;
        return;
    }

    HSAKMT_STATUS status;

    pid_t m_ChildPid = fork();
    if (m_ChildPid == 0) {
        unsigned int* pDb = NULL;
        unsigned int *nullPtr = NULL;

        pKFDExceptionTest->TearDown();
        pKFDExceptionTest->SetUp();

        HsaMemFlags m_MemoryFlags;
        m_MemoryFlags.Value = 0;
       // setting memory flags with default values , can be modified according to needs
        m_MemoryFlags.ui32.NonPaged = 1;                         // Paged
        m_MemoryFlags.ui32.HostAccess = 0;                       // Host accessible
        ASSERT_SUCCESS_GPU(hsaKmtAllocMemory(gpuNode, PAGE_SIZE, m_MemoryFlags,
                                  reinterpret_cast<void**>(&pDb)), gpuNode);
        // verify that pDb is not null before it's being used
        ASSERT_NE_GPU(nullPtr, pDb, gpuNode) << "hsaKmtAllocMemory returned a null pointer";
        ASSERT_SUCCESS_GPU(hsaKmtMapMemoryToGPU(pDb, PAGE_SIZE, NULL), gpuNode);
        EXPECT_SUCCESS_GPU(hsaKmtUnmapMemoryToGPU(pDb), gpuNode);

        pKFDExceptionTest->TestSdmaException(gpuNode, pDb);
        EXPECT_SUCCESS_GPU(hsaKmtFreeMemory(pDb, PAGE_SIZE), gpuNode);

        exit(0);
    } else {
        int childStatus;

        waitpid(m_ChildPid, &childStatus, 0);
        if (hsakmt_is_dgpu()) {
            EXPECT_EQ_GPU(WIFEXITED(childStatus), true, gpuNode);
            EXPECT_EQ_GPU(WEXITSTATUS(childStatus), HSAKMT_STATUS_SUCCESS, gpuNode);
        } else {
            EXPECT_EQ_GPU(WIFSIGNALED(childStatus), true, gpuNode);
            EXPECT_EQ_GPU(WTERMSIG(childStatus), SIGSEGV, gpuNode);
        }
    }
}

TEST_F(KFDExceptionTest, SdmaQueueException) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL)

    ASSERT_SUCCESS(KFDTest_Launch(SdmaQueueException));

    TEST_END
}
