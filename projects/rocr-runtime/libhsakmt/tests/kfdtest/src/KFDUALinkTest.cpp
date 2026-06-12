// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hsakmt/drm/amdgpu_drm.h"
#include "KFDUALinkTest.hpp"
#include "Dispatch.hpp"
#include "KFDTestUtil.hpp"
#include "SDMAQueue.hpp"
#include "SDMAPacket.hpp"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>

//#define USE_SDMA_QUEUE

extern bool g_IsRemoteChildProcess;
extern std::string g_RemoteHost;

using KFDRemoteTestInfra::ExpectRemoteMsg;
using KFDRemoteTestInfra::InitStdIoRemoteSession;
using KFDRemoteTestInfra::PrintProtocolMsg;
using KFDRemoteTestInfra::SendRemoteMsg;
using KFDRemoteTestInfra::ReceiveUalinkHandleMsg;

void KFDUALinkTest::SetUp() {
    ROUTINE_START

    KFDMultiNodeP2PTest::SetUp();

    ROUTINE_END
}

void KFDUALinkTest::TearDown() {
    ROUTINE_START

    if (m_ImporterBo.bo)
        CleanupImporter(m_ImporterBo.bo, m_ImporterBo.va, m_ImporterBo.addr,
                        m_ImporterBo.size);
    if (m_ExporterBo.bo)
        CleanupExporter(m_ExporterBo.bo, m_ExporterBo.va, m_ExporterBo.addr,
                        m_ExporterBo.size);

    KFDMultiNodeP2PTest::TearDown();

    ROUTINE_END
}

int KFDUALinkTest::SetupExporter(int exporterGPU, amdgpu_bo_handle& expBo_handle,
                                 amdgpu_va_handle& vahandle, uint64_t& expBoAddr,
                                 void*& bo_cpu_addr, uint64_t size,
                                 uint64_t (&ualink_handle)[2])
{
    union drm_amdgpu_ualink_handle fh;
    int rn, ret, fd;
    uint32_t bo_handle;

    rn = FindDRMRenderNode(exporterGPU);
    if (rn < 0) {
        LOG() << "Could not find render node for exporter GPU node "
                << exporterGPU << std::endl;
        return 1;
    }

    fd = amdgpu_device_get_fd(m_RenderNodes[rn].device_handle);

    /* Alloc and map a zeroed BO for the specified size, so tests can rely on
     * its initial contents.
     */
    ret = amdgpu_bo_alloc_and_map(m_RenderNodes[rn].device_handle,
            size, PAGE_SIZE, AMDGPU_GEM_DOMAIN_VRAM,
            AMDGPU_GEM_CREATE_VRAM_CLEARED, &expBo_handle,
            &bo_cpu_addr, &expBoAddr, &vahandle);
    if (ret) {
        LOG() << "Failed to alloc buffer of size: " << size << " on exporter "
                << exporterGPU << std::endl;
        return 1;
    }
    m_ExporterBo = {expBo_handle, vahandle, expBoAddr, size};

    LOG() << "amdgpu_device_get_fd(): " << fd << std::endl;
    LOG() << "m_RenderNodes[rn].fd: " << m_RenderNodes[rn].fd << std::endl;
    /* There is no direct way to get the BO handle. Could only find
     * using amdgpu_bo_export() with amdgpu_bo_handle_type_kms type
     * to get the BO handle
     */
    ret = amdgpu_bo_export(expBo_handle, amdgpu_bo_handle_type_kms, &bo_handle);
    if (ret) {
        LOG() << "Failed to get buf handle for allocated buffer on exporter GPU: "
                << exporterGPU << std::endl;
        CleanupExporter(expBo_handle, vahandle, expBoAddr, size);
        return 1;
    }

    LOG() << "Alloced buffer of size: " << size << " GPU addr: " << expBoAddr <<
                " CPU addr: " << bo_cpu_addr << " handle: " << bo_handle << std::endl;

    /* Get the ualink_handle for the BO */
    memset(&fh, 0, sizeof(fh));

    fh.in.op = DRM_AMDGPU_UALINK_HANDLE_OP_EXPORT;
    fh.in.gem_handle = bo_handle;

    ret = drmCommandWriteRead(fd, DRM_AMDGPU_UALINK_HANDLE,
                                &fh, sizeof(fh));
    if (ret) {
        LOG() << "Failed to get ualink_handle for the BO" << std::endl;
        CleanupExporter(expBo_handle, vahandle, expBoAddr, size);
        return 1;
    }

    ualink_handle[0] = fh.out.export_ualink_handle[0];
    ualink_handle[1] = fh.out.export_ualink_handle[1];

    return 0;
}

void KFDUALinkTest::CleanupExporter(amdgpu_bo_handle expBo_handle,
                                    amdgpu_va_handle vaHandle,
                                    uint64_t expBoAddr, uint64_t size)
{
    EXPECT_EQ(0, amdgpu_bo_unmap_and_free(expBo_handle, vaHandle,
                 expBoAddr, size));
    m_ExporterBo = {};
}

int KFDUALinkTest::SetupImporter(int importerGPU, amdgpu_bo_handle& imp_app_bo,
                                 amdgpu_va_handle& va_handle, uint64_t (&ualink_handle)[2],
                                 uint64_t size, uint64_t& imp_app_addr)
{
    struct amdgpu_bo_import_result imp_app_import;
    union drm_amdgpu_ualink_handle fh;
    int rn, ret, fd, dmabuf_fd;

    rn = FindDRMRenderNode(importerGPU);
    if (rn < 0) {
        LOG() << "Could not find render node for importer GPU node "
                << importerGPU << std::endl;
        return 1;
    }

    fd = amdgpu_device_get_fd(m_RenderNodes[rn].device_handle);
    /* Send the ioctl to get a DMA Buf fd corresponding to the ualink_handle */
    memset(&fh, 0, sizeof(fh));

    fh.in.op = DRM_AMDGPU_UALINK_HANDLE_OP_IMPORT;
    fh.in.import_ualink_handle[0] = ualink_handle[0];
    fh.in.import_ualink_handle[1] = ualink_handle[1];

    ret = drmCommandWriteRead(fd, DRM_AMDGPU_UALINK_HANDLE,
                                &fh, sizeof(fh));
    if (ret) {
        LOG() << "Failed to get DmaBuf fd for the handle: " << ualink_handle[0]
                << " " << ualink_handle[1] << std::endl;
        return 1;
    }

    dmabuf_fd = fh.out.import_dmabuf_handle;

    LOG() << "Importer: Import the DMA Buf into the Importer" << std::endl;
    ret = amdgpu_bo_import(m_RenderNodes[rn].device_handle,
                           amdgpu_bo_handle_type_dma_buf_fd,
                           dmabuf_fd, &imp_app_import);
    close(dmabuf_fd);
    if (ret) {
        LOG() << "Failed to import the DmaBuf on importer GPU: " << importerGPU
                << std::endl;
        return 1;
    }
    imp_app_bo = imp_app_import.buf_handle;
    EXPECT_EQ(size, imp_app_import.alloc_size);

    LOG() << "Importer: imp_app_import.buf_handle: " << imp_app_import.buf_handle
                << " imp_app_import.alloc_size: " << imp_app_import.alloc_size
                << std::endl;

    ret = amdgpu_va_range_alloc(m_RenderNodes[rn].device_handle,
                                amdgpu_gpu_va_range_general,
                                imp_app_import.alloc_size, PAGE_SIZE,
                                0, &imp_app_addr, &va_handle, 0);
    if (ret) {
        LOG() << "Failed to reserve a VA range on importer GPU: " << importerGPU
                << std::endl;
        amdgpu_bo_free(imp_app_bo);
        return 1;
    }

    /* Map the BO using GEM API at new virtual address reserved above */
    ret = amdgpu_bo_va_op(imp_app_bo, 0, imp_app_import.alloc_size,
                          imp_app_addr, 0, AMDGPU_VA_OP_MAP);
    if (ret) {
        LOG() << "Failed to map the imported BO on importer GPU: " << importerGPU
                << std::endl;
        amdgpu_va_range_free(va_handle);
        amdgpu_bo_free(imp_app_bo);
        return 1;
    }

    LOG() << "Importer: NPA BO mapped into application at VA: " << imp_app_addr
            << std::endl;
    m_ImporterBo = {imp_app_bo, va_handle, imp_app_addr, imp_app_import.alloc_size};

    return 0;
}

void KFDUALinkTest::CleanupImporter(amdgpu_bo_handle imp_app_bo,
                                    amdgpu_va_handle va_handle,
                                    uint64_t imp_app_addr, uint64_t size)
{
    /* Unmap the importer BO using GEM API */
    LOG() << "Importer: Unmap the imported BO at VA address: " << imp_app_addr
            << std::endl;
    LOG() << "Importer: imp_app_bo: " << imp_app_bo << std::endl;

    EXPECT_EQ(0, amdgpu_bo_va_op(imp_app_bo, 0, size, imp_app_addr, 0,
                                 AMDGPU_VA_OP_UNMAP));
    amdgpu_va_range_free(va_handle);
    amdgpu_bo_free(imp_app_bo);
    m_ImporterBo = {};
}

/* Copy size bytes from src to dst on gpuNode: with CopyPageIsa on a compute
 * queue (size must be a multiple of PAGE_SIZE), or with SDMA when
 * USE_SDMA_QUEUE is defined.
 */
void KFDUALinkTest::CopyBuffer(int gpuNode, void *dst, void *src, uint64_t size)
{
#ifdef USE_SDMA_QUEUE
    SDMAQueue queue;
    ASSERT_SUCCESS(queue.Create(gpuNode));

    /* Each copy packet moves at most 2MB; keep a submission small enough for
     * the default one-page ring.
     */
    constexpr uint64_t kChunk = 32ULL * 1024 * 1024;
    for (uint64_t off = 0; off < size; off += kChunk) {
        uint64_t len = std::min(kChunk, size - off);
        queue.PlaceAndSubmitPacket(SDMACopyDataPacket(queue.GetFamilyId(),
                static_cast<char *>(dst) + off, static_cast<char *>(src) + off, len));
        queue.Wait4PacketConsumption();
    }
#else
    ASSERT_EQ(0ULL, size % PAGE_SIZE) << "CopyPageIsa copies whole pages only";

    PM4Queue queue;
    ASSERT_SUCCESS(queue.Create(gpuNode));
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, gpuNode, true/*zero*/, false/*local*/, true/*exec*/);
    ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(CopyPageIsa, isaBuffer.As<char*>()));

    Dispatch dispatch(isaBuffer);
    dispatch.SetArgs(src, dst);
    dispatch.SetDim(size / PAGE_SIZE, 1, 1);
    dispatch.Submit(queue);
    dispatch.Sync(g_TestTimeOut);
#endif
    ASSERT_SUCCESS(queue.Destroy());
}

/* Fill buf with an offset-dependent pattern so that a page mapped at the
 * wrong offset does not read back as valid data.
 */
static void FillPattern(void *buf, uint64_t size, uint32_t seed)
{
    uint32_t *ptr = static_cast<uint32_t *>(buf);

    for (uint64_t i = 0; i < size / sizeof(uint32_t); i++)
        ptr[i] = seed ^ static_cast<uint32_t>(i);
}

/* Return the number of dwords in buf that do not match FillPattern(seed). */
static uint64_t CheckPattern(const void *buf, uint64_t size, uint32_t seed)
{
    const uint32_t *ptr = static_cast<const uint32_t *>(buf);
    const uint64_t numDwords = size / sizeof(uint32_t);
    uint64_t mismatches = 0;

    for (uint64_t i = 0; i < numDwords; i++) {
        uint32_t expected = seed ^ static_cast<uint32_t>(i);

        if (ptr[i] != expected) {
            if (mismatches < 16)
                LOG() << std::hex << "Mismatch at offset 0x" << i * sizeof(uint32_t)
                      << ": expected 0x" << expected << " got 0x" << ptr[i]
                      << std::dec << std::endl;
            mismatches++;
        }
    }

    if (mismatches)
        LOG() << std::dec << mismatches << " of " << numDwords
              << " dwords mismatched" << std::endl;

    return mismatches;
}

TEST_P(KFDUALinkTest, WriteExporterReadImporter) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (m_FamilyId != FAMILY_GFX125X) {
        LOG() << "Skipping test: Test requires gfx125x asic." << std::endl;
        return;
    }

    constexpr uint32_t exporterWritePattern = 0xcafedead;
    uint64_t kSize = GetParam();

    if (g_IsRemoteChildProcess) {
        KFDRemoteTestInfra::RemoteSession childSession;
        KFDRemoteTestInfra::InitStdIoRemoteSession(childSession);

        int exporterGPU = m_NodeInfo.HsaDefaultGPUNode();
        ASSERT_GE(exporterGPU, 0) << "failed to get default GPU Node";

        amdgpu_bo_handle expBoHandle = NULL;
        amdgpu_va_handle expBoVaHandle;
        uint64_t expBoAddr = 0;
        void* expBoCpuAddr = nullptr;
        uint64_t ualink_handle[2] = {0, 0};
        std::string result = "";

        LOG() << "Exporter: Setting up exporter" << std::endl;
        ASSERT_SUCCESS(SetupExporter(exporterGPU, expBoHandle, expBoVaHandle, expBoAddr,
                            expBoCpuAddr, kSize, ualink_handle));

        ASSERT_TRUE(SendRemoteMsg(childSession, "EXPORT_DONE"));
        LOG() << "Exporter: Sent EXPORT_DONE" << std::endl;

        char ualink_handleMsg[160] = {};
        LOG() << "Exporter: Sending UALINK_HANDLE: " << ualink_handle[0] << " " << ualink_handle[1] << std::endl;
        snprintf(ualink_handleMsg, sizeof(ualink_handleMsg), "UALINK_HANDLE %llu %llu",
                 (unsigned long long)ualink_handle[0],
                 (unsigned long long)ualink_handle[1]);
        ASSERT_TRUE(SendRemoteMsg(childSession, ualink_handleMsg));

        LOG() << "Exporter: Sent UALINK_HANDLE" << std::endl;

        LOG() << "Exporter: Waiting for IMPORT_DONE" << std::endl;
        ASSERT_TRUE(ExpectRemoteMsg(childSession, "IMPORT_DONE", HANDSHAKE_TIMEOUT_MS));
        LOG() << "Exporter: Received IMPORT_DONE" << std::endl;

        /* Write a pattern to the whole buffer on exporter. */
        FillPattern(expBoCpuAddr, kSize, exporterWritePattern);
        LOG() << std::dec << "Exporter: Wrote pattern to " << kSize << " bytes on GPU: "
                << exporterGPU << std::endl;
        ASSERT_TRUE(SendRemoteMsg(childSession, "EXPORT_WRITE_DONE"));
        LOG() << "Exporter: Sent EXPORT_WRITE_DONE" << std::endl;

        /* Wait for the Importer to read the pattern. */
        ASSERT_TRUE(ExpectRemoteMsg(childSession, "IMPORT_READ_DONE", DataTimeoutMs(kSize)));
        LOG() << "Exporter: Received IMPORT_READ_DONE" << std::endl;

        LOG() << "Exporter: Waiting for RESULT" << std::endl;
        ASSERT_TRUE(KFDRemoteTestInfra::ReceiveResultStringMsg(childSession, result,
                    DataTimeoutMs(kSize)));
        LOG() << "Exporter: Received RESULT: " << result << std::endl;

        EXPECT_STREQ(result.c_str(), "PASS");
        CleanupExporter(expBoHandle, expBoVaHandle, expBoAddr, kSize);
        return;
    }

    if (g_RemoteHost.empty()) {
        LOG() << "Skipping test: requires --remote <host> for multi-node run." << std::endl;
        return;
    }

    LOG() << "Parent: Starting remote test" << std::endl;
    ASSERT_TRUE(KFDRemoteTestInfra::StartRemoteTest(g_RemoteHost,
                                m_RemoteBinaryPath,
                                CurrentTestFilter(),
                                m_RemoteSession));

    LOG() << "Parent: Waiting for EXPORT_DONE" << std::endl;
    ASSERT_TRUE(KFDRemoteTestInfra::WaitExpectedMessage(
                    m_RemoteSession,
                    "EXPORT_DONE",
                    DataTimeoutMs(kSize),
                    "Child[stderr]: "));
    KFDRemoteTestInfra::PrintProtocolMsg("Child", "EXPORT_DONE");

    uint64_t ualink_handle[2] = {0, 0};
    ASSERT_TRUE(KFDRemoteTestInfra::ReceiveUalinkHandleMsg(m_RemoteSession, ualink_handle, HANDSHAKE_TIMEOUT_MS));

    LOG() << "Parent: Setting up importer, ualink handle: " << ualink_handle[0] << " " << ualink_handle[1] << std::endl;
    int importerGPU = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(importerGPU, 0) << "failed to get default GPU Node";

    amdgpu_bo_handle impAppBoHandle = NULL;
    amdgpu_va_handle impAppBoVaHandle;
    uint64_t impAppAddr = 0;

    ASSERT_SUCCESS(SetupImporter(importerGPU, impAppBoHandle, impAppBoVaHandle, ualink_handle,
                                 kSize, impAppAddr));
    LOG() << "Parent: Sent IMPORT_DONE" << std::endl;
    ASSERT_TRUE(SendRemoteMsg(m_RemoteSession, "IMPORT_DONE"));

    HsaMemoryBuffer readBuffer(kSize, importerGPU);

    /* Wait for the Exporter to write the pattern. */
    ASSERT_TRUE(KFDRemoteTestInfra::WaitExpectedMessage(
                m_RemoteSession,
                "EXPORT_WRITE_DONE",
                DataTimeoutMs(kSize),
                "Child[stderr]: "));
    KFDRemoteTestInfra::PrintProtocolMsg("Child", "EXPORT_WRITE_DONE");

    /* Copy the whole imported buffer over UALink into system memory. */
    ASSERT_NO_FATAL_FAILURE(CopyBuffer(importerGPU, readBuffer.As<void *>(),
                                       reinterpret_cast<void *>(impAppAddr), kSize));

    ASSERT_TRUE(SendRemoteMsg(m_RemoteSession, "IMPORT_READ_DONE"));
    LOG() << "Importer: Sent IMPORT_READ_DONE" << std::endl;

    /* Verify the pattern on the Importer. */
    LOG() << std::dec << "Importer: Verifying " << kSize << " bytes on GPU: "
          << importerGPU << std::endl;
    if (CheckPattern(readBuffer.As<void *>(), kSize, exporterWritePattern)) {
        ADD_FAILURE();
        EXPECT_TRUE(SendResultStringMsg(m_RemoteSession, "FAIL"));
    } else {
        EXPECT_TRUE(SendResultStringMsg(m_RemoteSession, "PASS"));
    }

    CleanupImporter(impAppBoHandle, impAppBoVaHandle, impAppAddr, kSize);
    KFDRemoteTestInfra::CloseRemoteInput(m_RemoteSession);
    KFDRemoteTestInfra::DrainRemoteStderr(m_RemoteSession, "Child[stderr]: ");
    ASSERT_TRUE(KFDRemoteTestInfra::WaitRemoteExit(m_RemoteSession, 0));
    TEST_END
}

INSTANTIATE_TEST_CASE_P(
    , KFDUALinkTest,
    ::testing::Values(uint64_t{4096}, uint64_t{2*1024*1024}, uint64_t{1024*1024*1024})
);

TEST_P(KFDUALinkTest, WriteImporterReadExporter) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (m_FamilyId != FAMILY_GFX125X) {
        LOG() << "Skipping test: Test requires gfx125x asic." << std::endl;
        return;
    }

    constexpr uint32_t importerWritePattern = 0xdeadbeef;
    uint64_t kSize = GetParam();

    /* Exporter side runs in child process */
    if (g_IsRemoteChildProcess) {
        KFDRemoteTestInfra::RemoteSession childSession;
        KFDRemoteTestInfra::InitStdIoRemoteSession(childSession);

        int exporterGPU = m_NodeInfo.HsaDefaultGPUNode();
        ASSERT_GE(exporterGPU, 0) << "failed to get default GPU Node";

        amdgpu_bo_handle expBoHandle = NULL;
        amdgpu_va_handle expBoVaHandle;
        uint64_t expBoAddr = 0;
        void* expBoCpuAddr = nullptr;
        uint64_t ualink_handle[2] = {0, 0};

        LOG() << "Exporter: Setting up exporter" << std::endl;
        ASSERT_SUCCESS(SetupExporter(exporterGPU, expBoHandle, expBoVaHandle, expBoAddr,
                            expBoCpuAddr, kSize, ualink_handle));

        ASSERT_TRUE(SendRemoteMsg(childSession, "EXPORT_DONE"));
        LOG() << "Exporter: Sent EXPORT_DONE" << std::endl;

        char ualink_handleMsg[160] = {};
        LOG() << "Exporter: Sending UALINK_HANDLE: " << ualink_handle[0] << " " << ualink_handle[1] << std::endl;
        snprintf(ualink_handleMsg, sizeof(ualink_handleMsg), "UALINK_HANDLE %llu %llu",
                 (unsigned long long)ualink_handle[0],
                 (unsigned long long)ualink_handle[1]);
        ASSERT_TRUE(SendRemoteMsg(childSession, ualink_handleMsg));

        LOG() << "Exporter: Sent UALINK_HANDLE" << std::endl;

        LOG() << "Exporter: Waiting for IMPORT_DONE" << std::endl;
        ASSERT_TRUE(ExpectRemoteMsg(childSession, "IMPORT_DONE", HANDSHAKE_TIMEOUT_MS));
        LOG() << "Exporter: Received IMPORT_DONE" << std::endl;

        LOG() << "Exporter: Wait for IMPORT_WRITE_DONE" << std::endl;
        ASSERT_TRUE(ExpectRemoteMsg(childSession, "IMPORT_WRITE_DONE", DataTimeoutMs(kSize)));
        LOG() << "Exporter: Received IMPORT_WRITE_DONE" << std::endl;

        /* CPU reads of VRAM through the BAR are too slow for large buffers,
         * so copy the buffer to system memory with the GPU and verify there.
         */
        HsaMemoryBuffer readBuffer(kSize, exporterGPU);
        ASSERT_NO_FATAL_FAILURE(CopyBuffer(exporterGPU, readBuffer.As<void *>(),
                                           reinterpret_cast<void *>(expBoAddr), kSize));
        LOG() << std::dec << "Exporter: Verifying " << kSize << " bytes on GPU: "
              << exporterGPU << std::endl;
        EXPECT_EQ(0ULL, CheckPattern(readBuffer.As<void *>(), kSize, importerWritePattern));

        ASSERT_TRUE(SendRemoteMsg(childSession, "CLEANUP"));
        CleanupExporter(expBoHandle, expBoVaHandle, expBoAddr, kSize);
        return;
    }

    if (g_RemoteHost.empty()) {
        LOG() << "Skipping test: requires --remote <host> for multi-node run." << std::endl;
        return;
    }

    LOG() << "Parent: Starting remote test" << std::endl;
    ASSERT_TRUE(KFDRemoteTestInfra::StartRemoteTest(g_RemoteHost,
                                m_RemoteBinaryPath,
                                CurrentTestFilter(),
                                m_RemoteSession));

    LOG() << "Parent: Waiting for EXPORT_DONE" << std::endl;
    ASSERT_TRUE(KFDRemoteTestInfra::WaitExpectedMessage(
                    m_RemoteSession,
                    "EXPORT_DONE",
                    DataTimeoutMs(kSize),
                    "Child[stderr]: "));
    KFDRemoteTestInfra::PrintProtocolMsg("Child", "EXPORT_DONE");

    uint64_t ualink_handle[2] = {0, 0};
    ASSERT_TRUE(KFDRemoteTestInfra::ReceiveUalinkHandleMsg(m_RemoteSession, ualink_handle, HANDSHAKE_TIMEOUT_MS));

    LOG() << "Parent: Setting up importer" << std::endl;
    int importerGPU = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(importerGPU, 0) << "failed to get default GPU Node";

    amdgpu_bo_handle impAppBoHandle = NULL;
    amdgpu_va_handle impAppBoVaHandle;
    uint64_t impAppAddr = 0;

    ASSERT_SUCCESS(SetupImporter(importerGPU, impAppBoHandle, impAppBoVaHandle, ualink_handle,
                                 kSize, impAppAddr));
    LOG() << "Parent: Sent IMPORT_DONE" << std::endl;
    ASSERT_TRUE(SendRemoteMsg(m_RemoteSession, "IMPORT_DONE"));

    /* Write a pattern to the whole imported buffer over UALink. */
    HsaMemoryBuffer srcBuffer(kSize, importerGPU);
    FillPattern(srcBuffer.As<void *>(), kSize, importerWritePattern);
    ASSERT_NO_FATAL_FAILURE(CopyBuffer(importerGPU, reinterpret_cast<void *>(impAppAddr),
                                       srcBuffer.As<void *>(), kSize));
    LOG() << std::dec << "Importer: Wrote pattern to " << kSize << " bytes on GPU: "
          << importerGPU << std::endl;

    ASSERT_TRUE(SendRemoteMsg(m_RemoteSession, "IMPORT_WRITE_DONE"));

    ASSERT_TRUE(KFDRemoteTestInfra::WaitExpectedMessage(
        m_RemoteSession,
        "CLEANUP",
        DataTimeoutMs(kSize),
        "Child[stderr]: "));
    KFDRemoteTestInfra::PrintProtocolMsg("Child", "CLEANUP");

    CleanupImporter(impAppBoHandle, impAppBoVaHandle, impAppAddr, kSize);
    KFDRemoteTestInfra::CloseRemoteInput(m_RemoteSession);
    KFDRemoteTestInfra::DrainRemoteStderr(m_RemoteSession, "Child[stderr]: ");
    ASSERT_TRUE(KFDRemoteTestInfra::WaitRemoteExit(m_RemoteSession, 0));
    TEST_END
}

void KFDUALinkTest::AddressFaultImporter(uint64_t kSize,
                                         uint32_t exporterWritePattern) {
    LOG() << "Parent: Starting remote test" << std::endl;
    ASSERT_TRUE(KFDRemoteTestInfra::StartRemoteTest(g_RemoteHost,
                                m_RemoteBinaryPath,
                                CurrentTestFilter(),
                                m_RemoteSession));

    LOG() << "Parent: Waiting for EXPORT_DONE" << std::endl;
    ASSERT_TRUE(KFDRemoteTestInfra::WaitExpectedMessage(
                    m_RemoteSession,
                    "EXPORT_DONE",
                    HANDSHAKE_TIMEOUT_MS,
                    "Child[stderr]: "));
    KFDRemoteTestInfra::PrintProtocolMsg("Child", "EXPORT_DONE");

    uint64_t ualink_handle[2] = {0, 0};
    ASSERT_TRUE(KFDRemoteTestInfra::ReceiveUalinkHandleMsg(m_RemoteSession, ualink_handle, HANDSHAKE_TIMEOUT_MS));

    LOG() << "Parent: Setting up importer" << std::endl;
    int importerGPU = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(importerGPU, 0) << "failed to get default GPU Node";

    amdgpu_bo_handle impAppBoHandle = NULL;
    amdgpu_va_handle impAppBoVaHandle;
    uint64_t impAppAddr = 0;
    HsaEvent *vmFaultEvent;
    HsaEventDescriptor eventDesc;
    eventDesc.EventType = HSA_EVENTTYPE_MEMORY;
    eventDesc.NodeId = importerGPU;
    eventDesc.SyncVar.SyncVar.UserData = NULL;
    eventDesc.SyncVar.SyncVarSize = 0;
    bool testFailed = false;

    ASSERT_SUCCESS(SetupImporter(importerGPU, impAppBoHandle, impAppBoVaHandle,
                                 ualink_handle, kSize, impAppAddr));
    LOG() << "Parent: Sent IMPORT_DONE" << std::endl;
    ASSERT_TRUE(SendRemoteMsg(m_RemoteSession, "IMPORT_DONE"));

    HsaMemoryBuffer sysBuffer(PAGE_SIZE, importerGPU);
    HSAuint32 *sys = sysBuffer.As<HSAuint32*>();
    sys[0] = 0;

    /* Declared before the first goto below so the jumps to the cleanup labels
     * do not cross the initialization of these scalars (ill-formed in C++).
     */
    HSAuint64 pSrc = (HSAuint64)sys;
    HSAuint64 pDst = (HSAuint64)impAppAddr;
    HSAuint64 page_mask = ~((HSAuint64)PAGE_SIZE - 1);
    HSAuint64 faultAddress = 0;

#ifdef USE_SDMA_QUEUE
    SDMAQueue sdmaQueue;
    ASSERT_SUCCESS(sdmaQueue.Create(importerGPU));
#else
    PM4Queue impPm4Queue;
    ASSERT_SUCCESS(impPm4Queue.Create(importerGPU));
    HsaMemoryBuffer isaBuffer(PAGE_SIZE, importerGPU, true, false, true);
    ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(CopyDwordIsa, isaBuffer.As<char*>()));
    /* Constructed here so the gotos below do not jump past its initialization. */
    Dispatch dispatch2(isaBuffer);
#endif

    /* Wait for the Exporter to write the pattern. */
    ASSERT_TRUE(KFDRemoteTestInfra::WaitExpectedMessage(
                m_RemoteSession,
                "EXPORT_WRITE_DONE",
                HANDSHAKE_TIMEOUT_MS,
                "Child[stderr]: "));
    KFDRemoteTestInfra::PrintProtocolMsg("Child", "EXPORT_WRITE_DONE");

    /* Verify the pattern on the Importer. */
#ifdef USE_SDMA_QUEUE
    sdmaQueue.PlaceAndSubmitPacket(SDMACopyDataPacket(sdmaQueue.GetFamilyId(),
            reinterpret_cast<void *>(sys), reinterpret_cast<void *>(impAppAddr), 4));
    sdmaQueue.Wait4PacketConsumption();
#else
    Dispatch dispatch(isaBuffer);
    dispatch.SetArgs(reinterpret_cast<void *>(impAppAddr), reinterpret_cast<void *>(sys));
    dispatch.Submit(impPm4Queue);
    dispatch.Sync(g_TestTimeOut);
#endif

    ASSERT_TRUE(SendRemoteMsg(m_RemoteSession, "IMPORT_READ_DONE"));
    LOG() << "Importer: Sent IMPORT_READ_DONE" << std::endl;
    LOG() << "Importer: Expected pattern: " << exporterWritePattern <<
          " but got " << sys[0] << " on GPU: " << importerGPU << std::endl;

    if (sys[0] != exporterWritePattern) {
        testFailed = true;
        goto queuefail;
    }

    /* Wait for the Exporter to free the exported memory. */
    ASSERT_TRUE(KFDRemoteTestInfra::WaitExpectedMessage(
        m_RemoteSession,
        "EXPORT_MEMORY_FREED",
        HANDSHAKE_TIMEOUT_MS,
        "Child[stderr]: "));
    KFDRemoteTestInfra::PrintProtocolMsg("Child", "EXPORT_MEMORY_FREED");

    Delay(REVOKE_SETTLE_MS);

    m_ChildStatus = hsaKmtCreateEvent(&eventDesc, true, false, &vmFaultEvent);
    if (m_ChildStatus != HSAKMT_STATUS_SUCCESS) {
        WARN() << "Event create failed on gpuNode: " << importerGPU << std::endl;
        testFailed = true;
        goto queuefail;
    }

    LOG() << "Try to write to memory on Importer. This should result in a VM fault on the Importer" << std::endl;
    // Try to write to remote memory. This should throw a VM fault since the memory was
    // unmapped at the NPA address on the exporter.
    sys[0] = 0x10101010;
#ifdef USE_SDMA_QUEUE
    sdmaQueue.PlaceAndSubmitPacket(SDMAWriteDataPacket(sdmaQueue.GetFamilyId(),
            reinterpret_cast<void *>(pDst), sys[0]));
    sdmaQueue.Wait4PacketConsumption();
#else
    dispatch2.SetArgs(reinterpret_cast<void *>(pSrc), reinterpret_cast<void *>(pDst));
    dispatch2.Submit(impPm4Queue);
#endif

    m_ChildStatus = hsaKmtWaitOnEvent(vmFaultEvent, g_TestTimeOut);
    if (m_ChildStatus != HSAKMT_STATUS_SUCCESS) {
        WARN() << "Wait failed. No Exception triggered on gpuNode: " << importerGPU << std::endl;
        testFailed = true;
        goto eventfail;
    }

    if (vmFaultEvent->EventData.EventType != HSA_EVENTTYPE_MEMORY) {
        WARN() << "Unexpected Event Received on gpuNode: " << importerGPU << " "
                << vmFaultEvent->EventData.EventType << std::endl;
        testFailed = true;
        goto eventfail;
    }

    faultAddress = vmFaultEvent->EventData.EventData.MemoryAccessFault.VirtualAddress;
    if (faultAddress != (pSrc & page_mask) &&
        faultAddress != (pDst & page_mask) ) {
        WARN() << "gpuNode: " << importerGPU << " Unexpected Fault Address " << faultAddress
               << " expected " << (pSrc & page_mask) << " or "
               << (pDst & page_mask) << std::endl;
        testFailed = true;
    }

eventfail:
    hsaKmtDestroyEvent(vmFaultEvent);
queuefail:
    // Cleanup the queue.
#ifdef USE_SDMA_QUEUE
    ASSERT_SUCCESS(sdmaQueue.Destroy());
#else
    ASSERT_SUCCESS(impPm4Queue.Destroy());
#endif

    CleanupImporter(impAppBoHandle, impAppBoVaHandle, impAppAddr, kSize);
    if (testFailed) {
        ADD_FAILURE();
        EXPECT_TRUE(SendResultStringMsg(m_RemoteSession, "FAIL"));
        LOG() << "Parent: Sent FAIL" << std::endl;
    } else {
        EXPECT_TRUE(SendResultStringMsg(m_RemoteSession, "PASS"));
        LOG() << "Parent: Sent PASS" << std::endl;
    }
    KFDRemoteTestInfra::CloseRemoteInput(m_RemoteSession);
    KFDRemoteTestInfra::DrainRemoteStderr(m_RemoteSession, "Child[stderr]: ");
    ASSERT_TRUE(KFDRemoteTestInfra::WaitRemoteExit(m_RemoteSession, 0));
}

TEST_F(KFDUALinkTest, AddressFault) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (m_FamilyId != FAMILY_GFX125X) {
        LOG() << "Skipping test: Test requires gfx125x asic." << std::endl;
        return;
    }

    constexpr uint64_t kSize = PAGE_SIZE;
    constexpr uint32_t exporterWritePattern = 0x12345678;

    /* Exporter side runs in child process */
    if (g_IsRemoteChildProcess) {
        KFDRemoteTestInfra::RemoteSession childSession;
        KFDRemoteTestInfra::InitStdIoRemoteSession(childSession);

        int exporterGPU = m_NodeInfo.HsaDefaultGPUNode();
        ASSERT_GE(exporterGPU, 0) << "failed to get default GPU Node";

        amdgpu_bo_handle expBoHandle = NULL;
        amdgpu_va_handle expBoVaHandle;
        uint64_t expBoAddr = 0;
        void* expBoCpuAddr = nullptr;
        uint64_t ualink_handle[2] = {0, 0};
        std::string result = "";

        LOG() << "Exporter: Setting up exporter" << std::endl;
        ASSERT_SUCCESS(SetupExporter(exporterGPU, expBoHandle, expBoVaHandle, expBoAddr,
                            expBoCpuAddr, kSize, ualink_handle));

        ASSERT_TRUE(SendRemoteMsg(childSession, "EXPORT_DONE"));
        LOG() << "Exporter: Sent EXPORT_DONE" << std::endl;

        char ualink_handleMsg[160] = {};
        LOG() << "Exporter: Sending UALINK_HANDLE: " << ualink_handle[0] << " " << ualink_handle[1] << std::endl;
        snprintf(ualink_handleMsg, sizeof(ualink_handleMsg), "UALINK_HANDLE %llu %llu",
                 (unsigned long long)ualink_handle[0],
                 (unsigned long long)ualink_handle[1]);
        ASSERT_TRUE(SendRemoteMsg(childSession, ualink_handleMsg));

        LOG() << "Exporter: Sent UALINK_HANDLE" << std::endl;

        LOG() << "Exporter: Waiting for IMPORT_DONE" << std::endl;
        ASSERT_TRUE(ExpectRemoteMsg(childSession, "IMPORT_DONE", HANDSHAKE_TIMEOUT_MS));
        LOG() << "Exporter: Received IMPORT_DONE" << std::endl;

        /* Write a pattern to memory on exporter. */
        *reinterpret_cast<uint32_t*>(expBoCpuAddr) = exporterWritePattern;
        LOG() << "Exporter: Writing pattern to memory: " <<
                *reinterpret_cast<uint32_t*>(expBoCpuAddr) << " on GPU: "
                << exporterGPU << std::endl;
        ASSERT_TRUE(SendRemoteMsg(childSession, "EXPORT_WRITE_DONE"));
        LOG() << "Exporter: Sent EXPORT_WRITE_DONE" << std::endl;

        /* Wait for the Importer to read the pattern. */
        ASSERT_TRUE(ExpectRemoteMsg(childSession, "IMPORT_READ_DONE", HANDSHAKE_TIMEOUT_MS));
        LOG() << "Exporter: Received IMPORT_READ_DONE" << std::endl;

        LOG() << "Exporter: Free the exporter memory and send EXPORT_MEMORY_FREED" << std::endl;
        CleanupExporter(expBoHandle, expBoVaHandle, expBoAddr, kSize);
        ASSERT_TRUE(SendRemoteMsg(childSession, "EXPORT_MEMORY_FREED"));
        LOG() << "Exporter: Sent EXPORT_MEMORY_FREED" << std::endl;

        LOG() << "Exporter: Waiting for RESULT" << std::endl;
        ASSERT_TRUE(KFDRemoteTestInfra::ReceiveResultStringMsg(childSession, result,
                    HANDSHAKE_TIMEOUT_MS));
        LOG() << "Exporter: Received RESULT: " << result << std::endl;

        EXPECT_STREQ(result.c_str(), "PASS");

        return;
    }

    if (g_RemoteHost.empty()) {
        LOG() << "Skipping test: requires --remote <host> for multi-node run." << std::endl;
        return;
    }

    /* The expected VM fault leaves this process unable to run further GPU
     * work, so the importer side runs in a forked child.
     */
    pid_t childPid = fork();
    ASSERT_GE(childPid, 0) << "fork failed: " << strerror(errno);
    if (childPid == 0) {
        TearDown();
        SetUp();
        if (!HasFatalFailure())
            AddressFaultImporter(kSize, exporterWritePattern);
        TearDown();
        exit(HasFailure() ? EXIT_FAILURE : EXIT_SUCCESS);
    }

    int childStatus = 0;
    while (waitpid(childPid, &childStatus, 0) < 0 && errno == EINTR)
        ;
    EXPECT_TRUE(WIFEXITED(childStatus) && WEXITSTATUS(childStatus) == EXIT_SUCCESS)
        << "Importer child failed, wait status " << childStatus;
    TEST_END
}

TEST_F(KFDUALinkTest, AtomicsAdd) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    if (m_FamilyId != FAMILY_GFX125X) {
        LOG() << "Skipping test: Test requires gfx125x asic." << std::endl;
        return;
    }

    constexpr uint64_t kSize = PAGE_SIZE;
    /* Exporter side runs in child process */
    if (g_IsRemoteChildProcess) {
        KFDRemoteTestInfra::RemoteSession childSession;
        KFDRemoteTestInfra::InitStdIoRemoteSession(childSession);

        int exporterGPU = m_NodeInfo.HsaDefaultGPUNode();
        ASSERT_GE(exporterGPU, 0) << "failed to get default GPU Node";

        amdgpu_bo_handle expBoHandle = NULL;
        amdgpu_va_handle expBoVaHandle;
        uint64_t expBoAddr = 0;
        void* expBoCpuAddr = nullptr;
        uint64_t ualink_handle[2] = {0, 0};

        LOG() << "Exporter: Setting up exporter" << std::endl;
        ASSERT_SUCCESS(SetupExporter(exporterGPU, expBoHandle, expBoVaHandle, expBoAddr,
                            expBoCpuAddr, kSize, ualink_handle));

        ASSERT_TRUE(SendRemoteMsg(childSession, "EXPORT_DONE"));
        LOG() << "Exporter: Sent EXPORT_DONE" << std::endl;

        char ualink_handleMsg[160] = {};
        LOG() << "Exporter: Sending UALINK_HANDLE: " << ualink_handle[0] << " " << ualink_handle[1] << std::endl;
        snprintf(ualink_handleMsg, sizeof(ualink_handleMsg), "UALINK_HANDLE %llu %llu",
                 (unsigned long long)ualink_handle[0],
                 (unsigned long long)ualink_handle[1]);
        ASSERT_TRUE(SendRemoteMsg(childSession, ualink_handleMsg));

        LOG() << "Exporter: Sent UALINK_HANDLE" << std::endl;

        LOG() << "Exporter: Waiting for IMPORT_DONE" << std::endl;
        ASSERT_TRUE(ExpectRemoteMsg(childSession, "IMPORT_DONE", HANDSHAKE_TIMEOUT_MS));
        LOG() << "Exporter: Received IMPORT_DONE" << std::endl;

        /* Setup a PM4 queue on Exporter to do atomic increment on the exporter memory. */
        PM4Queue expPm4Queue;
        ASSERT_SUCCESS(expPm4Queue.Create(exporterGPU));
        HsaMemoryBuffer expIsaBuffer(PAGE_SIZE, exporterGPU, true/*zero*/, false/*local*/, true/*exec*/);
        ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(AtomicIncIsa, expIsaBuffer.As<char*>()));

        /* Setup the Atomic Inc dispatch on the exporter. */
        Dispatch expDispatch(expIsaBuffer);
        expDispatch.SetArgs(reinterpret_cast<void *>(expBoAddr), NULL);
        expDispatch.SetDim(1024, 1, 1);

        /* Both sides add 1024, so the counter must start at 0. */
        volatile uint32_t *counter = reinterpret_cast<volatile uint32_t *>(expBoCpuAddr);
        *counter = 0;
        ASSERT_EQ(0U, *counter);

        ASSERT_TRUE(SendRemoteMsg(childSession, "SETUP_DISPATCH"));
        LOG() << "Exporter: Sent SETUP_DISPATCH" << std::endl;

        LOG() << "Exporter: Wait for IMPORT_DISPATCH_START" << std::endl;
        ASSERT_TRUE(ExpectRemoteMsg(childSession, "IMPORT_DISPATCH_START", HANDSHAKE_TIMEOUT_MS));
        LOG() << "Exporter: Received IMPORT_DISPATCH_START" << std::endl;

        LOG() << std::dec << "Exporter: Value read before exporter dispatch: "
              << *counter << " on GPU: "
              << exporterGPU << std::endl;
        expDispatch.Submit(expPm4Queue);
        expDispatch.Sync(g_TestTimeOut);

        LOG() << "Exporter: Wait for IMPORT_DISPATCH_DONE" << std::endl;
        ASSERT_TRUE(ExpectRemoteMsg(childSession, "IMPORT_DISPATCH_DONE", HANDSHAKE_TIMEOUT_MS));
        LOG() << "Exporter: Received IMPORT_DISPATCH_DONE" << std::endl;

        EXPECT_EQ(2048U, *counter);
        LOG() << std::dec << "Exporter: Value read: " << *counter << " on GPU: "
              << exporterGPU << std::endl;

        ASSERT_SUCCESS(expPm4Queue.Destroy());
        ASSERT_TRUE(SendRemoteMsg(childSession, "CLEANUP"));
        LOG() << "Exporter: Sent CLEANUP" << std::endl;
        ASSERT_TRUE(ExpectRemoteMsg(childSession, "CLEANUP_DONE", HANDSHAKE_TIMEOUT_MS));

        LOG() << "Exporter: Received CLEANUP_DONE" << std::endl;
        CleanupExporter(expBoHandle, expBoVaHandle, expBoAddr, kSize);
        return;
    }

    if (g_RemoteHost.empty()) {
        LOG() << "Skipping test: requires --remote <host> for multi-node run." << std::endl;
        return;
    }

    LOG() << "Parent: Starting remote test" << std::endl;
    ASSERT_TRUE(KFDRemoteTestInfra::StartRemoteTest(g_RemoteHost,
                                m_RemoteBinaryPath,
                                CurrentTestFilter(),
                                m_RemoteSession));

    LOG() << "Parent: Waiting for EXPORT_DONE" << std::endl;
    ASSERT_TRUE(KFDRemoteTestInfra::WaitExpectedMessage(
                    m_RemoteSession,
                    "EXPORT_DONE",
                    HANDSHAKE_TIMEOUT_MS,
                    "Child[stderr]: "));
    KFDRemoteTestInfra::PrintProtocolMsg("Child", "EXPORT_DONE");

    uint64_t ualink_handle[2] = {0, 0};
    ASSERT_TRUE(KFDRemoteTestInfra::ReceiveUalinkHandleMsg(m_RemoteSession, ualink_handle, HANDSHAKE_TIMEOUT_MS));

    LOG() << "Parent: Import handle from exporter" << std::endl;
    int importerGPU = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(importerGPU, 0) << "failed to get default GPU Node";

    amdgpu_bo_handle impAppBoHandle = NULL;
    amdgpu_va_handle impAppBoVaHandle;
    uint64_t impAppAddr = 0;

    ASSERT_SUCCESS(SetupImporter(importerGPU, impAppBoHandle, impAppBoVaHandle, ualink_handle,
                                 kSize, impAppAddr));
    LOG() << "Parent: Sent IMPORT_DONE" << std::endl;
    ASSERT_TRUE(SendRemoteMsg(m_RemoteSession, "IMPORT_DONE"));

    /* Setup a PM4 queue on Exporter to do atomic increment on the exporter memory. */
    PM4Queue impPm4Queue;
    ASSERT_SUCCESS(impPm4Queue.Create(importerGPU));
    HsaMemoryBuffer impIsaBuffer(PAGE_SIZE, importerGPU, true/*zero*/, false/*local*/, true/*exec*/);
    ASSERT_SUCCESS(m_pAsm->RunAssembleBuf(AtomicIncIsa, impIsaBuffer.As<char*>()));

    ASSERT_TRUE(KFDRemoteTestInfra::WaitExpectedMessage(
        m_RemoteSession,
        "SETUP_DISPATCH",
        HANDSHAKE_TIMEOUT_MS,
        "Child[stderr]: "));
    KFDRemoteTestInfra::PrintProtocolMsg("Child", "SETUP_DISPATCH");

    /* Setup the Atomic Inc dispatch on the importer. */
    Dispatch impDispatch(impIsaBuffer);
    impDispatch.SetArgs(reinterpret_cast<void *>(impAppAddr), NULL);
    impDispatch.SetDim(1024, 1, 1);

    ASSERT_TRUE(SendRemoteMsg(m_RemoteSession, "IMPORT_DISPATCH_START"));
    LOG() << "Parent: Sent IMPORT_DISPATCH_START" << std::endl;

    impDispatch.Submit(impPm4Queue);
    impDispatch.Sync(g_TestTimeOut);

    LOG() << "Parent: Send IMPORT_DISPATCH_DONE" << std::endl;
    ASSERT_TRUE(SendRemoteMsg(m_RemoteSession, "IMPORT_DISPATCH_DONE"));
    LOG() << "Parent: Sent IMPORT_DISPATCH_DONE" << std::endl;

    ASSERT_SUCCESS(impPm4Queue.Destroy());

    ASSERT_TRUE(KFDRemoteTestInfra::WaitExpectedMessage(
        m_RemoteSession,
        "CLEANUP",
        HANDSHAKE_TIMEOUT_MS,
        "Child[stderr]: "));
    KFDRemoteTestInfra::PrintProtocolMsg("Child", "CLEANUP");

    CleanupImporter(impAppBoHandle, impAppBoVaHandle, impAppAddr, kSize);
    ASSERT_TRUE(SendRemoteMsg(m_RemoteSession, "CLEANUP_DONE"));
    LOG() << "Parent: Sent CLEANUP_DONE" << std::endl;

    KFDRemoteTestInfra::CloseRemoteInput(m_RemoteSession);
    KFDRemoteTestInfra::DrainRemoteStderr(m_RemoteSession, "Child[stderr]: ");
    ASSERT_TRUE(KFDRemoteTestInfra::WaitRemoteExit(m_RemoteSession, 0));
    TEST_END
}
