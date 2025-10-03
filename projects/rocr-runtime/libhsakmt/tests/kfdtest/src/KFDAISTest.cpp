/*
 * Copyright (C) 2025  Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/sysmacros.h>
#include "KFDAISTest.hpp"

void KFDAISTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    ROUTINE_END
}

void KFDAISTest::TearDown() {
    ROUTINE_START

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

/*
 * Currently, keep it simple and support only files backed by NVME device.
 * As we know files in NVME is backed by PCI. The driver will futher check if
 * PCI peer-to-peer is possible.
 */
int KFDAISTest::checkIfFileIsInNVME(int fd)
{
    struct stat st;
    char path[PATH_MAX], link[PATH_MAX];

    /* Check if the file lies in NVME*/
    if (fstat(fd, &st) == -1) {
        LOG() << "Failed to stat file " << fd << std::endl;
        return -1;
    }
    snprintf(path, sizeof(path), "/sys/dev/block/%u:%u",
             major(st.st_dev), minor(st.st_dev));

    ssize_t len = readlink(path, link, sizeof(link)-1);
    if (len != -1) {
        link[len] = '\0';
        if (!strstr(link, "nvme")) {
            LOG() << "File doesn't exist on a NVME driver " << std::endl;
            return -1;
        }
    } else {
        LOG() << "File not backed by PCI device" << std::endl;
        return -1;
    }

    return 0;
}

int KFDAISTest::createFileInNVME(void)
{
    m_fd = open(FILENAME, O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, S_IRUSR | S_IWUSR);
    if (m_fd < 0) {
        LOG() << "Failed to create file: " << FILENAME << std::endl;
        return -1;
    }

    if (ftruncate(m_fd, FILE_SIZE) < 0) {
        LOG() << "Failed to set file size: " << FILENAME << std::endl;
        return -1;
    }
    lseek(m_fd, 0, SEEK_SET);
    return 0;
}

/*
 * Fill the file with patterns
 *  [0 - BUFFER_SIZE-1]             <-- START_PATTERN
 *  [BUFFER_SIZE - 2*BUFFER_SIZE - ]<-- START_PATTERN + 1
 *  .
 *  so on for N_BUFFERS
*/
int KFDAISTest::fillFileWithPatterns(void)
{
    const int START_PATTERN = 0x11111111;
    ssize_t written;

    for (int i=0; i < N_BUFFERS; i++) {
        m_pPatterns[i] = START_PATTERN + i;
        std::vector<int> buffer(BUFFER_SIZE / sizeof(int), m_pPatterns[i]);

        written = write(m_fd, buffer.data(), BUFFER_SIZE);

        if (written < 0) {
            LOG() << "Failed to write to file: " << FILENAME << std::endl;
            return -1;
        }
    }

    return 0;
}

/*
 * Check if the buffers copied from the file has the correct pattern
 * To keep it simple only the first and the last buffer is checked
 */
int KFDAISTest::checkBuffersWithPatterns(int gpuNode)
{
    int *lbuf;

    for (int i=0; i < N_BUFFERS; i++) {
        lbuf = (int *)m_pBufs[i] + BUFFER_SIZE/sizeof(int) - 1;

        if (m_pPatterns[i] != *(int *)m_pBufs[i] ||
            m_pPatterns[i] != *lbuf)
            return -1;
    }
    return 0;
}

int KFDAISTest::directReadWritefromFile(void *buf, int size,
                                        HSAint64 file_offset, HsaAisFlags rwFlags)
{
    HSAint32 status;
    HSAuint64 size_copied;

    if (hsaKmtAisReadWriteFile(buf, size, m_fd, file_offset,
                       rwFlags, &size_copied, &status) != HSAKMT_STATUS_SUCCESS)
        return -1;

    if (status || size != size_copied)
        return -1;

    return 0;
}

/*
 * Alloc N_BUFFERS of BUFFER_SIZE in VRAM
*/
int KFDAISTest::allocBuffers(HSAuint32 gpuNode)
{
    HsaMemMapFlags mapFlags = {0};
    HsaMemFlags    flags = {0};
    HSAKMT_STATUS ret;

    flags.Value = 0;
    flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    flags.ui32.HostAccess = 1;
    flags.ui32.NonPaged = 1;

    for (int i=0; i < N_BUFFERS; i++) {
        ret = hsaKmtAllocMemory(gpuNode, BUFFER_SIZE, flags, &m_pBufs[i]);
        if (ret != HSAKMT_STATUS_SUCCESS)
            return ret;

        if (hsakmt_is_dgpu()) {
            ret = hsaKmtMapMemoryToGPUNodes(m_pBufs[i], BUFFER_SIZE, NULL,
                       mapFlags, 1, reinterpret_cast<HSAuint32 *>(&gpuNode));
            if (ret != HSAKMT_STATUS_SUCCESS)
                return ret;
        }
    }

    return ret;
}

void KFDAISTest::freeBuffers(void)
{
    for (int i = 0; i < m_pBufs.size(); i++) {
        if (m_pBufs[i] != NULL) {
            if (hsakmt_is_dgpu())
                EXPECT_SUCCESS(hsaKmtUnmapMemoryToGPU(m_pBufs[i]));
            EXPECT_SUCCESS(hsaKmtFreeMemory(m_pBufs[i], BUFFER_SIZE));
        }
    }
}

TEST_F(KFDAISTest, AISReadTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    HSAuint32 defaultGPUNode = m_NodeInfo.HsaDefaultGPUNode();
    ASSERT_GE(defaultGPUNode, 0) << "failed to get default GPU Node";

    ASSERT_EQ(createFileInNVME(), 0);

    if (checkIfFileIsInNVME(m_fd) < 0) {
        LOG() << "Skipping test. File not in NVME. AIS not supported " << FILENAME << std::endl;
        return;
    }

    /* Fill in test file with different patterns at pre-defined offset */
    ASSERT_EQ(fillFileWithPatterns(), 0);

    /* Allocate multiple buffers */
    ASSERT_EQ(allocBuffers(defaultGPUNode), 0);

    /* Read into each buffer from different pre-defined offset */
    for(int i=0; i < N_BUFFERS; i++)
        ASSERT_EQ(directReadWritefromFile(m_pBufs[i], BUFFER_SIZE, BUFFER_SIZE * i, HSA_AIS_READ), 0);

    /* Check if the read data matches the expected pattern */
    ASSERT_EQ(checkBuffersWithPatterns(defaultGPUNode), 0);

    freeBuffers();
    deleteFileInNVME();

    TEST_END
}
