/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef __KFD_AIS_TEST__H__
#define __KFD_AIS_TEST__H__

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include "KFDBaseComponentTest.hpp"

class KFDAISTest : public KFDBaseComponentTest {
public:
    /*
     * Configurable constants - adjust these for different test scenarios
     */
    static constexpr int BUFFER_SIZE = 32 * 4096;           // 128KB per buffer
    static constexpr int FILE_SIZE_BASIC = 1024 * 1024;     // 1MB for basic tests
    static constexpr int N_THREADS = 8;                      // Thread count for MT test
    static constexpr int N_BUFFERS_BASIC = FILE_SIZE_BASIC / BUFFER_SIZE;

    static constexpr int MT_ALIGN = 4096;                    // O_DIRECT alignment

    static constexpr const char *TEST_FILENAME = "kfdais_test_file.bin";
    static constexpr const char *TEST_FILENAME_OUT = "kfdais_test_file_out.bin";

    KFDAISTest(void) : m_fd(-1), m_fdOut(-1), m_gpuNode(-1), m_isOnNVME(false),
                       m_bufSize(BUFFER_SIZE) {}
    ~KFDAISTest(void) {}

protected:
    virtual void SetUp();
    virtual void TearDown();

    /*
     * NVME detection - check if file descriptor points to NVME-backed storage
     * Returns 0 if on NVME, -1 otherwise
     */
    int checkIfFileIsOnNVME(int fd);

    /*
     * Find an NVME-backed path for testing
     * Tries common NVME mount points, falls back to /tmp
     * Returns true if NVME found, false if using fallback
     */
    bool findNVMEPath(std::string &outPath);

    /*
     * Create test file at specified path
     * Opens with O_DIRECT for AIS compatibility
     */
    int createTestFile(const std::string &path, size_t size);

    /*
     * Delete test files and close file descriptors
     */
    void deleteTestFiles();

    /*
     * Fill file with patterns - each chunkSize section gets a unique pattern
     * Pattern for section i = START_PATTERN + i
     */
    int fillFileWithPatterns(int fd, size_t fileSize, size_t chunkSize,
                             std::vector<uint32_t> &patterns);

    /*
     * Validate VRAM buffers contain expected patterns.
     * Compares every word of each bufferSize-byte buffer against its pattern.
     */
    int checkBuffersWithPatterns(const std::vector<void*> &bufs, size_t bufferSize,
                                  const std::vector<uint32_t> &patterns);

    /*
     * Allocate VRAM buffers for AIS operations
     * Buffers are NonPaged + HostAccess for AIS compatibility
     */
    int allocVRAMBuffers(int gpuNode, int numBuffers, size_t bufferSize,
                         std::vector<void*> &bufs);

    /*
     * Free previously allocated VRAM buffers
     */
    void freeVRAMBuffers(std::vector<void*> &bufs, size_t bufferSize);

    /*
     * Wrapper for hsaKmtAisReadWriteFile
     * Returns 0 on success, -1 on failure
     */
    int aisReadWrite(void *buf, size_t size, int fd, off_t fileOffset,
                     HsaAisFlags flags, HSAuint64 *sizeCopied = nullptr);

    int m_fd;                       // Input file descriptor
    int m_fdOut;                    // Output file descriptor (for write tests)
    int m_gpuNode;                  // Default GPU node
    bool m_isOnNVME;                // True if test file is on NVME
    std::string m_testDir;          // Directory for test files
    std::string m_testFilePath;     // Full path to input test file
    std::string m_testFilePathOut;  // Full path to output test file
    std::vector<void*> m_pBufs;     // VRAM buffer pointers
    size_t m_bufSize;               // Size of each buffer in m_pBufs (for TearDown)
    std::vector<uint32_t> m_patterns; // Expected patterns for validation
};

#endif // __KFD_AIS_TEST__H__
