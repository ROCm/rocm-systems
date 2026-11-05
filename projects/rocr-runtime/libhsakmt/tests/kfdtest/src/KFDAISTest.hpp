// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
    static constexpr int N_BUFFERS_BASIC = FILE_SIZE_BASIC / BUFFER_SIZE;

    static constexpr int MT_ALIGN = 4096;                    // O_DIRECT alignment

    /*
     * mkstemp() templates. Unique per-run names keep concurrent kfdtest
     * instances (e.g. CI shards) from colliding, and remove the
     * truncate/symlink-clobber risk of a fixed name in a world-writable
     * directory such as /tmp or /dev/shm.
     */
    static constexpr const char *TEST_FILE_TEMPLATE = "kfdais_test_file.XXXXXX";
    static constexpr const char *TEST_FILE_OUT_TEMPLATE = "kfdais_test_out.XXXXXX";

    /* aisReadWrite() return codes. */
    static constexpr int AIS_OK = 0;
    static constexpr int AIS_FAILED = -1;
    static constexpr int AIS_UNSUPPORTED = -2;  // kernel reported ENODEV

    KFDAISTest(void) : m_fd(-1), m_fdOut(-1), m_gpuNode(-1), m_isOnNVME(false),
                       m_bufSize(BUFFER_SIZE) {}
    ~KFDAISTest(void) {}

protected:
    virtual void SetUp();
    virtual void TearDown();

    /*
     * Whether the KFD reports AIS as initialized on this node.
     *
     * AIS support is decided entirely by the kernel (kfd_ais_init): either a
     * large-BAR, non-SRIOV dGPU on a CONFIG_PCI_P2PDMA kernel, or an
     * XGMI-attached device with GC IP 12.1.0. There is deliberately no
     * ASIC/family list here - duplicating that policy in userspace would
     * drift as new ASICs land. Query the node capability instead.
     */
    bool isAISSupported(int node);

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
     * Create a uniquely-named test file of the given size under 'dir'.
     * Opened with O_DIRECT where the filesystem supports it. The resulting
     * path is returned in outPath so TearDown can unlink it.
     */
    int createTestFile(const std::string &dir, const char *templateName,
                       size_t size, std::string &outPath);

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
     * Wrapper for hsaKmtAisReadWriteFile.
     * Returns AIS_OK, AIS_FAILED, or AIS_UNSUPPORTED (kernel returned ENODEV,
     * i.e. AIS is not initialized for this device).
     *
     * Set 'quiet' when calling from worker threads: LOG() is not safe to
     * interleave across threads, so callers log after joining instead.
     */
    int aisReadWrite(void *buf, size_t size, int fd, off_t fileOffset,
                     HsaAisFlags flags, HSAuint64 *sizeCopied = nullptr,
                     bool quiet = false);

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
