// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <algorithm>
#include <memory>

#include "KFDAISTest.hpp"

/*
 * O_DIRECT read()/write() require a page-aligned buffer; std::vector is only
 * ~16-byte aligned and fails EINVAL on NVMe. Return a self-freeing buffer
 * aligned to MT_ALIGN, or nullptr on failure.
 */
static std::unique_ptr<void, decltype(&free)> allocAligned(size_t size) {
    void *p = nullptr;
    if (posix_memalign(&p, KFDAISTest::MT_ALIGN, size) != 0)
        p = nullptr;
    return std::unique_ptr<void, decltype(&free)>(p, free);
}

void KFDAISTest::SetUp() {
    ROUTINE_START

    KFDBaseComponentTest::SetUp();

    m_gpuNode = m_NodeInfo.HsaDefaultGPUNode();
    m_fd = -1;
    m_fdOut = -1;
    m_isOnNVME = false;
    m_bufSize = BUFFER_SIZE;
    m_pBufs.clear();
    m_patterns.clear();

    ROUTINE_END
}

void KFDAISTest::TearDown() {
    ROUTINE_START

    deleteTestFiles();
    freeVRAMBuffers(m_pBufs, m_bufSize);
    m_pBufs.clear();
    m_patterns.clear();

    KFDBaseComponentTest::TearDown();

    ROUTINE_END
}

bool KFDAISTest::isAISSupported(int node) {
    const HsaNodeProperties *props = m_NodeInfo.GetNodeProperties(node);

    if (props == NULL)
        return false;

    /*
     * The KFD currently advertises "AIS is initialized for this device" by
     * temporarily reusing the VA_LIMIT capability bit (see the comment on
     * HSA_CAP_VA_LIMIT in kfd_topology.c). Keep that knowledge in this one
     * helper so there is a single edit site once a dedicated capability bit
     * is allocated.
     *
     * TODO (AILIKFD-831): move to a dedicated HSA_CAPABILITY2 bit; the
     * VA_LIMIT reuse is temporary.
     */
    return props->Capability.ui32.VALimit;
}

int KFDAISTest::checkIfFileIsOnNVME(int fd) {
    struct stat st;
    char sysPath[PATH_MAX];
    char linkTarget[PATH_MAX];

    if (fstat(fd, &st) == -1) {
        LOG() << "Failed to stat file descriptor " << fd << std::endl;
        return -1;
    }

    snprintf(sysPath, sizeof(sysPath), "/sys/dev/block/%u:%u",
             major(st.st_dev), minor(st.st_dev));

    ssize_t len = readlink(sysPath, linkTarget, sizeof(linkTarget) - 1);
    if (len == -1) {
        LOG() << "File not backed by block device (may be tmpfs)" << std::endl;
        return -1;
    }

    linkTarget[len] = '\0';

    if (strstr(linkTarget, "nvme") == NULL) {
        LOG() << "File is not on NVME device (path: " << linkTarget << ")" << std::endl;
        return -1;
    }

    LOG() << "File is on NVME device (path: " << linkTarget << ")" << std::endl;
    return 0;
}

bool KFDAISTest::findNVMEPath(std::string &outPath) {
    const char* candidates[] = {
        "/mnt/nvme/",
        "/data/nvme/",
        "/scratch/nvme/",
        "/nvme/",
        "/mnt/scratch/",
        "/data/",
        "/scratch/",
        /*
         * Try /tmp/ itself before giving up: on many single-drive dev/test
         * boxes /tmp is just a directory on the NVMe-backed root fs, not a
         * separate mount or tmpfs. Checking it here (instead of only as the
         * unconditional non-NVME fallback below) lets checkIfFileIsOnNVME()
         * actually detect that case. Confirmed needed on real MI300X test
         * hardware (AILIKFD-831, 2026-07-27): root fs was nvme0n1p2 and
         * every candidate above this line was absent.
         */
        "/tmp/"
    };

    for (const char* path : candidates) {
        struct stat st;
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
            std::string tmpl = std::string(path) + "kfdtest_ais_probe.XXXXXX";
            std::vector<char> tmp(tmpl.begin(), tmpl.end());
            tmp.push_back(0);
            int fd = mkstemp(tmp.data());
            std::string testPath(tmp.data());
            if (fd >= 0) {
                bool isNVME = (checkIfFileIsOnNVME(fd) == 0);
                close(fd);
                unlink(testPath.c_str());
                if (isNVME) {
                    outPath = path;
                    LOG() << "Found NVME path: " << path << std::endl;
                    return true;
                }
            }
        }
    }

    outPath = "/tmp/";
    LOG() << "No NVME path found, using /tmp/ (AIS tests will detect non-NVME)" << std::endl;
    return false;
}

int KFDAISTest::createTestFile(const std::string &dir, const char *templateName,
                               size_t size, std::string &outPath) {
    std::string tmpl = dir + templateName;
    std::vector<char> nameBuf(tmpl.begin(), tmpl.end());
    nameBuf.push_back('\0');

    /*
     * mkstemp() gives a unique, 0600, caller-owned file. That removes the
     * truncate/symlink-clobber risk of a fixed name in a world-writable
     * directory and lets concurrent kfdtest instances coexist.
     */
    int fd = mkstemp(nameBuf.data());
    if (fd < 0) {
        LOG() << "Failed to create temp file in " << dir
              << " (" << strerror(errno) << ")" << std::endl;
        return -1;
    }
    outPath = nameBuf.data();

    /*
     * mkstemp() cannot request O_DIRECT, so reopen the file we just created.
     * Some filesystems (e.g. tmpfs-backed /dev/shm) reject O_DIRECT with
     * EINVAL; fall back to the buffered descriptor so tests that only need a
     * file to exist still run. The kernel decides whether the backing device
     * actually supports the transfer.
     */
    int directFd = open(outPath.c_str(), O_RDWR | O_DIRECT);
    if (directFd >= 0) {
        close(fd);
        fd = directFd;
    } else if (errno == EINVAL) {
        LOG() << "O_DIRECT unavailable on " << dir << ", using buffered I/O" << std::endl;
    } else {
        LOG() << "Failed to reopen " << outPath << " (" << strerror(errno) << ")" << std::endl;
        close(fd);
        unlink(outPath.c_str());
        outPath.clear();
        return -1;
    }

    if (ftruncate(fd, size) < 0) {
        LOG() << "Failed to set file size to " << size << " (" << strerror(errno) << ")" << std::endl;
        close(fd);
        unlink(outPath.c_str());
        outPath.clear();
        return -1;
    }

    lseek(fd, 0, SEEK_SET);
    return fd;
}

void KFDAISTest::deleteTestFiles() {
    if (m_fd >= 0) {
        close(m_fd);
        m_fd = -1;
    }
    if (m_fdOut >= 0) {
        close(m_fdOut);
        m_fdOut = -1;
    }
    if (!m_testFilePath.empty()) {
        unlink(m_testFilePath.c_str());
        m_testFilePath.clear();
    }
    if (!m_testFilePathOut.empty()) {
        unlink(m_testFilePathOut.c_str());
        m_testFilePathOut.clear();
    }
}

int KFDAISTest::fillFileWithPatterns(int fd, size_t fileSize, size_t chunkSize,
                                     std::vector<uint32_t> &patterns) {
    const uint32_t START_PATTERN = 0x11111111;
    size_t numBuffers = fileSize / chunkSize;

    patterns.clear();
    patterns.resize(numBuffers);

    /* O_DIRECT write buffer must be page-aligned (a std::vector is not). */
    auto buffer = allocAligned(chunkSize);
    if (!buffer) {
        LOG() << "Failed to allocate aligned write buffer" << std::endl;
        return -1;
    }
    uint32_t *words = static_cast<uint32_t *>(buffer.get());
    const size_t nWords = chunkSize / sizeof(uint32_t);

    lseek(fd, 0, SEEK_SET);

    for (size_t i = 0; i < numBuffers; i++) {
        patterns[i] = START_PATTERN + static_cast<uint32_t>(i);
        std::fill(words, words + nWords, patterns[i]);

        ssize_t written = write(fd, buffer.get(), chunkSize);
        if (written != (ssize_t)chunkSize) {
            LOG() << "Failed to write buffer " << i << " (" << strerror(errno) << ")" << std::endl;
            return -1;
        }
    }

    fsync(fd);
    lseek(fd, 0, SEEK_SET);
    return 0;
}

int KFDAISTest::checkBuffersWithPatterns(const std::vector<void*> &bufs, size_t bufferSize,
                                          const std::vector<uint32_t> &patterns) {
    if (bufs.size() != patterns.size()) {
        LOG() << "Buffer/pattern count mismatch: " << bufs.size() << " vs " << patterns.size() << std::endl;
        return -1;
    }

    const size_t nWords = bufferSize / sizeof(uint32_t);
    std::vector<uint32_t> expected(nWords);

    for (size_t i = 0; i < bufs.size(); i++) {
        if (bufs[i] == NULL)
            continue;

        std::fill(expected.begin(), expected.end(), patterns[i]);

        if (memcmp(bufs[i], expected.data(), bufferSize) != 0) {
            const uint32_t *words = (const uint32_t *)bufs[i];
            size_t badWord = 0;
            while (badWord < nWords && words[badWord] == patterns[i])
                badWord++;
            LOG() << "Buffer " << i << " mismatch at word " << badWord
                  << ": expected 0x" << std::hex << patterns[i]
                  << " got 0x" << words[badWord] << std::dec << std::endl;
            return -1;
        }
    }

    return 0;
}

int KFDAISTest::allocVRAMBuffers(int gpuNode, int numBuffers, size_t bufferSize,
                                  std::vector<void*> &bufs) {
    HsaMemFlags flags = {0};
    HSAKMT_STATUS ret;

    flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
    flags.ui32.HostAccess = 1;
    flags.ui32.NonPaged = 1;

    bufs.clear();
    bufs.resize(numBuffers, NULL);

    for (int i = 0; i < numBuffers; i++) {
        ret = hsaKmtAllocMemory(gpuNode, bufferSize, flags, &bufs[i]);
        if (ret != HSAKMT_STATUS_SUCCESS) {
            LOG() << "Failed to allocate VRAM buffer " << i << std::endl;
            freeVRAMBuffers(bufs, bufferSize);
            return -1;
        }

        if (hsakmt_is_dgpu()) {
            HSAuint32 node = static_cast<HSAuint32>(gpuNode);
            ret = hsaKmtMapMemoryToGPUNodes(bufs[i], bufferSize, NULL,
                       flags, 1, &node);
            if (ret != HSAKMT_STATUS_SUCCESS) {
                LOG() << "Failed to map VRAM buffer " << i << " to GPU" << std::endl;
                freeVRAMBuffers(bufs, bufferSize);
                return -1;
            }
        }

        memset(bufs[i], 0, bufferSize);
    }

    return 0;
}

void KFDAISTest::freeVRAMBuffers(std::vector<void*> &bufs, size_t bufferSize) {
    for (size_t i = 0; i < bufs.size(); i++) {
        if (bufs[i] != NULL) {
            if (hsakmt_is_dgpu())
                hsaKmtUnmapMemoryToGPU(bufs[i]);
            hsaKmtFreeMemory(bufs[i], bufferSize);
            bufs[i] = NULL;
        }
    }
    bufs.clear();
}

int KFDAISTest::aisReadWrite(void *buf, size_t size, int fd, off_t fileOffset,
                              HsaAisFlags flags, HSAuint64 *sizeCopied,
                              bool quiet) {
    HSAuint64 copied = 0;
    HSAint32 status = 0;

    /*
     * The thunk collapses every ioctl failure into HSAKMT_STATUS_ERROR, so
     * errno is the only way to tell "AIS is not initialized for this device"
     * (-ENODEV, see kfd_chardev.c) apart from a genuine transfer failure.
     * hsakmt_ioctl() only retries on EINTR/EAGAIN, so errno from the final
     * ioctl survives.
     */
    errno = 0;
    HSAKMT_STATUS ret = hsaKmtAisReadWriteFile(buf, size, fd, fileOffset,
                                                flags, &copied, &status);
    int savedErrno = errno;

    if (sizeCopied)
        *sizeCopied = copied;

    if (ret != HSAKMT_STATUS_SUCCESS) {
        if (savedErrno == ENODEV)
            return AIS_UNSUPPORTED;
        if (!quiet)
            LOG() << "hsaKmtAisReadWriteFile failed: ret=" << ret
                  << " status=" << status
                  << (savedErrno ? " errno=" + std::to_string(savedErrno) +
                                   " (" + strerror(savedErrno) + ")"
                                 : std::string())
                  << std::endl;
        return AIS_FAILED;
    }

    if (status != 0) {
        if (!quiet)
            LOG() << "AIS operation returned error status: " << status << std::endl;
        return AIS_FAILED;
    }

    if (copied != size) {
        if (!quiet)
            LOG() << "Size mismatch: requested " << size << " copied " << copied << std::endl;
        return AIS_FAILED;
    }

    return AIS_OK;
}

/*
 * Test 1: AISGracefulFailureTest
 *
 * Exercises the AIS path against a tmpfs-backed file (/dev/shm). tmpfs has
 * no backing block or network device, so kfd_ais_rw_file()'s storage-type
 * dispatch falls through to the AIS_STORAGE_UNKNOWN case ("AIS: unsupported
 * storage type") and returns -ENODEV -- the same errno userspace sees when
 * AIS isn't initialized at all for the device (kfd_chardev.c), so this is
 * reported via the same AIS_UNSUPPORTED/SKIP path as isAISSupported()==false.
 * Confirmed against the shipped amd-staging-dkms-7.1.1 kfd_ais.c on real
 * MI300X hardware (AILIKFD-831, 2026-07-27).
 *
 * This test deliberately does NOT assert that the transfer fails: whether a
 * given filesystem is usable by AIS is not part of the API contract, and
 * asserting it makes the test fragile. What is asserted is that the API
 * returns a well-formed result without crashing or hanging, and that if it
 * does succeed the data actually landed correctly.
 */
TEST_F(KFDAISTest, AISGracefulFailureTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_GE(m_gpuNode, 0) << "Failed to get default GPU node";

    if (!isAISSupported(m_gpuNode)) {
        LOG() << "SKIP: AIS is not initialized on GPU node " << m_gpuNode << std::endl;
        return;
    }

    /*
     * Prefer a tmpfs so the no-direct-I/O path is exercised; fall back to
     * /tmp where /dev/shm is unavailable or unusable (some containers).
     */
    struct stat dirSt;
    m_testDir = (stat("/dev/shm", &dirSt) == 0 && S_ISDIR(dirSt.st_mode))
                    ? "/dev/shm/" : "/tmp/";

    m_fd = createTestFile(m_testDir, TEST_FILE_TEMPLATE, FILE_SIZE_BASIC, m_testFilePath);
    ASSERT_GE(m_fd, 0) << "Failed to create test file";

    m_isOnNVME = (checkIfFileIsOnNVME(m_fd) == 0);

    ASSERT_EQ(fillFileWithPatterns(m_fd, FILE_SIZE_BASIC, BUFFER_SIZE, m_patterns), 0);

    m_bufSize = BUFFER_SIZE;
    ASSERT_EQ(allocVRAMBuffers(m_gpuNode, 1, BUFFER_SIZE, m_pBufs), 0);

    int result = aisReadWrite(m_pBufs[0], BUFFER_SIZE, m_fd, 0, HSA_AIS_READ, NULL);

    if (result == AIS_UNSUPPORTED) {
        LOG() << "SKIP: kernel reports AIS unavailable (ENODEV) on node "
              << m_gpuNode << std::endl;
        return;
    }

    if (result == AIS_OK) {
        /* It succeeded, so the data must be correct. */
        std::vector<void*> oneBuf(1, m_pBufs[0]);
        std::vector<uint32_t> onePattern(1, m_patterns[0]);
        EXPECT_EQ(checkBuffersWithPatterns(oneBuf, BUFFER_SIZE, onePattern), 0)
            << "AIS reported success but the data did not match";
        LOG() << "AIS transfer succeeded on " << m_testDir
              << " (NVME=" << m_isOnNVME << ")" << std::endl;
    } else {
        LOG() << "AIS transfer rejected on " << m_testDir
              << " - handled gracefully, no crash" << std::endl;
    }

    TEST_END
}

/*
 * Test 2: AISUnregisteredMemoryTest
 *
 * Negative test - passes a host pointer that was never registered with the
 * thunk's memory manager. The thunk must reject it via the address-range
 * guard in ais.c (hsakmt_fmm_get_handle -> HSAKMT_STATUS_INVALID_PARAMETER)
 * and must not crash.
 *
 * This guard runs entirely in userspace - hsakmt_fmm_get_handle() fails and
 * ais.c returns HSAKMT_STATUS_INVALID_PARAMETER before issuing any ioctl - so
 * the check is independent of ASIC and of whether the kernel initialized AIS.
 * It is therefore deliberately NOT gated on isAISSupported(): this is the one
 * AIS test that provides coverage on every machine.
 */
TEST_F(KFDAISTest, AISUnregisteredMemoryTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_GE(m_gpuNode, 0) << "Failed to get default GPU node";

    m_testDir = "/tmp/";

    m_fd = createTestFile(m_testDir, TEST_FILE_TEMPLATE, FILE_SIZE_BASIC, m_testFilePath);
    ASSERT_GE(m_fd, 0) << "Failed to create test file";

    std::vector<uint8_t> unregistered(BUFFER_SIZE, 0);

    int result = aisReadWrite(unregistered.data(), BUFFER_SIZE, m_fd, 0,
                              HSA_AIS_READ, NULL);

    EXPECT_EQ(result, AIS_FAILED) << "AIS must reject an unregistered memory address";
    LOG() << "SUCCESS: AIS rejected unregistered memory without crashing" << std::endl;

    TEST_END
}

/*
 * Test 3: AISReadTest
 *
 * Basic AIS read test - reads file contents directly into VRAM
 * and validates the data patterns.
 */
TEST_F(KFDAISTest, AISReadTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_GE(m_gpuNode, 0) << "Failed to get default GPU node";

    if (!isAISSupported(m_gpuNode)) {
        LOG() << "SKIP: AIS is not initialized on GPU node " << m_gpuNode << std::endl;
        return;
    }

    if (!findNVMEPath(m_testDir)) {
        LOG() << "SKIP: no NVME-backed directory available for AIS transfers" << std::endl;
        return;
    }

    m_fd = createTestFile(m_testDir, TEST_FILE_TEMPLATE, FILE_SIZE_BASIC, m_testFilePath);
    ASSERT_GE(m_fd, 0) << "Failed to create test file";

    m_isOnNVME = (checkIfFileIsOnNVME(m_fd) == 0);

    ASSERT_EQ(fillFileWithPatterns(m_fd, FILE_SIZE_BASIC, BUFFER_SIZE, m_patterns), 0);

    m_bufSize = BUFFER_SIZE;
    ASSERT_EQ(allocVRAMBuffers(m_gpuNode, N_BUFFERS_BASIC, BUFFER_SIZE, m_pBufs), 0);

    for (int i = 0; i < N_BUFFERS_BASIC; i++) {
        int rc = aisReadWrite(m_pBufs[i], BUFFER_SIZE, m_fd,
                              i * BUFFER_SIZE, HSA_AIS_READ, NULL);
        if (rc == AIS_UNSUPPORTED) {
            LOG() << "SKIP: kernel reports AIS unavailable (ENODEV) on node "
                  << m_gpuNode << std::endl;
            return;
        }
        ASSERT_EQ(rc, AIS_OK) << "AIS read failed for buffer " << i;
    }

    ASSERT_EQ(checkBuffersWithPatterns(m_pBufs, BUFFER_SIZE, m_patterns), 0)
        << "Pattern validation failed";

    LOG() << "AISReadTest PASSED: " << N_BUFFERS_BASIC << " buffers validated" << std::endl;

    TEST_END
}

/*
 * Test 4: AISWriteTest
 *
 * Basic AIS write test - writes VRAM contents directly to a file
 * and validates the data by reading it back.
 */
TEST_F(KFDAISTest, AISWriteTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_GE(m_gpuNode, 0) << "Failed to get default GPU node";

    if (!isAISSupported(m_gpuNode)) {
        LOG() << "SKIP: AIS is not initialized on GPU node " << m_gpuNode << std::endl;
        return;
    }

    if (!findNVMEPath(m_testDir)) {
        LOG() << "SKIP: no NVME-backed directory available for AIS transfers" << std::endl;
        return;
    }

    m_fd = createTestFile(m_testDir, TEST_FILE_TEMPLATE, FILE_SIZE_BASIC, m_testFilePath);
    ASSERT_GE(m_fd, 0) << "Failed to create input test file";

    m_isOnNVME = (checkIfFileIsOnNVME(m_fd) == 0);

    ASSERT_EQ(fillFileWithPatterns(m_fd, FILE_SIZE_BASIC, BUFFER_SIZE, m_patterns), 0);

    m_bufSize = BUFFER_SIZE;
    ASSERT_EQ(allocVRAMBuffers(m_gpuNode, N_BUFFERS_BASIC, BUFFER_SIZE, m_pBufs), 0);

    for (int i = 0; i < N_BUFFERS_BASIC; i++) {
        int rc = aisReadWrite(m_pBufs[i], BUFFER_SIZE, m_fd,
                              i * BUFFER_SIZE, HSA_AIS_READ, NULL);
        if (rc == AIS_UNSUPPORTED) {
            LOG() << "SKIP: kernel reports AIS unavailable (ENODEV) on node "
                  << m_gpuNode << std::endl;
            return;
        }
        ASSERT_EQ(rc, AIS_OK) << "AIS read failed for buffer " << i;
    }

    ASSERT_EQ(checkBuffersWithPatterns(m_pBufs, BUFFER_SIZE, m_patterns), 0)
        << "Pattern validation after read failed";

    m_fdOut = createTestFile(m_testDir, TEST_FILE_OUT_TEMPLATE, FILE_SIZE_BASIC,
                             m_testFilePathOut);
    ASSERT_GE(m_fdOut, 0) << "Failed to create output test file";

    for (int i = 0; i < N_BUFFERS_BASIC; i++) {
        ASSERT_EQ(aisReadWrite(m_pBufs[i], BUFFER_SIZE, m_fdOut,
                               i * BUFFER_SIZE, HSA_AIS_WRITE, NULL), AIS_OK)
            << "AIS write failed for buffer " << i;
    }

    /*
     * No fsync() needed: kfd_ais_rw_file() uses a sync kiocb
     * (init_sync_kiocb) and mirrors the fd's own O_DIRECT flag, so the write
     * has completed and this readback shares the fd's cache mode.
     */
    const size_t nWords = BUFFER_SIZE / sizeof(uint32_t);
    /* O_DIRECT readback buffer must be page-aligned. */
    auto readBuf = allocAligned(BUFFER_SIZE);
    ASSERT_TRUE(readBuf != nullptr) << "Failed to allocate aligned readback buffer";
    std::vector<uint32_t> expected(nWords);
    lseek(m_fdOut, 0, SEEK_SET);

    for (int i = 0; i < N_BUFFERS_BASIC; i++) {
        ssize_t bytesRead = read(m_fdOut, readBuf.get(), BUFFER_SIZE);
        ASSERT_EQ(bytesRead, (ssize_t)BUFFER_SIZE) << "Failed to read back buffer " << i;

        std::fill(expected.begin(), expected.end(), m_patterns[i]);
        ASSERT_EQ(memcmp(readBuf.get(), expected.data(), BUFFER_SIZE), 0)
            << "Output file pattern mismatch at buffer " << i;
    }

    LOG() << "AISWriteTest PASSED: " << N_BUFFERS_BASIC << " buffers written and validated" << std::endl;

    TEST_END
}

/*
 * Test 5: AISMultiThreadedTest
 *
 * Multi-threaded AIS test - threads each read a different section of a
 * large file into separate VRAM buffers concurrently, then write them
 * back to a new file. Thread count is configurable via --ais_threads.
 */
TEST_F(KFDAISTest, AISMultiThreadedTest) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_GE(m_gpuNode, 0) << "Failed to get default GPU node";

    if (!isAISSupported(m_gpuNode)) {
        LOG() << "SKIP: AIS is not initialized on GPU node " << m_gpuNode << std::endl;
        return;
    }

    int numThreads = g_AisThreads;
    ASSERT_GT(numThreads, 0) << "--ais_threads must be positive";

    if (!findNVMEPath(m_testDir)) {
        LOG() << "SKIP: no NVME-backed directory available for AIS transfers" << std::endl;
        return;
    }

    /*
     * Total workload defaults to 1GB (--ais_size_mb overrides). It is then
     * clamped down to fit the two resources AIS actually consumes, so the test
     * never overcommits regardless of the flag value or the machine:
     *   - Free disk on the test dir: input + output files each need `total`
     *     bytes, so require 2x total plus headroom.
     *   - VRAM: all per-thread buffers are resident VRAM at once; keep under
     *     1/4 of VRAM.
     * The result is split evenly across threads, 4KB-aligned for O_DIRECT and
     * never smaller than BUFFER_SIZE.
     */
    size_t totalSize = (size_t)g_AisSizeMB * 1024 * 1024;

    HSAuint64 vramSize = GetVramSize(m_gpuNode);
    ASSERT_GT(vramSize, 0u) << "Failed to query VRAM size";
    totalSize = std::min<uint64_t>(totalSize, vramSize / 4);

    struct statvfs vfs;
    if (statvfs(m_testDir.c_str(), &vfs) == 0) {
        uint64_t freeBytes = (uint64_t)vfs.f_bavail * vfs.f_frsize;
        /* Need input + output (2x); leave 1/8 of free space as headroom. */
        uint64_t diskBudget = (freeBytes - freeBytes / 8) / 2;
        totalSize = std::min<uint64_t>(totalSize, diskBudget);
    } else {
        /*
         * Without a free-space figure we cannot bound the two test files, and
         * filling the device would affect everything else on the machine.
         */
        LOG() << "SKIP: statvfs(" << m_testDir << ") failed ("
              << strerror(errno) << "); cannot bound disk usage" << std::endl;
        return;
    }

    /*
     * If the budgets can't cover even one BUFFER_SIZE buffer per thread, skip
     * rather than forcing perBuf up to BUFFER_SIZE, which would overcommit the
     * disk/VRAM budget we just computed.
     */
    size_t minTotal = (size_t)BUFFER_SIZE * (size_t)numThreads;
    if (totalSize < minTotal) {
        LOG() << "SKIP: disk/VRAM budget too small for "
              << numThreads << " threads x " << (BUFFER_SIZE >> 10) << "KB" << std::endl;
        return;
    }

    /*
     * BUFFER_SIZE is a multiple of MT_ALIGN and totalSize >= BUFFER_SIZE *
     * numThreads (checked above), so aligning down cannot drop perBuf below
     * BUFFER_SIZE.
     */
    size_t perBuf = (totalSize / numThreads) & ~((size_t)MT_ALIGN - 1);
    size_t fileSize = perBuf * numThreads;
    m_bufSize = perBuf;

    LOG() << std::dec << "Running AISMultiThreadedTest with " << numThreads
          << " threads, " << (perBuf >> 10) << "KB/thread, "
          << (fileSize >> 20) << "MB total" << std::endl;

    m_fd = createTestFile(m_testDir, TEST_FILE_TEMPLATE, fileSize, m_testFilePath);
    ASSERT_GE(m_fd, 0) << "Failed to create test file";

    m_isOnNVME = (checkIfFileIsOnNVME(m_fd) == 0);

    ASSERT_EQ(fillFileWithPatterns(m_fd, fileSize, perBuf, m_patterns), 0);

    ASSERT_EQ(allocVRAMBuffers(m_gpuNode, numThreads, perBuf, m_pBufs), 0);

    std::vector<std::thread> readThreads;
    std::vector<int> readResults(numThreads, 0);

    /*
     * Worker threads run with quiet=true: LOG() is a shared stream and is not
     * safe to interleave across threads, so failures are reported here after
     * every thread has been joined.
     */
    for (int i = 0; i < numThreads; i++) {
        readThreads.emplace_back([this, i, perBuf, &readResults]() {
            readResults[i] = aisReadWrite(m_pBufs[i], perBuf, m_fd,
                                          (off_t)i * perBuf, HSA_AIS_READ,
                                          NULL, true /* quiet */);
        });
    }

    for (auto &t : readThreads) {
        t.join();
    }

    for (int i = 0; i < numThreads; i++) {
        if (readResults[i] == AIS_UNSUPPORTED) {
            LOG() << "SKIP: kernel reports AIS unavailable (ENODEV) on node "
                  << m_gpuNode << std::endl;
            return;
        }
    }

    for (int i = 0; i < numThreads; i++) {
        ASSERT_EQ(readResults[i], AIS_OK) << "Thread " << i << " AIS read failed";
    }

    ASSERT_EQ(checkBuffersWithPatterns(m_pBufs, perBuf, m_patterns), 0)
        << "Pattern validation failed after multi-threaded read";

    LOG() << "Multi-threaded read complete, starting multi-threaded write" << std::endl;

    m_fdOut = createTestFile(m_testDir, TEST_FILE_OUT_TEMPLATE, fileSize,
                             m_testFilePathOut);
    ASSERT_GE(m_fdOut, 0) << "Failed to create output test file";

    std::vector<std::thread> writeThreads;
    std::vector<int> writeResults(numThreads, 0);

    for (int i = 0; i < numThreads; i++) {
        writeThreads.emplace_back([this, i, perBuf, &writeResults]() {
            writeResults[i] = aisReadWrite(m_pBufs[i], perBuf, m_fdOut,
                                           (off_t)i * perBuf, HSA_AIS_WRITE,
                                           NULL, true /* quiet */);
        });
    }

    for (auto &t : writeThreads) {
        t.join();
    }

    for (int i = 0; i < numThreads; i++) {
        ASSERT_EQ(writeResults[i], AIS_OK) << "Thread " << i << " AIS write failed";
    }

    /* Readback needs no fsync() for the same reason as AISWriteTest. */
    const size_t nWords = perBuf / sizeof(uint32_t);
    /* O_DIRECT readback buffer must be page-aligned. */
    auto readBuf = allocAligned(perBuf);
    ASSERT_TRUE(readBuf != nullptr) << "Failed to allocate aligned readback buffer";
    std::vector<uint32_t> expected(nWords);
    lseek(m_fdOut, 0, SEEK_SET);

    for (int i = 0; i < numThreads; i++) {
        ssize_t bytesRead = read(m_fdOut, readBuf.get(), perBuf);
        ASSERT_EQ(bytesRead, (ssize_t)perBuf) << "Failed to read back buffer " << i;

        std::fill(expected.begin(), expected.end(), m_patterns[i]);
        ASSERT_EQ(memcmp(readBuf.get(), expected.data(), perBuf), 0)
            << "Output file pattern mismatch at buffer " << i;
    }

    LOG() << "AISMultiThreadedTest PASSED: " << numThreads
          << " threads successfully performed concurrent AIS read/write" << std::endl;

    TEST_END
}
