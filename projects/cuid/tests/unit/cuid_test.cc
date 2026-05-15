/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <gtest/gtest.h>
#include "include/amd_cuid.h"
#include "src/cuid_file.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <sys/wait.h>
#include <unistd.h>

// unit test public ABI functions here

// maybe also write tests to reverse CUID into its components like serial number, product family, etc.

void test_get_library_version() {
    uint32_t major = 0, minor = 0, patch = 0;
    amdcuid_get_library_version(&major, &minor, &patch);
    EXPECT_EQ(major, AMDCUID_LIB_VERSION_MAJOR);
    EXPECT_EQ(minor, AMDCUID_LIB_VERSION_MINOR);
    EXPECT_EQ(patch, AMDCUID_LIB_VERSION_PATCH);
}

void test_library_version_string() {
    const char* version_str = amdcuid_library_version_to_string();
    char expected_str[16];
    snprintf(expected_str, sizeof(expected_str), "%u.%u.%u",
             AMDCUID_LIB_VERSION_MAJOR,
             AMDCUID_LIB_VERSION_MINOR,
             AMDCUID_LIB_VERSION_PATCH);
    EXPECT_STREQ(version_str, expected_str);
}

void test_status_to_string() {
    amdcuid_status_t statuses[] = {
        AMDCUID_STATUS_SUCCESS,
        AMDCUID_STATUS_FILE_NOT_FOUND,              ///< CUID file not found
        AMDCUID_STATUS_DEVICE_NOT_FOUND,            ///< Device(s) not found
        AMDCUID_STATUS_INVALID_ARGUMENT,            ///< Invalid argument passed to function
        AMDCUID_STATUS_PERMISSION_DENIED,           ///< Insufficient permissions for operation
        AMDCUID_STATUS_UNSUPPORTED,                 ///< Operation or device type not supported on system
        AMDCUID_STATUS_WRONG_DEVICE_TYPE,           ///< Incorrect device type for function
        AMDCUID_STATUS_INSUFFICIENT_SIZE,           ///< Provided buffer or array is too small
        AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND,    ///< Hardware fingerprint could not be found
        AMDCUID_STATUS_KEY_ERROR,                   ///< An error occurred related to the hash key
        AMDCUID_STATUS_HMAC_ERROR,                 ///< An error occurred during HMAC computation
        AMDCUID_STATUS_FILE_ERROR,                 ///< File I/O error occurred when reading or writing the CUID files
        AMDCUID_STATUS_INVALID_FORMAT,             ///< Data format given or read is invalid or malformed
        AMDCUID_STATUS_PCI_ERROR,                  ///< An error occurred while accessing or parsing PCI configuration space
        AMDCUID_STATUS_SMBIOS_ERROR,               ///< An error occurred while accessing or parsing the SMBIOS table
        AMDCUID_STATUS_ACPI_ERROR,                 ///< An error occurred while accessing or parsing the ACPI table
        AMDCUID_STATUS_CPUINFO_ERROR,              ///< An error occurred while accessing or parsing CPUINFO
        AMDCUID_STATUS_IPC_ERROR                   ///< An error occurred during IPC communication with the daemon
    };

    for (auto status : statuses) {
        const char* status_str = amdcuid_status_to_string(status);
        EXPECT_NE(status_str, nullptr);
        EXPECT_GT(strlen(status_str), 0);
    }
}

void test_id_to_string() {
    amdcuid_id_t test_id = {0x1, 0x2, 0x3, 0x4,
                          0x5, 0x6, 0x7, 0x8,
                          0x9, 0xA, 0xB, 0xC,
                          0xD, 0xE, 0xF, 0x0};
    const char* id_str = amdcuid_id_to_string(test_id);
    EXPECT_NE(id_str, nullptr);
    EXPECT_STREQ(id_str, "01020304-0506-0708-090a-0b0c0d0e0f00");
}

void test_get_all_handles() {
    uint32_t count = 1;

    amdcuid_id_t* handles = new amdcuid_id_t[count];

    amdcuid_status_t status = amdcuid_get_all_handles(handles, &count);
    EXPECT_EQ(status, AMDCUID_STATUS_INSUFFICIENT_SIZE);

    handles = new amdcuid_id_t[count];
    status = amdcuid_get_all_handles(handles, &count);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    for (uint32_t i = 0; i < count; ++i) {
        const char* id_str = amdcuid_id_to_string(handles[i]);
        EXPECT_NE(id_str, nullptr);
        EXPECT_GT(strlen(id_str), 0);
    }

    delete[] handles;
}

void test_get_handle_by_bdf() {
    const char* test_bdf = "0000:03:00.0";
    amdcuid_device_type_t device_type = AMDCUID_DEVICE_TYPE_GPU;
    amdcuid_id_t handle;

    amdcuid_status_t status = amdcuid_get_handle_by_dev_path(test_bdf, device_type, &handle);
    if (status == AMDCUID_STATUS_SUCCESS) {
        const char* id_str = amdcuid_id_to_string(handle);
        EXPECT_NE(id_str, nullptr);
        EXPECT_GT(strlen(id_str), 0);
    } else {
        EXPECT_EQ(status, AMDCUID_STATUS_DEVICE_NOT_FOUND);
    }
}

void test_get_handle_by_dev_path() {
    const char* test_dev_path = "/dev/dri/renderD128";
    amdcuid_device_type_t device_type = AMDCUID_DEVICE_TYPE_GPU;
    amdcuid_id_t handle;

    amdcuid_status_t status = amdcuid_get_handle_by_dev_path(test_dev_path, device_type, &handle);
    if (status == AMDCUID_STATUS_SUCCESS) {
        const char* id_str = amdcuid_id_to_string(handle);
        EXPECT_NE(id_str, nullptr);
        EXPECT_GT(strlen(id_str), 0);
    } else {
        EXPECT_EQ(status, AMDCUID_STATUS_DEVICE_NOT_FOUND);
    }
}

void test_get_handle_by_fd() {
    const char* test_dev_path = "/dev/dri/renderD128";
    int fd = open(test_dev_path, O_RDONLY);
    if (fd < 0) {
        GTEST_SKIP() << "Skipping test_get_handle_by_fd: unable to open device file";
    }

    amdcuid_device_type_t device_type = AMDCUID_DEVICE_TYPE_GPU;
    amdcuid_id_t handle;

    amdcuid_status_t status = amdcuid_get_handle_by_fd(fd, device_type, &handle);
    if (status == AMDCUID_STATUS_SUCCESS) {
        const char* id_str = amdcuid_id_to_string(handle);
        EXPECT_NE(id_str, nullptr);
        EXPECT_GT(strlen(id_str), 0);
    } else {
        EXPECT_EQ(status, AMDCUID_STATUS_DEVICE_NOT_FOUND);
    }

    close(fd);
}

void test_refresh() {
    amdcuid_status_t status = amdcuid_refresh();
    if (status == AMDCUID_STATUS_SUCCESS) {
        SUCCEED();
    } else {
        EXPECT_EQ(status, AMDCUID_STATUS_PERMISSION_DENIED);
    }
}

void test_query_device_property() {

    uint32_t count = 100;
    amdcuid_id_t* handles = new amdcuid_id_t[count];
    amdcuid_status_t status = amdcuid_get_all_handles(handles, &count);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    // Query device type for the first handle
    amdcuid_device_type_t device_type;
    uint32_t length = sizeof(device_type);
    status = amdcuid_query_device_property(handles[0], AMDCUID_QUERY_DEVICE_TYPE, &device_type, &length);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(length, sizeof(device_type));

    delete[] handles;
}

void test_set_hash_key() {
    // This test requires elevated permissions; skip if not running as root
    if (geteuid() != 0) {
        GTEST_SKIP() << "Skipping test_set_hash_key: requires elevated permissions";
    }

    const uint8_t test_key[32] = {0};
    amdcuid_status_t status = amdcuid_set_hash_key(test_key);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

}

void test_generate_hash_key() {
    // This test requires elevated permissions; skip if not running as root
    if (geteuid() != 0) {
        GTEST_SKIP() << "Skipping test_generate_hash_key: requires elevated permissions";
    }

    uint8_t generated_key[32];
    amdcuid_status_t status = amdcuid_generate_hash_key(generated_key);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    // Check that the key is not all zeros
    bool all_zeros = true;
    for (size_t i = 0; i < sizeof(generated_key); ++i) {
        if (generated_key[i] != 0) {
            all_zeros = false;
            break;
        }
    }
    EXPECT_FALSE(all_zeros);
}


TEST(CUIDLibraryVersionTest, GetLibraryVersion) {
    test_get_library_version();
}

TEST(CUIDLibraryVersionTest, LibraryVersionString) {
    test_library_version_string();
}

TEST(CUIDStatusTest, StatusToString) {
    test_status_to_string();
}

TEST(CUIDIdTest, IdToString) {
    test_id_to_string();
}

TEST(CUIDHandleTest, GetAllHandles) {
    test_get_all_handles();
}

TEST(CUIDHandleTest, GetHandleByBDF) {
    test_get_handle_by_bdf();
}

TEST(CUIDHandleTest, GetHandleByDevPath) {
    test_get_handle_by_dev_path();
}

TEST(CUIDHandleTest, GetHandleByFD) {
    test_get_handle_by_fd();
}

TEST(CUIDRefreshTest, RefreshDevices) {
    test_refresh();
}

TEST(CUIDQueryTest, QueryDeviceProperty) {
    test_query_device_property();
}

TEST(CUIDHMACTest, SetHashKey) {
    test_set_hash_key();
}

TEST(CUIDHMACTest, GenerateHashKey) {
    test_generate_hash_key();
}

// ============================================================================
// CuidFileLock Tests
// ============================================================================

// Test basic lock acquisition and release
void test_file_lock_basic() {
    const std::string test_file = "/tmp/cuid_test_lock_basic";
    
    // Test exclusive lock
    {
        CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
        EXPECT_FALSE(lock.is_locked());
        EXPECT_TRUE(lock.acquire());
        EXPECT_TRUE(lock.is_locked());
        lock.release();
        EXPECT_FALSE(lock.is_locked());
    }
    
    // Test shared lock
    {
        CuidFileLock lock(test_file, CuidLockType::SHARED);
        EXPECT_FALSE(lock.is_locked());
        EXPECT_TRUE(lock.acquire());
        EXPECT_TRUE(lock.is_locked());
        // Lock should be released by destructor
    }
    
    // Clean up
    unlink((test_file + ".lock").c_str());
}

// Test RAII - lock released on scope exit
void test_file_lock_raii() {
    const std::string test_file = "/tmp/cuid_test_lock_raii";
    
    {
        CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
        EXPECT_TRUE(lock.acquire());
        EXPECT_TRUE(lock.is_locked());
        // Scope ends here - destructor should release lock
    }
    
    // Now we should be able to acquire again immediately
    {
        CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
        EXPECT_TRUE(lock.try_acquire());  // Should succeed immediately
        EXPECT_TRUE(lock.is_locked());
    }
    
    // Clean up
    unlink((test_file + ".lock").c_str());
}

// Test that multiple shared locks can be acquired simultaneously
void test_file_lock_multiple_shared() {
    const std::string test_file = "/tmp/cuid_test_lock_shared";
    
    CuidFileLock lock1(test_file, CuidLockType::SHARED);
    CuidFileLock lock2(test_file, CuidLockType::SHARED);
    
    EXPECT_TRUE(lock1.acquire());
    EXPECT_TRUE(lock2.try_acquire());  // Should succeed - shared locks are compatible
    
    EXPECT_TRUE(lock1.is_locked());
    EXPECT_TRUE(lock2.is_locked());
    
    // Clean up
    unlink((test_file + ".lock").c_str());
}

// Test that exclusive lock blocks other locks (using fork for true multi-process test)
void test_file_lock_exclusive_blocks() {
    const std::string test_file = "/tmp/cuid_test_lock_exclusive";
    
    // Clean up from any previous run
    unlink((test_file + ".lock").c_str());
    
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process: acquire exclusive lock and hold it
        CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
        if (!lock.acquire()) {
            _exit(1);  // Failed to acquire lock
        }
        
        // Hold lock for 500ms
        usleep(500000);
        _exit(0);
    } else if (pid > 0) {
        // Parent process: wait a bit, then try to acquire
        usleep(100000);  // 100ms - let child acquire first
        
        CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
        
        // try_acquire should fail because child holds exclusive lock
        EXPECT_FALSE(lock.try_acquire());
        
        // Wait for child to finish
        int status;
        waitpid(pid, &status, 0);
        EXPECT_TRUE(WIFEXITED(status));
        EXPECT_EQ(WEXITSTATUS(status), 0);
        
        // Now lock should be available
        EXPECT_TRUE(lock.try_acquire());
    } else {
        FAIL() << "fork() failed";
    }
    
    // Clean up
    unlink((test_file + ".lock").c_str());
}

// Test timeout functionality
void test_file_lock_timeout() {
    const std::string test_file = "/tmp/cuid_test_lock_timeout";
    
    // Clean up from any previous run
    unlink((test_file + ".lock").c_str());
    
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process: acquire exclusive lock and hold it for 2 seconds
        CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
        if (!lock.acquire()) {
            _exit(1);
        }
        sleep(2);
        _exit(0);
    } else if (pid > 0) {
        // Parent process: wait a bit, then try to acquire with timeout
        usleep(100000);  // 100ms - let child acquire first
        
        CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
        
        // 1 second timeout should fail (child holds for 2 seconds)
        auto start = std::chrono::steady_clock::now();
        bool acquired = lock.acquire_with_timeout(1);
        auto elapsed = std::chrono::steady_clock::now() - start;
        
        EXPECT_FALSE(acquired);
        // Should have waited approximately 1 second
        EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 900);
        EXPECT_LE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 1500);
        
        // Wait for child to finish
        int status;
        waitpid(pid, &status, 0);
        
        // Now should be able to acquire
        EXPECT_TRUE(lock.acquire_with_timeout(1));
    } else {
        FAIL() << "fork() failed";
    }
    
    // Clean up
    unlink((test_file + ".lock").c_str());
}

// Test timeout with 0 (non-blocking) and -1 (infinite)
void test_file_lock_timeout_special_cases() {
    const std::string test_file = "/tmp/cuid_test_lock_timeout_special";
    
    // Clean up
    unlink((test_file + ".lock").c_str());
    
    // Test timeout=0 (should behave like try_acquire)
    {
        CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
        EXPECT_TRUE(lock.acquire_with_timeout(0));  // Should succeed immediately
        EXPECT_TRUE(lock.is_locked());
    }
    
    // Test timeout=-1 (should behave like acquire - infinite wait)
    {
        CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
        EXPECT_TRUE(lock.acquire_with_timeout(-1));  // Should succeed
        EXPECT_TRUE(lock.is_locked());
    }
    
    // Clean up
    unlink((test_file + ".lock").c_str());
}

TEST(CUIDFileLockTest, BasicLockAcquireRelease) {
    test_file_lock_basic();
}

TEST(CUIDFileLockTest, RAIIAutoRelease) {
    test_file_lock_raii();
}

TEST(CUIDFileLockTest, MultipleSharedLocks) {
    test_file_lock_multiple_shared();
}

TEST(CUIDFileLockTest, ExclusiveLockBlocks) {
    test_file_lock_exclusive_blocks();
}

TEST(CUIDFileLockTest, AcquireWithTimeout) {
    test_file_lock_timeout();
}

TEST(CUIDFileLockTest, TimeoutSpecialCases) {
    test_file_lock_timeout_special_cases();
}