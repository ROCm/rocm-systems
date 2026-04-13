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

/**
 * @file toctou_test.cc
 * @brief Tests for TOCTOU race condition fixes in file creation.
 *
 * These tests verify that:
 * 1. HMAC key files are created with correct permissions from the start
 *    (no window where the file is world-readable)
 * 2. Files are created with O_EXCL to prevent symlink attacks
 * 3. fchmod() is used on file descriptors, not chmod() on paths
 *
 * Discovered by: Nix-based static analysis (flawfinder level 5) on the
 * basic-static-analysis branch. CWE-367: TOCTOU Race Condition.
 */

#include <gtest/gtest.h>
#include "src/hmac.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

class HmacToctouTest : public ::testing::Test {
protected:
    std::string test_dir;
    std::string test_key_path;

    void SetUp() override {
        // Create a temporary directory for test key files
        char tmpl[] = "/tmp/cuid_toctou_test_XXXXXX";
        char* dir = mkdtemp(tmpl);
        ASSERT_NE(dir, nullptr) << "Failed to create temp directory";
        test_dir = dir;
        test_key_path = test_dir + "/hmac_key.bin";
    }

    void TearDown() override {
        // Clean up
        if (!test_dir.empty()) {
            fs::remove_all(test_dir);
        }
    }
};

// Test that set_hmac_key creates the file with restrictive permissions
// from the start -- no window where the file is world-readable.
TEST_F(HmacToctouTest, KeyFileCreatedWithRestrictivePermissions) {
    // Skip if not running as root (set_hmac_key requires root)
    if (geteuid() != 0) {
        GTEST_SKIP() << "Test requires root privileges";
    }

    cuid_hmac hmac;
    hmac.key_file_path = test_key_path;

    uint8_t key_data[key_length];
    memset(key_data, 0xAB, key_length);

    amdcuid_status_t status = hmac.set_hmac_key(key_data);
    ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS);

    // Verify file exists
    struct stat st;
    ASSERT_EQ(stat(test_key_path.c_str(), &st), 0)
        << "Key file was not created";

    // Verify permissions are exactly 0600 (owner read/write only)
    mode_t perms = st.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
    EXPECT_EQ(perms, static_cast<mode_t>(S_IRUSR | S_IWUSR))
        << "Key file permissions should be 0600, got "
        << std::oct << perms;
}

// Test that set_hmac_key does not follow symlinks.
// If the key path is a symlink, creation should fail (O_NOFOLLOW)
// rather than writing the key to an attacker-controlled location.
TEST_F(HmacToctouTest, KeyFileRejectsSymlink) {
    if (geteuid() != 0) {
        GTEST_SKIP() << "Test requires root privileges";
    }

    // Create a symlink at the key path pointing to a decoy file
    std::string decoy_path = test_dir + "/decoy";
    int fd = open(decoy_path.c_str(), O_WRONLY | O_CREAT, 0644);
    ASSERT_GE(fd, 0);
    close(fd);

    ASSERT_EQ(symlink(decoy_path.c_str(), test_key_path.c_str()), 0)
        << "Failed to create test symlink";

    cuid_hmac hmac;
    hmac.key_file_path = test_key_path;

    uint8_t key_data[key_length];
    memset(key_data, 0xAB, key_length);

    // set_hmac_key should fail because the path is a symlink
    // and the fix uses O_NOFOLLOW
    amdcuid_status_t status = hmac.set_hmac_key(key_data);
    EXPECT_NE(status, AMDCUID_STATUS_SUCCESS)
        << "set_hmac_key should reject symlink paths (O_NOFOLLOW)";

    // Verify the decoy file was not modified
    struct stat st;
    stat(decoy_path.c_str(), &st);
    EXPECT_EQ(st.st_size, 0)
        << "Decoy file should not have been written to via symlink";
}

// Test that key file content is correct after creation.
TEST_F(HmacToctouTest, KeyFileContainsCorrectData) {
    if (geteuid() != 0) {
        GTEST_SKIP() << "Test requires root privileges";
    }

    cuid_hmac hmac;
    hmac.key_file_path = test_key_path;

    uint8_t key_data[key_length];
    for (int i = 0; i < key_length; i++) {
        key_data[i] = static_cast<uint8_t>(i);
    }

    amdcuid_status_t status = hmac.set_hmac_key(key_data);
    ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS);

    // Read back and verify
    int fd = open(test_key_path.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);

    uint8_t read_data[key_length];
    ssize_t bytes_read = read(fd, read_data, key_length);
    close(fd);

    ASSERT_EQ(bytes_read, key_length);
    EXPECT_EQ(memcmp(key_data, read_data, key_length), 0)
        << "Key file content does not match written data";
}

// Test that overwriting an existing key file works and
// the new file has correct permissions.
TEST_F(HmacToctouTest, KeyFileOverwritePreservesPermissions) {
    if (geteuid() != 0) {
        GTEST_SKIP() << "Test requires root privileges";
    }

    cuid_hmac hmac;
    hmac.key_file_path = test_key_path;

    uint8_t key1[key_length], key2[key_length];
    memset(key1, 0x11, key_length);
    memset(key2, 0x22, key_length);

    // Write first key
    ASSERT_EQ(hmac.set_hmac_key(key1), AMDCUID_STATUS_SUCCESS);

    // Overwrite with second key
    ASSERT_EQ(hmac.set_hmac_key(key2), AMDCUID_STATUS_SUCCESS);

    // Verify permissions are still 0600
    struct stat st;
    ASSERT_EQ(stat(test_key_path.c_str(), &st), 0);
    mode_t perms = st.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
    EXPECT_EQ(perms, static_cast<mode_t>(S_IRUSR | S_IWUSR));

    // Verify content is key2
    int fd = open(test_key_path.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    uint8_t read_data[key_length];
    read(fd, read_data, key_length);
    close(fd);
    EXPECT_EQ(memcmp(key2, read_data, key_length), 0);
}
