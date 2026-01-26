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

void test_get_num_handles() {
    uint32_t count = 0;
    amdcuid_status_t status = amdcuid_get_num_handles(&count);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_GT(count, 0);
}

void test_get_all_handles() {
    uint32_t count = 0;

    amdcuid_status_t status = amdcuid_get_num_handles(&count);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    amdcuid_id_t* handles = new amdcuid_id_t[count];

    status = amdcuid_get_all_handles(handles, count);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    for (uint32_t i = 0; i < count; ++i) {
        const char* id_str = amdcuid_id_to_string(handles[i]);
        EXPECT_NE(id_str, nullptr);
        EXPECT_GT(strlen(id_str), 0);
    }

    delete[] handles;
}

void test_query_device_property() {

    uint32_t count = 0;
    amdcuid_status_t status = amdcuid_get_num_handles(&count);
    EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);

    amdcuid_id_t* handles = new amdcuid_id_t[count];
    status = amdcuid_get_all_handles(handles, count);
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

TEST(CUIDHandleTest, GetNumHandles) {
    test_get_num_handles();
}

TEST(CUIDHandleTest, GetAllHandles) {
    test_get_all_handles();
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