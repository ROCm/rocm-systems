/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "alt_rsmi.h"
#include "common/ProcessIsolatedTestRunner.hpp"

#include <cerrno>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <cstdlib>
#include <limits>
#include <filesystem>

// ============================================================================
// Internal structures and variables from alt_rsmi.cc (TEST USE ONLY)
// ============================================================================
// When alt_rsmi.cc is compiled with ARSMI_TEST_BUILD, internal variables
// have external linkage and can be accessed by test utilities

struct ARSMI_systemNode {
    uint32_t s_node_id = 0;
    uint64_t s_gpu_id = 0;
    uint64_t s_unique_id = 0;
    uint64_t s_location_id = 0;
    uint64_t s_bdf = 0;
    uint64_t s_domain = 0;
    uint8_t  s_bus = 0;
    uint8_t  s_device = 0;
    uint8_t  s_function = 0;
    uint8_t  s_partition_id = 0;
    std::string s_card;
};

// External declarations of internal variables from alt_rsmi.cc
extern thread_local const char *kKFDNodesPathRoot;
extern thread_local int ARSMI_num_devices;
extern thread_local std::vector<ARSMI_systemNode> ARSMI_orderedNodes;
extern thread_local std::vector<std::vector<ARSMI_linkInfo>> ARSMI_orderedLinks;

// ============================================================================
// Test utilities for manipulating alt_rsmi.cc internal state
// ============================================================================
namespace AltRsmiTestUtils {

// Storage for the test path to ensure the pointer remains valid
static std::string sTestNodesPath;

// Set the KFD nodes path for testing
// This redirects file reads to test directories
static void SetNodesPath(const std::string& path) {
    sTestNodesPath = path;
    kKFDNodesPathRoot = sTestNodesPath.c_str();
}

// Reset ARSMI internal state between tests
// This ensures test isolation
static void ResetState() {
    ARSMI_num_devices = -1;
    ARSMI_orderedNodes.clear();
    ARSMI_orderedLinks.clear();
}

// Get current number of devices (for verification)
static int GetNumDevices() {
    return ARSMI_num_devices;
}

} // namespace AltRsmiTestUtils

// Test paths for creating mock KFD filesystem
// Use std::filesystem::temp_directory_path() for portability
static const std::string kTestKFDBasePath =
    std::filesystem::temp_directory_path().string() + "/test_kfd_arsmi";
static const std::string kTestKFDPath =
    std::filesystem::temp_directory_path().string() + "/test_kfd_arsmi/topology/nodes";

namespace RcclUnitTesting {

// Helper functions for creating test filesystem structures
// All file operations are scoped to kTestKFDBasePath for safety
namespace {
  // Internal helper to create directories recursively (operates on absolute paths)
  int createDirectoryImpl(const std::string &path) {
    size_t pos = 0;
    std::string currentPath;

    // Iterate through each component of the path
    while ((pos = path.find('/', pos)) != std::string::npos) {
      currentPath = path.substr(0, pos++);
      if (!currentPath.empty() && mkdir(currentPath.c_str(), 0700) == -1 &&
          errno != EEXIST) {
        return -1; // Return error if directory creation fails
      }
    }

    // Create the final directory
    if (mkdir(path.c_str(), 0700) == -1 && errno != EEXIST) {
      return -1; // Return error if directory creation fails
    }

    return 0; // Success
  }

  // Internal helper to remove a directory recursively (operates on absolute paths)
  int removeDirectoryImpl(const std::string &path) {
    DIR *dir = opendir(path.c_str());
    if (!dir) {
      // ENOENT means directory doesn't exist - not an error during cleanup
      if (errno == ENOENT) {
        return 0;
      }
      std::cerr << "Failed to open directory: " << path << " (errno: " << errno
                << ")" << std::endl;
      return -1;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
      // Skip "." and ".." entries
      if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
        continue;
      }

      std::string fullPath = path + "/" + entry->d_name;

      // Check if the entry is a directory
      struct stat entryStat;
      if (stat(fullPath.c_str(), &entryStat) == -1) {
        std::cerr << "Failed to stat: " << fullPath << " (errno: " << errno
                  << ")" << std::endl;
        closedir(dir);
        return -1;
      }

      if (S_ISDIR(entryStat.st_mode)) {
        // Recursively remove subdirectory
        if (removeDirectoryImpl(fullPath) == -1) {
          closedir(dir);
          return -1;
        }
      } else {
        // Remove file
        if (unlink(fullPath.c_str()) == -1) {
          std::cerr << "Failed to remove file: " << fullPath
                    << " (errno: " << errno << ")" << std::endl;
          closedir(dir);
          return -1;
        }
      }
    }

    closedir(dir);

    // Remove the directory itself
    if (rmdir(path.c_str()) == -1) {
      std::cerr << "Failed to remove directory: " << path
                << " (errno: " << errno << ")" << std::endl;
      return -1;
    }

    return 0; // Success
  }

  // Validate that a path is within the test sandbox
  bool isPathInSandbox(const std::string &path) {
    // Must start with kTestKFDBasePath to be valid
    return path.find(kTestKFDBasePath) == 0;
  }

  // Create directory within test sandbox (relative to kTestKFDPath)
  // Pass "" to create the base kTestKFDPath directory
  int createDirectory(const std::string &relativePath = "") {
    std::string fullPath = relativePath.empty()
        ? std::string(kTestKFDPath)
        : std::string(kTestKFDPath) + "/" + relativePath;

    if (!isPathInSandbox(fullPath)) {
      std::cerr << "error: path " << fullPath << " is outside test sandbox" << std::endl;
      return -1;
    }
    return createDirectoryImpl(fullPath);
  }

  // Remove entire test sandbox directory
  int removeTestSandbox() {
    return removeDirectoryImpl(kTestKFDBasePath);
  }

  // Create a file within test sandbox (relative to kTestKFDPath)
  void createFile(const std::string &relativePath, const std::string &content) {
    std::string fullPath = std::string(kTestKFDPath) + "/" + relativePath;

    if (!isPathInSandbox(fullPath)) {
      std::cerr << "error: path " << fullPath << " is outside test sandbox" << std::endl;
      return;
    }

    std::ofstream file(fullPath);
    if (!file) {
      std::cerr << "Failed to create file: " << fullPath << ", errno: " << errno << std::endl;
      return;
    }
    file << content;
    file.close();
  }

  // Remove a file within test sandbox (relative to kTestKFDPath)
  int removeFile(const std::string &relativePath) {
    std::string fullPath = std::string(kTestKFDPath) + "/" + relativePath;

    if (!isPathInSandbox(fullPath)) {
      std::cerr << "Security error: path " << fullPath << " is outside test sandbox" << std::endl;
      return -1;
    }

    if (unlink(fullPath.c_str()) == -1) {
      std::cerr << "Failed to remove file: " << fullPath << " (errno: " << errno
                << ")" << std::endl;
      return -1; // Return error if file removal fails
    }
    return 0; // Success
  }

  // Function to create the test directory structure and files
  void setupTestFiles() {
    createDirectory();  // Creates kTestKFDPath

    // Create node 0 with valid data
    createDirectory("0");
    createFile("0/gpu_id", "4098\n");
    createFile("0/properties", "unique_id 16336014475442738425\n"
                               "location_id 23552\n"
                               "domain 0\n"
                               "vendor_id 4098\n");

    createDirectory("0/io_links/0");
    createFile("0/io_links/0/properties",
               "type 2\n"
               "version_major 0\n"
               "version_minor 0\n"
               "node_from 0\n"
               "node_to 1\n"
               "weight 21\n"
               "min_latency 0\n"
               "max_latency 0\n"
               "min_bandwidth 0\n"
               "max_bandwidth 64000\n"
               "recommended_transfer_size 0\n"
               "recommended_sdma_engine_id_mask 0\n"
               "flags 0\n");

    createDirectory("0/io_links/1");
    createFile("0/io_links/1/properties",
               "type 11\n"
               "version_major 0\n"
               "version_minor 0\n"
               "node_from 0\n"
               "node_to 1\n"
               "weight 21\n"
               "min_latency 0\n"
               "max_latency 0\n"
               "min_bandwidth 0\n"
               "max_bandwidth 50000\n"
               "recommended_transfer_size 0\n"
               "recommended_sdma_engine_id_mask 0\n"
               "flags 0\n");

    createDirectory("1");
    createFile("1/gpu_id", "4098\n");
    createFile("1/properties", "unique_id 16336014475442738426\n"
                               "location_id 23553\n"
                               "domain 1\n"
                               "vendor_id 4098\n");

    createDirectory("1/io_links/0");
    createFile("1/io_links/0/properties",
               "type 2\n"
               "version_major 0\n"
               "version_minor 0\n"
               "node_from 1\n"
               "node_to 0\n"
               "weight 21\n"
               "min_latency 0\n"
               "max_latency 0\n"
               "min_bandwidth 0\n"
               "max_bandwidth 32000\n"
               "recommended_transfer_size 0\n"
               "recommended_sdma_engine_id_mask 0\n"
               "flags 0\n");

    createDirectory("1/io_links/1");
    createFile("1/io_links/1/properties",
               "type 11\n"
               "version_major 0\n"
               "version_minor 0\n"
               "node_from 1\n"
               "node_to 0\n"
               "weight 21\n"
               "min_latency 0\n"
               "max_latency 0\n"
               "min_bandwidth 0\n"
               "max_bandwidth 50000\n"
               "recommended_transfer_size 0\n"
               "recommended_sdma_engine_id_mask 0\n"
               "flags 0\n");

    uint32_t invalid_dev_id = 9999; // Device ID that doesn't exist
    createDirectory(std::to_string(invalid_dev_id) + "/io_links/");
  }

  // Common setup for all tests
  void setupTestEnvironment() {
    // Reset ARSMI state for test isolation
    AltRsmiTestUtils::ResetState();

    // Redirect kKFDNodesPathRoot to test directory
    AltRsmiTestUtils::SetNodesPath(kTestKFDPath);

    // Create the test directory structure
    setupTestFiles();
  }

  // Common cleanup for all tests (not strictly necessary with process isolation,
  // but good practice to clean up temp files)
  void cleanupTestEnvironment() {
    AltRsmiTestUtils::ResetState();
    removeTestSandbox();
  }

} // anonymous namespace

// Tests using process isolation for complete state isolation
TEST(AltRsmiTest, ARSMIInitDefault) {
  RUN_ISOLATED_TEST(
    "ARSMIInitDefault",
    []() {
      setupTestEnvironment();

      int result = ARSMI_init();
      ASSERT_EQ(result, 0);

      // Verify that devices were discovered
      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_EQ(num_devices, 2);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitMissingIoLinksPropertiesFile) {
  RUN_ISOLATED_TEST(
    "ARSMIInitMissingIoLinksPropertiesFile",
    []() {
      setupTestEnvironment();

      // Remove properties file for io_links
      removeFile("0/io_links/0/properties");
      removeFile("0/io_links/1/properties");

      int result = ARSMI_init();
      ASSERT_EQ(result, 0);

      // Should still initialize successfully even with missing link properties
      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_GT(num_devices, 0);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitMissingNodeToProperty) {
  RUN_ISOLATED_TEST(
    "ARSMIInitMissingNodeToProperty",
    []() {
      setupTestEnvironment();

      createFile("0/io_links/1/properties",
                 "type 2\n"
                 "version_major 0\n"
                 "version_minor 0\n"
                 "node_from 0\n"
                 // "node_to 0\n"  // Missing node_to
                 "weight 21\n"
                 "min_latency 0\n"
                 "max_latency 0\n"
                 "min_bandwidth 0\n"
                 "max_bandwidth 64000\n"
                 "recommended_transfer_size 0\n"
                 "recommended_sdma_engine_id_mask 0\n"
                 "flags 0\n");

      int result = ARSMI_init();
      ASSERT_EQ(result, 0); // Expect success even with missing node_to

      // Verify devices are still initialized
      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_GT(num_devices, 0);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitMissingWeightProperty) {
  RUN_ISOLATED_TEST(
    "ARSMIInitMissingWeightProperty",
    []() {
      setupTestEnvironment();

      createFile("0/io_links/1/properties",
                 "type 2\n"
                 "version_major 0\n"
                 "version_minor 0\n"
                 "node_from 0\n"
                 "node_to 1\n"
                 // "weight 21\n"  // Missing weight
                 "min_latency 0\n"
                 "max_latency 0\n"
                 "min_bandwidth 0\n"
                 "max_bandwidth 64000\n"
                 "recommended_transfer_size 0\n"
                 "recommended_sdma_engine_id_mask 0\n"
                 "flags 0\n");

      int result = ARSMI_init();
      // returns 1 when weight property is missing
      ASSERT_EQ(result, 1);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitMissingTypeProperty) {
  RUN_ISOLATED_TEST(
    "ARSMIInitMissingTypeProperty",
    []() {
      setupTestEnvironment();

      createFile("0/io_links/1/properties",
                 // "type 5\n" // Missing type
                 "version_major 0\n"
                 "version_minor 0\n"
                 "node_from 0\n"
                 "node_to 1\n"
                 "weight 21\n"
                 "min_latency 0\n"
                 "max_latency 0\n"
                 "min_bandwidth 0\n"
                 "max_bandwidth 0\n"
                 "recommended_transfer_size 0\n"
                 "recommended_sdma_engine_id_mask 0\n"
                 "flags 0\n");

      int result = ARSMI_init();
      // returns 1 when type property is missing
      ASSERT_EQ(result, 1);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitTypePCIeProperty) {
  RUN_ISOLATED_TEST(
    "ARSMIInitTypePCIeProperty",
    []() {
      // Create a setup with ONLY PCIe links (type 2) to test PCIe specifically
      removeTestSandbox();
      createDirectory();

      // Create node 0 with PCIe-only link to node 1
      createDirectory("0");
      createFile("0/gpu_id", "4098\n");
      createFile("0/properties", "unique_id 100\n"
                                 "location_id 23552\n"
                                 "domain 0\n"
                                 "vendor_id 4098\n");
      createDirectory("0/io_links/0");
      createFile("0/io_links/0/properties",
                 "type 2\n"  // PCIe type
                 "version_major 0\n"
                 "version_minor 0\n"
                 "node_from 0\n"
                 "node_to 1\n"
                 "weight 21\n"
                 "min_latency 0\n"
                 "max_latency 0\n"
                 "min_bandwidth 0\n"
                 "max_bandwidth 16000\n"
                 "recommended_transfer_size 0\n"
                 "recommended_sdma_engine_id_mask 0\n"
                 "flags 0\n");

      // Create node 1 with PCIe-only link to node 0
      createDirectory("1");
      createFile("1/gpu_id", "4098\n");
      createFile("1/properties", "unique_id 101\n"
                                 "location_id 23553\n"
                                 "domain 1\n"
                                 "vendor_id 4098\n");
      createDirectory("1/io_links/0");
      createFile("1/io_links/0/properties",
                 "type 2\n"  // PCIe type
                 "version_major 0\n"
                 "version_minor 0\n"
                 "node_from 1\n"
                 "node_to 0\n"
                 "weight 21\n"
                 "min_latency 0\n"
                 "max_latency 0\n"
                 "min_bandwidth 0\n"
                 "max_bandwidth 16000\n"
                 "recommended_transfer_size 0\n"
                 "recommended_sdma_engine_id_mask 0\n"
                 "flags 0\n");

      AltRsmiTestUtils::ResetState();
      AltRsmiTestUtils::SetNodesPath(kTestKFDPath);

      int result = ARSMI_init();
      ASSERT_EQ(result, 0);

      // Verify link info is correctly identified as PCIe
      ARSMI_linkInfo info;
      ASSERT_EQ(ARSMI_topo_get_link_info(0, 1, &info), 0);
      ASSERT_EQ(info.type, ARSMI_IOLINK_TYPE_PCIEXPRESS);
      ASSERT_EQ(info.src_node, 0);
      ASSERT_EQ(info.dst_node, 1);
      ASSERT_EQ(info.hops, 2);
      ASSERT_EQ(info.weight, 21);
      ASSERT_EQ(info.max_bandwidth, 16000);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitMissingMinBWProperty) {
  RUN_ISOLATED_TEST(
    "ARSMIInitMissingMinBWProperty",
    []() {
      setupTestEnvironment();

      createFile("0/io_links/1/properties",
                 "type 11\n"
                 "version_major 0\n"
                 "version_minor 0\n"
                 "node_from 0\n"
                 "node_to 1\n"
                 "weight 21\n"
                 "min_latency 0\n"
                 "max_latency 0\n"
                 // "min_bandwidth 0\n"  // Missing min_bandwidth
                 "max_bandwidth 0\n"
                 "recommended_transfer_size 0\n"
                 "recommended_sdma_engine_id_mask 0\n"
                 "flags 0\n");

      int result = ARSMI_init();
      // returns 1 when min_bandwidth property is missing
      ASSERT_EQ(result, 1);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitMissingMaxBWProperty) {
  RUN_ISOLATED_TEST(
    "ARSMIInitMissingMaxBWProperty",
    []() {
      setupTestEnvironment();

      createFile("0/io_links/1/properties",
                 "type 5\n"
                 "version_major 0\n"
                 "version_minor 0\n"
                 "node_from 0\n"
                 "node_to 1\n"
                 "weight 21\n"
                 "min_latency 0\n"
                 "max_latency 0\n"
                 "min_bandwidth 0\n"
                 // "max_bandwidth 0\n"  // Missing max_bandwidth
                 "recommended_transfer_size 0\n"
                 "recommended_sdma_engine_id_mask 0\n"
                 "flags 0\n");

      int result = ARSMI_init();
      // returns 1 when max_bandwidth property is missing
      ASSERT_EQ(result, 1);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIGetNumDevicesUninitialized) {
  RUN_ISOLATED_TEST(
    "ARSMIGetNumDevicesUninitialized",
    []() {
      setupTestEnvironment();

      // Verify ARSMI is uninitialized (ARSMI_num_devices == -1)
      ASSERT_EQ(AltRsmiTestUtils::GetNumDevices(), -1);

      // Don't call ARSMI_init, let ARSMI_get_num_devices initialize
      uint32_t num_devices = 0;

      int result = ARSMI_get_num_devices(&num_devices);

      // Verify that the function initializes successfully
      ASSERT_EQ(result, 0);

      // Verify that auto-initialization occurred (ARSMI_num_devices >= 0)
      ASSERT_GE(AltRsmiTestUtils::GetNumDevices(), 0);

      // Verify that the number of devices is correctly set
      ASSERT_EQ(num_devices, 2);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIDevPciIdGetNullBdfId) {
  RUN_ISOLATED_TEST(
    "ARSMIDevPciIdGetNullBdfId",
    []() {
      setupTestEnvironment();

      uint32_t device_index = 0;
      int result = ARSMI_dev_pci_id_get(device_index, nullptr);

      ASSERT_EQ(result, EINVAL);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIDevPciIdGetValid) {
  RUN_ISOLATED_TEST(
    "ARSMIDevPciIdGetValid",
    []() {
      setupTestEnvironment();

      uint32_t device_index = 0;
      uint64_t bdfid = 0;

      int result = ARSMI_dev_pci_id_get(device_index, &bdfid);

      // Verify that the function succeeds
      ASSERT_EQ(result, 0);
      // BDF ID should be non-zero for valid devices
      ASSERT_NE(bdfid, 0);

      cleanupTestEnvironment();
    }
  );
}

// Tests covering invalid file/directory scenarios through public API
TEST(AltRsmiTest, ARSMIInitWithInvalidGpuIdData) {
  RUN_ISOLATED_TEST(
    "ARSMIInitWithInvalidGpuIdData",
    []() {
      // Create a gpu_id file with invalid (non-numeric) data
      removeTestSandbox();
      createDirectory();
      createDirectory("0");
      createFile("0/gpu_id", "invalid_gpu_id");
      createFile("0/properties", "unique_id 12345\n"
                                 "location_id 23552\n"
                                 "domain 0\n"
                                 "vendor_id 4098\n");

      AltRsmiTestUtils::ResetState();
      AltRsmiTestUtils::SetNodesPath(kTestKFDPath);

      int result = ARSMI_init();

      // Init should handle invalid gpu_id gracefully
      ASSERT_EQ(result, 0);

      // Should not discover any devices with invalid gpu_id
      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_EQ(num_devices, 0);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitWithEmptyPropertiesFile) {
  RUN_ISOLATED_TEST(
    "ARSMIInitWithEmptyPropertiesFile",
    []() {
      // Create an empty properties file
      removeTestSandbox();
      createDirectory();
      createDirectory("0");
      createFile("0/gpu_id", "4098\n");
      createFile("0/properties", "");

      AltRsmiTestUtils::ResetState();
      AltRsmiTestUtils::SetNodesPath(kTestKFDPath);

      int result = ARSMI_init();

      // Should succeed but not discover devices with empty properties
      ASSERT_EQ(result, 0);

      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_EQ(num_devices, 0);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitWithDirectoryInsteadOfPropertiesFile) {
  RUN_ISOLATED_TEST(
    "ARSMIInitWithDirectoryInsteadOfPropertiesFile",
    []() {
      // Create a directory instead of properties file
      removeTestSandbox();
      createDirectory();
      createDirectory("0");
      createFile("0/gpu_id", "4098\n");
      createDirectory("0/properties");

      AltRsmiTestUtils::ResetState();
      AltRsmiTestUtils::SetNodesPath(kTestKFDPath);

      int result = ARSMI_init();

      // Should handle this gracefully
      ASSERT_EQ(result, 0);

      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_EQ(num_devices, 0);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitWithMissingVendorId) {
  RUN_ISOLATED_TEST(
    "ARSMIInitWithMissingVendorId",
    []() {
      // Create node without vendor_id
      removeTestSandbox();
      createDirectory();
      createDirectory("0");
      createFile("0/gpu_id", "4098\n");
      createFile("0/properties", "unique_id 12345\n"
                                 "location_id 23552\n"
                                 "domain 0\n");
                                 // Missing vendor_id

      AltRsmiTestUtils::ResetState();
      AltRsmiTestUtils::SetNodesPath(kTestKFDPath);

      int result = ARSMI_init();
      ASSERT_EQ(result, 0);

      // Should not discover devices without vendor_id
      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_EQ(num_devices, 0);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitWithNonAMDVendorId) {
  RUN_ISOLATED_TEST(
    "ARSMIInitWithNonAMDVendorId",
    []() {
      // Create node with non-AMD vendor_id
      removeTestSandbox();
      createDirectory();
      createDirectory("0");
      createFile("0/gpu_id", "4098\n");
      createFile("0/properties", "unique_id 12345\n"
                                 "location_id 23552\n"
                                 "domain 0\n"
                                 "vendor_id 0x10DE\n"); // NVIDIA vendor ID

      AltRsmiTestUtils::ResetState();
      AltRsmiTestUtils::SetNodesPath(kTestKFDPath);

      int result = ARSMI_init();
      ASSERT_EQ(result, 0);

      // Should not discover non-AMD devices
      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_EQ(num_devices, 0);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ARSMIInitWithEmptyLinkPropertiesFile) {
  RUN_ISOLATED_TEST(
    "ARSMIInitWithEmptyLinkPropertiesFile",
    []() {
      setupTestEnvironment();

      // Create setup but with empty link properties
      createFile("0/io_links/0/properties", "");

      int result = ARSMI_init();

      // Should still initialize, just skip that link
      ASSERT_EQ(result, 0);

      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_GT(num_devices, 0);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, NullInfoPointer) {
  RUN_ISOLATED_TEST(
    "NullInfoPointer",
    []() {
      setupTestEnvironment();

      int result = ARSMI_topo_get_link_info(0, 1, nullptr);
      ASSERT_EQ(result, EINVAL); // Expect EINVAL for null `info` pointer

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, SourceDeviceIndexOutOfRange) {
  RUN_ISOLATED_TEST(
    "SourceDeviceIndexOutOfRange",
    []() {
      setupTestEnvironment();

      ARSMI_linkInfo info;
      // First initialize
      ASSERT_EQ(ARSMI_init(), 0);

      int result = ARSMI_topo_get_link_info(999, 1, &info); // Invalid source index
      ASSERT_EQ(result, EINVAL); // Expect EINVAL for out-of-range source index

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, DestinationDeviceIndexOutOfRange) {
  RUN_ISOLATED_TEST(
    "DestinationDeviceIndexOutOfRange",
    []() {
      setupTestEnvironment();

      ARSMI_linkInfo info;
      // First initialize
      ASSERT_EQ(ARSMI_init(), 0);

      int result = ARSMI_topo_get_link_info(0, 999, &info); // Invalid destination index
      ASSERT_EQ(result, EINVAL); // Expect EINVAL for out-of-range destination index

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, LinkInfoAutoInitializes) {
  RUN_ISOLATED_TEST(
    "LinkInfoAutoInitializes",
    []() {
      setupTestEnvironment();

      // Test that ARSMI_topo_get_link_info auto-initializes if not already initialized
      ARSMI_linkInfo info;
      int result = ARSMI_topo_get_link_info(0, 0, &info);

      // Should succeed - auto-initialization should work with test data
      ASSERT_EQ(result, 0);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ValidLinkInfoBetweenDevices) {
  RUN_ISOLATED_TEST(
    "ValidLinkInfoBetweenDevices",
    []() {
      setupTestEnvironment();

      // Initialize the system
      ASSERT_EQ(ARSMI_init(), 0);

      ARSMI_linkInfo info;
      int result = ARSMI_topo_get_link_info(0, 1, &info);

      // Should succeed
      ASSERT_EQ(result, 0);

      // Verify link info contains reasonable values
      ASSERT_EQ(info.src_node, 0);
      ASSERT_EQ(info.dst_node, 1);
      // Type should be XGMI (type 11 in properties)
      ASSERT_EQ(info.type, ARSMI_IOLINK_TYPE_XGMI);
      ASSERT_EQ(info.hops, 1);
      ASSERT_EQ(info.weight, 21);
      ASSERT_EQ(info.min_bandwidth, 0);
      ASSERT_EQ(info.max_bandwidth, 50000);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, ValidLinkInfoSelfLink) {
  RUN_ISOLATED_TEST(
    "ValidLinkInfoSelfLink",
    []() {
      setupTestEnvironment();

      // Initialize the system
      ASSERT_EQ(ARSMI_init(), 0);

      ARSMI_linkInfo info;
      int result = ARSMI_topo_get_link_info(0, 0, &info);

      // Should succeed - even self-links should return default values
      ASSERT_EQ(result, 0);

      cleanupTestEnvironment();
    }
  );
}

// TODO(alt_rsmi.cc): The current behavior of ARSMI_topo_get_link_info is questionable.
// When no direct link exists between devices, it silently returns success with
// fabricated default values (PCIe, 2 hops, weight 40). This can mislead callers
// into thinking a real link exists. The implementation should return an error
// code (e.g., ENOENT) when no link is defined. Once fixed, this test should be
// updated to verify the new behavior.
TEST(AltRsmiTest, LinkInfoWithNoDirectConnection) {
  RUN_ISOLATED_TEST(
    "LinkInfoWithNoDirectConnection",
    []() {
      // Setup with 2 nodes where they don't have direct XGMI connection
      removeTestSandbox();
      createDirectory();

      // Create node 0
      createDirectory("0");
      createFile("0/gpu_id", "4098\n");
      createFile("0/properties",
                 "unique_id 100\n"
                 "location_id 23552\n"
                 "domain 0\n"
                 "vendor_id 4098\n");
      // Create empty io_links directory (no actual links defined)
      createDirectory("0/io_links");

      // Create node 1
      createDirectory("1");
      createFile("1/gpu_id", "4098\n");
      createFile("1/properties",
                 "unique_id 101\n"
                 "location_id 23553\n"
                 "domain 1\n"
                 "vendor_id 4098\n");
      // Create empty io_links directory (no actual links defined)
      createDirectory("1/io_links");

      AltRsmiTestUtils::ResetState();
      AltRsmiTestUtils::SetNodesPath(kTestKFDPath);

      ASSERT_EQ(ARSMI_init(), 0);

      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_EQ(num_devices, 2);

      // Try to get link info between the two devices (no direct link defined)
      ARSMI_linkInfo info;
      int result = ARSMI_topo_get_link_info(0, 1, &info);

      // Should succeed but return default values since no io_links are defined
      ASSERT_EQ(result, 0);

      // When no direct link exists, src_node and dst_node remain as UINT_MAX
      ASSERT_EQ(info.src_node, std::numeric_limits<unsigned>::max());
      ASSERT_EQ(info.dst_node, std::numeric_limits<unsigned>::max());

      // Default values set
      ASSERT_EQ(info.hops, 2); // Default hops
      ASSERT_EQ(info.type, ARSMI_IOLINK_TYPE_PCIEXPRESS); // Default type
      ASSERT_EQ(info.weight, 40); // Default weight
      ASSERT_EQ(info.min_bandwidth, 0); // Default min_bandwidth
      ASSERT_EQ(info.max_bandwidth, 0); // Default max_bandwidth

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, MultipleDevicesWithXGMILinks) {
  RUN_ISOLATED_TEST(
    "MultipleDevicesWithXGMILinks",
    []() {
      setupTestEnvironment();

      // Test XGMI link type (type 11)
      ASSERT_EQ(ARSMI_init(), 0);

      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_EQ(num_devices, 2);

      // Get link info for XGMI connection
      ARSMI_linkInfo info;
      ASSERT_EQ(ARSMI_topo_get_link_info(0, 1, &info), 0);

      // Verify XGMI properties
      ASSERT_EQ(info.type, ARSMI_IOLINK_TYPE_XGMI);
      ASSERT_EQ(info.hops, 1);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, LinkTypeUndefined) {
  RUN_ISOLATED_TEST(
    "LinkTypeUndefined",
    []() {
      setupTestEnvironment();

      // Remove existing links and create setup with only undefined link type
      removeFile("0/io_links/0/properties");
      removeFile("0/io_links/1/properties");
      removeFile("1/io_links/0/properties");
      removeFile("1/io_links/1/properties");

      // Create link with undefined type (must be read last to not be overwritten)
      createDirectory("0/io_links/2");
      createFile("0/io_links/2/properties",
                 "type 99\n"  // Undefined type (not 2 or 11)
                 "version_major 0\n"
                 "version_minor 0\n"
                 "node_from 0\n"
                 "node_to 1\n"
                 "weight 21\n"
                 "min_latency 0\n"
                 "max_latency 0\n"
                 "min_bandwidth 0\n"
                 "max_bandwidth 50000\n"
                 "recommended_transfer_size 0\n"
                 "recommended_sdma_engine_id_mask 0\n"
                 "flags 0\n");

      ASSERT_EQ(ARSMI_init(), 0);

      ARSMI_linkInfo info;
      ASSERT_EQ(ARSMI_topo_get_link_info(0, 1, &info), 0);

      // Should have undefined type
      ASSERT_EQ(info.type, ARSMI_IOLINK_TYPE_UNDEFINED);
      ASSERT_EQ(info.hops, 0);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, DeviceOrderingByBDF) {
  RUN_ISOLATED_TEST(
    "DeviceOrderingByBDF",
    []() {
      setupTestEnvironment();

      // Test that devices are ordered by BDF correctly
      ASSERT_EQ(ARSMI_init(), 0);

      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_EQ(num_devices, 2);

      // Get BDF for both devices
      uint64_t bdf0 = 0, bdf1 = 0;
      ASSERT_EQ(ARSMI_dev_pci_id_get(0, &bdf0), 0);
      ASSERT_EQ(ARSMI_dev_pci_id_get(1, &bdf1), 0);

      // BDFs should be ordered (lower BDF first)
      // Based on test data: node0 (domain=0, location_id=23552) comes before
      // node1 (domain=1, location_id=23553)
      ASSERT_LT(bdf0, bdf1);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, FileExistsCheck) {
  RUN_ISOLATED_TEST(
    "FileExistsCheck",
    []() {
      // Test fileExists() indirectly by verifying behavior when files don't exist
      // This covers the fileExists(char const*) internal function

      removeTestSandbox();
      createDirectory();

      // Scenario 1: Node with missing gpu_id file - should be skipped
      createDirectory("0");
      // Don't create gpu_id file - fileExists() will return false for it
      createFile("0/properties",
                 "unique_id 100\n"
                 "location_id 23552\n"
                 "domain 0\n"
                 "vendor_id 4098\n");
      createDirectory("0/io_links");

      // Scenario 2: Node with missing properties file - should be skipped
      createDirectory("1");
      createFile("1/gpu_id", "4098\n");
      // Don't create properties file - fileExists() will return false for it
      createDirectory("1/io_links");

      // Scenario 3: Complete valid node
      createDirectory("2");
      createFile("2/gpu_id", "4098\n");
      createFile("2/properties",
                 "unique_id 102\n"
                 "location_id 23554\n"
                 "domain 2\n"
                 "vendor_id 4098\n");
      createDirectory("2/io_links");

      AltRsmiTestUtils::ResetState();
      AltRsmiTestUtils::SetNodesPath(kTestKFDPath);

      ASSERT_EQ(ARSMI_init(), 0);

      // Only the complete node should be discovered (fileExists filtered out the others)
      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_EQ(num_devices, 1);

      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, BDFSortingLambda) {
  RUN_ISOLATED_TEST(
    "BDFSortingLambda",
    []() {
      // Test the BDF sorting lambda comparator in ARSMI_init()
      // The lambda at line 183-186 sorts devices with the SAME unique_id by BDF
      // Create multiple partitions (same unique_id) with different BDF values in REVERSE order
      removeTestSandbox();
      createDirectory();

      // Create 4 nodes with the SAME unique_id but different location_ids (which affects BDF)
      // to exercise the lambda that sorts within the same unique_id group
      const std::string same_unique_id = "12345678901234567890";

      // Node 0: Highest BDF (will need to be moved to end by lambda)
      createDirectory("0");
      createFile("0/gpu_id", "4098\n");
      createFile("0/properties",
                 "unique_id " + same_unique_id + "\n"
                 "location_id 4294967040\n"  // Very high value for high BDF
                 "domain 3\n"
                 "vendor_id 4098\n");
      createDirectory("0/io_links");

      // Node 1: Second highest BDF
      createDirectory("1");
      createFile("1/gpu_id", "4098\n");
      createFile("1/properties",
                 "unique_id " + same_unique_id + "\n"
                 "location_id 16777216\n"
                 "domain 2\n"
                 "vendor_id 4098\n");
      createDirectory("1/io_links");

      // Node 2: Second lowest BDF
      createDirectory("2");
      createFile("2/gpu_id", "4098\n");
      createFile("2/properties",
                 "unique_id " + same_unique_id + "\n"
                 "location_id 65536\n"
                 "domain 1\n"
                 "vendor_id 4098\n");
      createDirectory("2/io_links");

      // Node 3: Lowest BDF (should be sorted to first by lambda)
      createDirectory("3");
      createFile("3/gpu_id", "4098\n");
      createFile("3/properties",
                 "unique_id " + same_unique_id + "\n"
                 "location_id 256\n"
                 "domain 0\n"
                 "vendor_id 4098\n");
      createDirectory("3/io_links");

      AltRsmiTestUtils::ResetState();
      AltRsmiTestUtils::SetNodesPath(kTestKFDPath);

      ASSERT_EQ(ARSMI_init(), 0);

      // All 4 nodes have the same unique_id, so they're all partitions of the same device
      // ARSMI_num_devices counts unique devices, but ARSMI_orderedNodes has all partitions
      uint32_t num_devices = 0;
      ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
      ASSERT_EQ(num_devices, 4);  // All 4 partitions should be counted

      // Access ARSMI_orderedNodes directly to verify the lambda sorted by s_bdf
      ASSERT_EQ(ARSMI_orderedNodes.size(), 4);

      // The lambda should have sorted these by s_bdf within the unique_id group
      // Verify ascending order
      ASSERT_LT(ARSMI_orderedNodes[0].s_bdf, ARSMI_orderedNodes[1].s_bdf);
      ASSERT_LT(ARSMI_orderedNodes[1].s_bdf, ARSMI_orderedNodes[2].s_bdf);
      ASSERT_LT(ARSMI_orderedNodes[2].s_bdf, ARSMI_orderedNodes[3].s_bdf);

      // Verify the sort reordered them: node 3 should be first (lowest BDF)
      ASSERT_EQ(ARSMI_orderedNodes[0].s_node_id, 3);  // Node 3 has domain 0, location 256
      ASSERT_EQ(ARSMI_orderedNodes[1].s_node_id, 2);  // Node 2 has domain 1, location 65536
      ASSERT_EQ(ARSMI_orderedNodes[2].s_node_id, 1);  // Node 1 has domain 2, location 16777216
      ASSERT_EQ(ARSMI_orderedNodes[3].s_node_id, 0);  // Node 0 has domain 3, location 4294967040

      // Verify they all have the same unique_id
      ASSERT_EQ(ARSMI_orderedNodes[0].s_unique_id, ARSMI_orderedNodes[1].s_unique_id);
      ASSERT_EQ(ARSMI_orderedNodes[1].s_unique_id, ARSMI_orderedNodes[2].s_unique_id);
      ASSERT_EQ(ARSMI_orderedNodes[2].s_unique_id, ARSMI_orderedNodes[3].s_unique_id);

      cleanupTestEnvironment();
    }
  );
}

} // namespace RcclUnitTesting
