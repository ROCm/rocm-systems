/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "alt_rsmi.h"
#include "debug.h"
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

// Weak stub for ncclDebugLog, required because alt_rsmi.cc uses debug.h macros.
// Debug builds: ncclDebugLog is exported from librccl.so (-fvisibility=default),
//   so the strong shared-library definition wins at link time — stub is unused.
// Release builds: ncclDebugLog is hidden in librccl.so (-fvisibility=hidden),
//   so it is not exported; this stub provides the symbol and silently drops logs.
void __attribute__((weak)) ncclDebugLog(ncclDebugLogLevel, unsigned long,
                                        const char*, int, const char*, ...) {}

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
extern thread_local const char *kDrmClassRoot;
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

// Storage for the DRM class root path (keeps c_str() pointer stable)
static std::string sDrmClassRoot;

// Redirect DRM sysfs reads to a test sandbox directory.
// MUST be called inside the RUN_ISOLATED_TEST lambda (thread_local is per-thread).
static void SetDrmRoot(const std::string& path) {
    sDrmClassRoot = path;
    kDrmClassRoot = sDrmClassRoot.c_str();
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

// PID-scoped DRM sandbox path — avoids collisions with parallel CI runs.
// BDF correspondence: location_id=23552 (0x5C00), domain=0
//   s_bus=0x5C=92, s_device=0, s_function=0 → PCI_SLOT_NAME=0000:5c:00.0
static const std::string kTestDrmBasePath =
    std::filesystem::temp_directory_path().string() + "/test_drm_arsmi_" + std::to_string(getpid());

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

  // -------------------------------------------------------------------------
  // DRM sandbox helpers (operate under kTestDrmBasePath)
  // -------------------------------------------------------------------------

  int removeDrmSandbox() {
    return removeDirectoryImpl(kTestDrmBasePath);
  }

  void createDrmDirectory(const std::string& relPath) {
    createDirectoryImpl(kTestDrmBasePath + "/" + relPath);
  }

  void createDrmFile(const std::string& relPath, const std::string& content) {
    std::string full = kTestDrmBasePath + "/" + relPath;
    std::ofstream f(full);
    if (!f) {
      std::cerr << "Failed to create DRM file: " << full << ", errno: " << errno << std::endl;
      return;
    }
    f << content;
  }

  // Create a DRM card directory with a uevent file whose PCI_SLOT_NAME matches
  // the BDF encoded in location_id (see kTestDrmBasePath comment for formula).
  void setupDrmCard(int card, const std::string& pciSlotName) {
    createDrmDirectory("card" + std::to_string(card));
    createDrmDirectory("card" + std::to_string(card) + "/device");
    createDrmFile("card" + std::to_string(card) + "/device/uevent",
                  "PCI_SLOT_NAME=" + pciSlotName + "\n");
  }

  // Populate a ualink/ directory under the given card with all attributes.
  // ppodIdHex: UUID string "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", or "" to omit.
  void setupUalink(int card,
                   const std::string& linkType,
                   const std::string& accelState,
                   uint32_t accelId,
                   uint32_t bw,
                   uint32_t lat,
                   uint32_t ppodSize,
                   uint32_t vpodId,
                   uint32_t vpodSize,
                   const std::string& addrMode,
                   const std::string& ppodIdUuid) {
    std::string base = "card" + std::to_string(card) + "/device/ualink";
    createDrmDirectory(base);
    createDrmFile(base + "/link_type",   linkType   + "\n");
    createDrmFile(base + "/accel_state", accelState + "\n");
    createDrmFile(base + "/accel_id",    std::to_string(accelId)  + "\n");
    createDrmFile(base + "/bandwidth",   std::to_string(bw)       + "\n");
    createDrmFile(base + "/latency",     std::to_string(lat)      + "\n");
    createDrmFile(base + "/ppod_size",   std::to_string(ppodSize) + "\n");
    createDrmFile(base + "/vpod_id",     std::to_string(vpodId)   + "\n");
    createDrmFile(base + "/vpod_size",   std::to_string(vpodSize) + "\n");
    createDrmFile(base + "/addr_mode",   addrMode   + "\n");
    if (!ppodIdUuid.empty())
      createDrmFile(base + "/ppod_id",   ppodIdUuid + "\n");
  }

  // Populate a fw_version/ directory under the given card.
  void setupFwVersion(int card, uint64_t version) {
    std::string base = "card" + std::to_string(card) + "/device/fw_version";
    createDrmDirectory(base);
    char hex[32];
    snprintf(hex, sizeof(hex), "0x%lx\n", version);
    createDrmFile(base + "/mec_fw_version", hex);
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

// ============================================================================
// ARSMI_get_fw_version tests (4)
// kDrmClassRoot is redirected to kTestDrmBasePath inside each lambda.
// removeDrmSandbox() is called first in each lambda so that leftover files
// from a previously failed test do not affect the current one (ASSERT_*
// causes early return, skipping the trailing cleanup call).
// ============================================================================

TEST(AltRsmiTest, FwVersion_NullPointer) {
  RUN_ISOLATED_TEST(
    "FwVersion_NullPointer",
    []() {
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      int result = ARSMI_get_fw_version(0, nullptr);
      ASSERT_EQ(result, EINVAL);
      removeDrmSandbox();
    }
  );
}

TEST(AltRsmiTest, FwVersion_FileFound) {
  RUN_ISOLATED_TEST(
    "FwVersion_FileFound",
    []() {
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      setupFwVersion(0, 0x12345);
      uint64_t fw = 0xdeadbeef;
      int result = ARSMI_get_fw_version(0, &fw);
      ASSERT_EQ(result, 0);
      ASSERT_EQ(fw, 0x12345u);
      removeDrmSandbox();
    }
  );
}

TEST(AltRsmiTest, FwVersion_NoFile) {
  RUN_ISOLATED_TEST(
    "FwVersion_NoFile",
    []() {
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      // No DRM card directories created — loop finds nothing.
      uint64_t fw = 0xdeadbeef;
      int result = ARSMI_get_fw_version(0, &fw);
      ASSERT_EQ(result, 0);
      ASSERT_EQ(fw, 0u);
      removeDrmSandbox();
    }
  );
}

TEST(AltRsmiTest, FwVersion_FirstCardWins) {
  RUN_ISOLATED_TEST(
    "FwVersion_FirstCardWins",
    []() {
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      // card0 has no fw_version; card1 has 0xabcd.
      // Loop must stop at card1 (the first readable file).
      createDrmDirectory("card0");
      createDrmDirectory("card0/device");
      setupFwVersion(1, 0xabcd);
      uint64_t fw = 0;
      int result = ARSMI_get_fw_version(0, &fw);
      ASSERT_EQ(result, 0);
      ASSERT_EQ(fw, 0xabcdu);
      removeDrmSandbox();
    }
  );
}

// ============================================================================
// ARSMI_get_fabric_info tests (11)
//
// BDF correspondence for the standard KFD fixture (location_id=23552, domain=0):
//   s_bus=0x5C, s_device=0, s_function=0 → PCI_SLOT_NAME=0000:5c:00.0
//
// Tests that need ARSMI_orderedNodes (all except NullPointer) call
// setupTestEnvironment() first to populate it via ARSMI_init().
// ============================================================================

TEST(AltRsmiTest, FabricInfo_NullPointer) {
  RUN_ISOLATED_TEST(
    "FabricInfo_NullPointer",
    []() {
      // Null check (line 456) fires before num_devices check — no KFD setup needed.
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      int result = ARSMI_get_fabric_info(0, nullptr);
      ASSERT_EQ(result, EINVAL);
      removeDrmSandbox();
    }
  );
}

TEST(AltRsmiTest, FabricInfo_OutOfRange) {
  RUN_ISOLATED_TEST(
    "FabricInfo_OutOfRange",
    []() {
      setupTestEnvironment();
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      ARSMI_init();
      // KFD fixture creates 2 nodes (0 and 1); dv_ind=2 is out of range.
      ARSMI_fabricInfo info;
      int result = ARSMI_get_fabric_info(2, &info);
      ASSERT_EQ(result, EINVAL);
      removeDrmSandbox();
      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, FabricInfo_CardNotFound) {
  RUN_ISOLATED_TEST(
    "FabricInfo_CardNotFound",
    []() {
      setupTestEnvironment();
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      ARSMI_init();
      // No DRM cards created — findCardForBdf returns -1 → ENODEV.
      ARSMI_fabricInfo info;
      int result = ARSMI_get_fabric_info(0, &info);
      ASSERT_EQ(result, ENODEV);
      removeDrmSandbox();
      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, FabricInfo_NoUalinkDir) {
  RUN_ISOLATED_TEST(
    "FabricInfo_NoUalinkDir",
    []() {
      setupTestEnvironment();
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      ARSMI_init();
      // DRM card exists with matching uevent, but no ualink/ subdirectory.
      setupDrmCard(0, "0000:5c:00.0");
      ARSMI_fabricInfo info;
      int result = ARSMI_get_fabric_info(0, &info);
      ASSERT_EQ(result, ENODEV);
      removeDrmSandbox();
      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, FabricInfo_UALoE_Active) {
  RUN_ISOLATED_TEST(
    "FabricInfo_UALoE_Active",
    []() {
      setupTestEnvironment();
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      ARSMI_init();
      setupDrmCard(0, "0000:5c:00.0");
      // ppod_id UUID: 01020304-0506-0708-090a-0b0c0d0e0f10
      setupUalink(0, "UALoE", "active", 7, 200000, 100,
                  4, 2, 8, "source-aliasing",
                  "01020304-0506-0708-090a-0b0c0d0e0f10");
      ARSMI_fabricInfo info;
      int result = ARSMI_get_fabric_info(0, &info);
      ASSERT_EQ(result, 0);
      ASSERT_EQ(info.supported, 1);
      ASSERT_EQ(info.fabric_type, ARSMI_FABRIC_TYPE_UALOE);
      ASSERT_EQ(info.accel_state, ARSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE);
      ASSERT_EQ(info.accel_id, 7u);
      ASSERT_EQ(info.bandwidth, 200000u);
      ASSERT_EQ(info.latency, 100u);
      ASSERT_EQ(info.ppod_size, 4u);
      ASSERT_EQ(info.vpod_id, 2u);
      ASSERT_EQ(info.vpod_size, 8u);
      ASSERT_EQ(info.addr_mode, ARSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_ALIASING);
      // ppod_id bytes: 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10
      ASSERT_EQ(info.ppod_id[0],  0x01);
      ASSERT_EQ(info.ppod_id[3],  0x04);
      ASSERT_EQ(info.ppod_id[15], 0x10);
      removeDrmSandbox();
      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, FabricInfo_UALLink_Ready) {
  RUN_ISOLATED_TEST(
    "FabricInfo_UALLink_Ready",
    []() {
      setupTestEnvironment();
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      ARSMI_init();
      setupDrmCard(0, "0000:5c:00.0");
      setupUalink(0, "UALLink", "ready", 1, 100000, 50,
                  2, 0, 4, "source-identification", "");
      ARSMI_fabricInfo info;
      int result = ARSMI_get_fabric_info(0, &info);
      ASSERT_EQ(result, 0);
      ASSERT_EQ(info.supported, 1);
      ASSERT_EQ(info.fabric_type, ARSMI_FABRIC_TYPE_UALLINK);
      ASSERT_EQ(info.accel_state, ARSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY);
      ASSERT_EQ(info.addr_mode, ARSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_IDENTIFICATION);
      removeDrmSandbox();
      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, FabricInfo_ConfiguredState) {
  RUN_ISOLATED_TEST(
    "FabricInfo_ConfiguredState",
    []() {
      setupTestEnvironment();
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      ARSMI_init();
      setupDrmCard(0, "0000:5c:00.0");
      // "configured" is not ACTIVE or READY → supported must be 0.
      setupUalink(0, "UALoE", "configured", 0, 0, 0, 0, 0, 0, "source-aliasing", "");
      ARSMI_fabricInfo info;
      int result = ARSMI_get_fabric_info(0, &info);
      ASSERT_EQ(result, 0);
      ASSERT_EQ(info.supported, 0);
      ASSERT_EQ(info.fabric_type, ARSMI_FABRIC_TYPE_UALOE);
      ASSERT_EQ(info.accel_state, ARSMI_FABRIC_ACCELERATOR_VPOD_STATE_CONFIGURED);
      removeDrmSandbox();
      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, FabricInfo_UnknownLinkType) {
  RUN_ISOLATED_TEST(
    "FabricInfo_UnknownLinkType",
    []() {
      setupTestEnvironment();
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      ARSMI_init();
      setupDrmCard(0, "0000:5c:00.0");
      setupUalink(0, "SomeFutureFabric", "active", 0, 0, 0, 0, 0, 0, "source-aliasing", "");
      ARSMI_fabricInfo info;
      int result = ARSMI_get_fabric_info(0, &info);
      ASSERT_EQ(result, 0);
      ASSERT_EQ(info.fabric_type, ARSMI_FABRIC_TYPE_UNKNOWN);
      ASSERT_EQ(info.supported, 0);
      removeDrmSandbox();
      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, FabricInfo_MalformedPpodId) {
  RUN_ISOLATED_TEST(
    "FabricInfo_MalformedPpodId",
    []() {
      setupTestEnvironment();
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      ARSMI_init();
      setupDrmCard(0, "0000:5c:00.0");
      // ppod_id file exists but cannot be parsed as a UUID — sscanf returns <16.
      setupUalink(0, "UALoE", "active", 0, 0, 0, 0, 0, 0, "source-aliasing", "not-a-uuid");
      ARSMI_fabricInfo info;
      int result = ARSMI_get_fabric_info(0, &info);
      ASSERT_EQ(result, 0);
      // ppod_id must be zeroed on parse failure.
      for (int i = 0; i < 16; i++) ASSERT_EQ(info.ppod_id[i], 0);
      removeDrmSandbox();
      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, FabricInfo_MissingPpodIdFile) {
  RUN_ISOLATED_TEST(
    "FabricInfo_MissingPpodIdFile",
    []() {
      setupTestEnvironment();
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      ARSMI_init();
      setupDrmCard(0, "0000:5c:00.0");
      // ppod_id file absent — readUalinkStr returns -1; initial memset keeps it zero.
      setupUalink(0, "UALoE", "active", 0, 0, 0, 0, 0, 0, "source-aliasing", "");
      ARSMI_fabricInfo info;
      int result = ARSMI_get_fabric_info(0, &info);
      ASSERT_EQ(result, 0);
      for (int i = 0; i < 16; i++) ASSERT_EQ(info.ppod_id[i], 0);
      removeDrmSandbox();
      cleanupTestEnvironment();
    }
  );
}

TEST(AltRsmiTest, FabricInfo_AddrModeUnknown) {
  RUN_ISOLATED_TEST(
    "FabricInfo_AddrModeUnknown",
    []() {
      setupTestEnvironment();
      removeDrmSandbox();
      AltRsmiTestUtils::SetDrmRoot(kTestDrmBasePath);
      ARSMI_init();
      setupDrmCard(0, "0000:5c:00.0");
      setupUalink(0, "UALoE", "active", 0, 0, 0, 0, 0, 0, "something-new", "");
      ARSMI_fabricInfo info;
      int result = ARSMI_get_fabric_info(0, &info);
      ASSERT_EQ(result, 0);
      ASSERT_EQ(info.addr_mode, ARSMI_FABRIC_NPA_ADDRESS_MODE_UNKNOWN);
      removeDrmSandbox();
      cleanupTestEnvironment();
    }
  );
}

} // namespace RcclUnitTesting
