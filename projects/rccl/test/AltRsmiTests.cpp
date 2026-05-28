/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "alt_rsmi.h"

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

struct ARSMI_systemNode {
  uint32_t s_node_id = 0;
  uint64_t s_gpu_id = 0;
  uint64_t s_unique_id = 0;
  uint64_t s_location_id = 0;
  uint64_t s_bdf = 0;
  uint64_t s_domain = 0;
  uint8_t s_bus = 0;
  uint8_t s_device = 0;
  uint8_t s_function = 0;
  uint8_t s_partition_id = 0;
  std::string s_card;
};

const char *kPathDRMRoot = "/sys/class/drm";
const char *kKFDNodesPathRoot = "/sys/class/kfd/kfd/topology/nodes";
uint32_t kAmdGpuId = 0x1002;

// Vector containing data about each node, ordered by bdf ID
thread_local std::vector<ARSMI_systemNode> ARSMI_orderedNodes;

// 2-D matrix with link information between each pair of nodes.
thread_local std::vector<std::vector<ARSMI_linkInfo>> ARSMI_orderedLinks;

// Number of devices recognized
thread_local int ARSMI_num_devices = -1;

int getNodeIndex(uint32_t node_id);

std::string DevicePath(uint32_t dev_id);

int isRegularFile(std::string fname, bool *is_reg);

bool isNumber(const std::string &s);

int openNodeFile(uint32_t dev_id, std::string node_file, std::ifstream *fs);

int countIoLinks(uint32_t dev_id);

int openLinkFile(uint32_t dev_id, uint32_t target_id, std::string node_file,
                 std::ifstream *fs);
int readGpuId(uint32_t node_id, uint64_t *gpu_id);

bool isNodeSupported(uint32_t node_indx);

int getPropertyValue(std::string property, uint64_t *value,
                     std::map<std::string, uint64_t> &properties);

bool fileExists(char const *filename);

int ARSMI_readDeviceProperties(uint32_t node_id,
                               std::map<std::string, uint64_t> &properties);

int ARSMI_readLinkProperties(uint32_t node_id, uint32_t target_node_id,
                             std::map<std::string, uint64_t> &properties);

// /sys/class/kfd/kfd/topology/nodes/*/properties
int read_node_properties(uint32_t node, std::string property_name,
                         uint64_t *val,
                         std::map<std::string, uint64_t> &properties);

// /sys/class/kfd/kfd/topology/nodes/*/io_links/*/properties
int read_link_properties(uint32_t node, uint32_t target,
                         std::string property_name, uint64_t *val,
                         std::map<std::string, uint64_t> &properties);

// /sys/class/kfd/kfd/topology/nodes/*/gpu_id
int getGpuId(uint32_t node, uint64_t *gpu_id);

namespace RcclUnitTesting {

class AltRsmiTest : public ::testing::Test {

protected:
  // Helper function to create directories recursively
  int createDirectory(const std::string &path) {
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

  // Helper function to remove a directory recursively
  int removeDirectory(const std::string &path) {
    DIR *dir = opendir(path.c_str());
    if (!dir) {
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
        if (removeDirectory(fullPath) == -1) {
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

  // Helper function to create a file with content
  void createFile(const std::string &path, const std::string &content) {
    std::ofstream file(path);
    if (!file) {
      std::cerr << "Failed to create file: " << path << ", errno: " << errno << std::endl;
      return;
    }
    file << content;
    file.close();
  }

  // Helper function to remove a file
  int removeFile(const std::string &path) {
    if (unlink(path.c_str()) == -1) {
      std::cerr << "Failed to remove file: " << path << " (errno: " << errno
                << ")" << std::endl;
      return -1; // Return error if file removal fails
    }
    return 0; // Success
  }

  // Function to create the test directory structure and files
  void setupTestFiles() {
    const std::string basePath = "/tmp/test_kfd/topology/nodes";

    createDirectory(basePath);

    // Create node 0 with valid data
    createDirectory(basePath + "/0");
    createFile(basePath + "/0/gpu_id", "4098\n");
    createFile(basePath + "/0/properties", "unique_id 16336014475442738425\n"
                                           "location_id 23552\n"
                                           "domain 2\n"
                                           "vendor_id 4098\n");

    createDirectory(basePath + "/0/io_links/0");
    createFile(basePath + "/0/io_links/0/properties",
               "type 5\n"
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

    createDirectory(basePath + "/0/io_links/1");
    createFile(basePath + "/0/io_links/1/properties",
               "type 2\n"
               "version_major 0\n"
               "version_minor 0\n"
               "node_from 0\n"
               "node_to 0\n"
               "weight 21\n"
               "min_latency 0\n"
               "max_latency 0\n"
               "min_bandwidth 0\n"
               "max_bandwidth 64000\n"
               "recommended_transfer_size 0\n"
               "recommended_sdma_engine_id_mask 0\n"
               "flags 0\n");

    createDirectory(basePath + "/1");
    createFile(basePath + "/1/properties", "unique_id 16336014475442738426\n"
                                "location_id 23553\n"
                                "domain 1\n"
                                "vendor_id 4098\n");

    createDirectory(basePath + "/1/io_links/0");
    createFile(basePath + "/1/io_links/0/properties",
               "type 5\n"
               "version_major 0\n"
               "version_minor 0\n"
               "node_from 1\n"
               "node_to 0\n"
               "weight 21\n"
               "min_latency 0\n"
               "max_latency 0\n"
               "min_bandwidth 0\n"
               "max_bandwidth 0\n"
               "recommended_transfer_size 0\n"
               "recommended_sdma_engine_id_mask 0\n"
               "flags 0\n");

    createDirectory(basePath + "/1/io_links/1");
    createFile(basePath + "/1/io_links/1/properties",
               "type 2\n"
               "version_major 0\n"
               "version_minor 0\n"
               "node_from 1\n"
               "node_to 0\n"
               "weight 21\n"
               "min_latency 0\n"
               "max_latency 0\n"
               "min_bandwidth 0\n"
               "max_bandwidth 0\n"
               "recommended_transfer_size 0\n"
               "recommended_sdma_engine_id_mask 0\n"
               "flags 0\n");

    uint32_t invalid_dev_id = 9999; // Device ID that doesn't exist
    createDirectory(basePath + "/" + std::to_string(invalid_dev_id) + "/io_links/");

    ARSMI_num_devices = 2;
    ARSMI_systemNode node0 = {0, 0, 0, 0, 200, 0, 0, 0, 0, 0, ""};
    ARSMI_systemNode node1 = {1, 0, 0, 0, 100, 0, 0, 0, 0, 0, ""};

    ARSMI_orderedNodes.clear();
    ARSMI_orderedNodes.push_back(node0); // Node 0
    ARSMI_orderedNodes.push_back(node1); // Node 1

    ARSMI_orderedLinks.clear();
    ARSMI_orderedLinks.resize(2);

    // Link info from node 0 to node 0 and node 1
    ARSMI_orderedLinks[0].push_back({0, 0, 0, ARSMI_IOLINK_TYPE_UNDEFINED, 0, 0, 0}); // self-link
    ARSMI_orderedLinks[0].push_back({0, 1, 1, ARSMI_IOLINK_TYPE_PCIEXPRESS, 40, 1000, 2000}); // 0->1

    // Link info from node 1 to node 0 and node 1
    ARSMI_orderedLinks[1].push_back({1, 0, 1, ARSMI_IOLINK_TYPE_PCIEXPRESS, 40, 1000, 2000}); // 1->0
    ARSMI_orderedLinks[1].push_back({1, 1, 0, ARSMI_IOLINK_TYPE_UNDEFINED, 0, 0, 0}); // self-link
  }

  void SetUp() override {
    // Redirect kKFDNodesPathRoot to a temporary directory for testing
    kKFDNodesPathRoot = "/tmp/test_kfd/topology/nodes";

    // Create the test directory structure and files
    setupTestFiles();
  }

  void TearDown() override {
    // Clean up the temporary directory
    removeDirectory("/tmp/test_kfd");
  }
};

TEST_F(AltRsmiTest, ARSMIInitDefault) {
  ARSMI_num_devices = -1; // Force uninitialized state
  int result = ARSMI_init();

  ASSERT_EQ(result, 0);
}

TEST_F(AltRsmiTest, ARSMIInitMissingIoLinksPropertiesFile) {
  ARSMI_num_devices = -1; // Force uninitialized state
  // Remove properties file for io_links
  removeFile("/tmp/test_kfd/topology/nodes/0/io_links/0/properties");
  removeFile("/tmp/test_kfd/topology/nodes/0/io_links/1/properties");

  int result = ARSMI_init();

  ASSERT_EQ(result, 0);
}

TEST_F(AltRsmiTest, ARSMIInitMissingNodeToProperty) {
  ARSMI_num_devices = -1; // Force uninitialized state
  createFile("/tmp/test_kfd/topology/nodes/0/io_links/1/properties",
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

  ASSERT_EQ(result, 0); // Expect success
}

TEST_F(AltRsmiTest, ARSMIInitMissingWeightProperty) {
  ARSMI_num_devices = -1; // Force uninitialized state
  createFile("/tmp/test_kfd/topology/nodes/0/io_links/1/properties",
             "type 2\n"
             "version_major 0\n"
             "version_minor 0\n"
             "node_from 0\n"
             "node_to 0\n"
             // "weight 21\n"  // Missing weight
             "min_latency 0\n"
             "max_latency 0\n"
             "min_bandwidth 0\n"
             "max_bandwidth 64000\n"
             "recommended_transfer_size 0\n"
             "recommended_sdma_engine_id_mask 0\n"
             "flags 0\n");

  int result = ARSMI_init();

  ASSERT_NE(result, 0); // Expect non-zero error code
}

TEST_F(AltRsmiTest, ARSMIInitMissingTypeProperty) {
  ARSMI_num_devices = -1; // Force uninitialized state
  createFile("/tmp/test_kfd/topology/nodes/0/io_links/1/properties",
             // "type 5\n" // Missing type
             "version_major 0\n"
             "version_minor 0\n"
             "node_from 0\n"
             "node_to 0\n"
             "weight 21\n"
             "min_latency 0\n"
             "max_latency 0\n"
             "min_bandwidth 0\n"
             "max_bandwidth 0\n"
             "recommended_transfer_size 0\n"
             "recommended_sdma_engine_id_mask 0\n"
             "flags 0\n");

  int result = ARSMI_init();

  ASSERT_NE(result, 0); // Expect non-zero error code
}

TEST_F(AltRsmiTest, ARSMIInitTypePCIeProperty) {
  ARSMI_num_devices = -1; // Force uninitialized state
  int result = ARSMI_init();

  ASSERT_EQ(result, 0);
}

TEST_F(AltRsmiTest, ARSMIInitMissingMinBWProperty) {
  ARSMI_num_devices = -1; // Force uninitialized state
  createFile("/tmp/test_kfd/topology/nodes/0/io_links/1/properties",
             "type 11\n"
             "version_major 0\n"
             "version_minor 0\n"
             "node_from 0\n"
             "node_to 0\n"
             "weight 21\n"
             "min_latency 0\n"
             "max_latency 0\n"
             // "min_bandwidth 0\n"  // Missing min_bandwidth
             "max_bandwidth 0\n"
             "recommended_transfer_size 0\n"
             "recommended_sdma_engine_id_mask 0\n"
             "flags 0\n");

  int result = ARSMI_init();

  ASSERT_NE(result, 0); // Expect non-zero error code
}

TEST_F(AltRsmiTest, ARSMIInitMissingMaxBWProperty) {
  ARSMI_num_devices = -1; // Force uninitialized state
  createFile("/tmp/test_kfd/topology/nodes/0/io_links/1/properties",
             "type 5\n"
             "version_major 0\n"
             "version_minor 0\n"
             "node_from 0\n"
             "node_to 0\n"
             "weight 21\n"
             "min_latency 0\n"
             "max_latency 0\n"
             "min_bandwidth 0\n"
             // "max_bandwidth 0\n"  // Missing max_bandwidth
             "recommended_transfer_size 0\n"
             "recommended_sdma_engine_id_mask 0\n"
             "flags 0\n");

  int result = ARSMI_init();

  ASSERT_NE(result, 0); // Expect non-zero error code
}

TEST_F(AltRsmiTest, ARSMIGetNumDevicesUninitialized) {
  ARSMI_num_devices = -1; // Force uninitialized state
  uint32_t num_devices = 0;

  int result = ARSMI_get_num_devices(&num_devices);

  // Verify that the function initializes successfully
  ASSERT_EQ(result, 0);

  // Verify that the number of devices is correctly set
  ASSERT_EQ(num_devices, ARSMI_num_devices);
}

TEST_F(AltRsmiTest, ARSMIDevPciIdGetNullBdfId) {
  uint32_t device_index = 0;
  int result = ARSMI_dev_pci_id_get(device_index, nullptr);

  ASSERT_EQ(result, EINVAL);
}

TEST_F(AltRsmiTest, ARSMIDevPciIdGetUninitialized) {
  ARSMI_num_devices = -1; // Force uninitialized state
  uint32_t device_index = 0;
  uint64_t bdfid = 0;

  // Fail to initialize the function
  // kKFDNodesPathRoot = "/invalid/path/to/file";
  kKFDNodesPathRoot = "/invalid/path/to/file";

  int result = ARSMI_dev_pci_id_get(device_index, &bdfid);

  // Verify that the function fails to initialize and returns an error
  ASSERT_NE(result, 0);
}

TEST_F(AltRsmiTest, GetNodeIndexInvalidNode) {
  uint32_t invalid_node_id = 9999; // Node ID that doesn't exist
  int result = getNodeIndex(invalid_node_id);
  ASSERT_EQ(result, -1); // Expect -1 for invalid node
}

TEST_F(AltRsmiTest, DevicePathInvalidDeviceId) {
  uint32_t invalid_dev_id = 9999; // Device ID that doesn't exist
  std::string path = DevicePath(invalid_dev_id);
  ASSERT_FALSE(path.empty()); // Path should still be constructed, but it won't
                              // point to a valid device
}

TEST_F(AltRsmiTest, IsRegularFileInvalidPath) {
  std::string invalid_path = "/invalid/path/to/file";
  bool is_reg = false;
  int result = isRegularFile(invalid_path, &is_reg);
  ASSERT_NE(result, 0); // Expect non-zero error code
  ASSERT_FALSE(is_reg); // Expect is_reg to be false
}

TEST_F(AltRsmiTest, IsNumberInvalidInput) {
  ASSERT_FALSE(isNumber("abc123")); // Non-numeric string
  ASSERT_FALSE(isNumber(""));       // Empty string
  ASSERT_FALSE(isNumber(" "));      // Whitespace string
}

TEST_F(AltRsmiTest, OpenNodeFileInvalidPath) {
  std::ifstream fs;
  int result = openNodeFile(9999, "invalid_file", &fs);
  ASSERT_NE(result, 0);       // Expect non-zero error code
  ASSERT_FALSE(fs.is_open()); // File stream should not be open
}

TEST_F(AltRsmiTest, OpenNodeFileNotRegularFile) {
  removeFile("/tmp/test_kfd/topology/nodes/0/properties");

  // Create a directory instead of a regular file
  createDirectory("/tmp/test_kfd/topology/nodes/0/properties");

  std::ifstream fs;
  int result = openNodeFile(0, "properties", &fs);

  // Verify that the function returns ENOENT
  ASSERT_EQ(result, ENOENT);
}

TEST_F(AltRsmiTest, OpenNodeFileInvalidNodeFile) {
  uint32_t invalid_dev_id = 9999; // Device ID that doesn't exist
  std::ifstream fs;
  int result =
      openNodeFile(invalid_dev_id, "invalid_file", &fs);
  ASSERT_NE(result, 0);       // Expect non-zero error code
  ASSERT_FALSE(fs.is_open()); // File stream should not be open
}

TEST_F(AltRsmiTest, CountIoLinksInvalidDeviceId) {
  uint32_t invalid_dev_id = 9999; // Device ID that doesn't exist
  int result = countIoLinks(invalid_dev_id);
  ASSERT_EQ(result, 0); // Expect 0 links for an invalid device
}

TEST_F(AltRsmiTest, OpenLinkFileInvalidLinkFile) {
  uint32_t invalid_dev_id = 9999;    // Device ID that doesn't exist
  uint32_t invalid_target_id = 9999; // Target ID that doesn't exist
  std::ifstream fs;
  int result = openLinkFile(invalid_dev_id, invalid_target_id,
                                             "invalid_file", &fs);
  ASSERT_NE(result, 0);       // Expect non-zero error code
  ASSERT_FALSE(fs.is_open()); // File stream should not be open
}

TEST_F(AltRsmiTest, OpenLinkFileInvalidPath) {
  std::ifstream fs;
  int result = openLinkFile(9999, 9999, "invalid_file", &fs);
  ASSERT_NE(result, 0);       // Expect non-zero error code
  ASSERT_FALSE(fs.is_open()); // File stream should not be open
}

TEST_F(AltRsmiTest, OpenLinkFileNotRegularFile) {
  removeFile("/tmp/test_kfd/topology/nodes/0/io_links/1/properties");

  // Create a directory instead of a regular file
  createDirectory("/tmp/test_kfd/topology/nodes/0/io_links/1/properties");

  std::ifstream fs;
  int result = openLinkFile(0, 1, "properties", &fs);

  // Verify that the function returns ENOENT
  ASSERT_EQ(result, ENOENT);
}

TEST_F(AltRsmiTest, GetGpuIdInvalidNode) {
  uint64_t gpu_id = 0;
  int result = getGpuId(9999, &gpu_id);
  ASSERT_NE(result, 0); // Expect non-zero error code
  ASSERT_EQ(gpu_id, 0); // GPU ID should not be modified
}

TEST_F(AltRsmiTest, GetGpuIdInvalidId) {
  uint64_t *gpu_id = nullptr;
  int result = getGpuId(9999, gpu_id);
  ASSERT_NE(result, 0); // Expect non-zero error code
}

TEST_F(AltRsmiTest, ReadGpuIdInvalidNode) {
  uint32_t invalid_node_id = 9999; // Node ID that doesn't exist
  uint64_t gpu_id = 0;
  int result = readGpuId(invalid_node_id, &gpu_id);
  ASSERT_NE(result, 0); // Expect non-zero error code
  ASSERT_EQ(gpu_id, 0); // GPU ID should not be modified
}

TEST_F(AltRsmiTest, ReadGpuIdInvalidData) {
  // Create the directory structure
  removeDirectory("/tmp/test_kfd");
  createDirectory("/tmp/test_kfd/topology/nodes/0");

  // Create a gpu_id file with invalid (non-numeric) data
  std::ofstream gpu_id_file("/tmp/test_kfd/topology/nodes/0/gpu_id");
  gpu_id_file << "invalid_gpu_id"; // Non-numeric data
  gpu_id_file.close();

  uint64_t gpu_id = 0;

  // Call the readGpuId function
  int result = readGpuId(0, &gpu_id);

  // Verify that the function returns ENXIO
  ASSERT_EQ(result, ENXIO);
}

TEST_F(AltRsmiTest, IsNodeSupportedInvalidNode) {
  uint32_t invalid_node_id = 9999; // Node ID that doesn't exist
  bool result = isNodeSupported(invalid_node_id);
  ASSERT_FALSE(result); // Expect false for unsupported node
}

TEST_F(AltRsmiTest, IsNodeSupportedEmptyFile) {
  // Create an empty properties file
  std::ofstream properties_file("/tmp/test_kfd/topology/nodes/0/properties");
  properties_file.close();

  // Call the isNodeSupported function
  bool result = isNodeSupported(0);

  // Verify that the function returns false for an empty file
  ASSERT_FALSE(result);
}

TEST_F(AltRsmiTest, GetPropertyValueInvalidProperty) {
  std::map<std::string, uint64_t> properties = {{"valid_property", 12345}};
  uint64_t value = 0;
  int result =
      getPropertyValue("invalid_property", &value, properties);
  ASSERT_NE(result, 0); // Expect non-zero error code
  ASSERT_EQ(value, 0);  // Value should not be modified
}

TEST_F(AltRsmiTest, GetPropertyValueNullValuePointer) {
  std::map<std::string, uint64_t> properties = {{"key1", 12345}};
  uint64_t *value = nullptr;

  // Call the function with a null value pointer
  int result = getPropertyValue("key1", value, properties);

  // Verify that the function returns EINVAL
  ASSERT_EQ(result, EINVAL);
}

TEST_F(AltRsmiTest, GetPropertyValueEmptyPropertiesMap) {
  std::map<std::string, uint64_t> properties; // Empty map
  uint64_t value = 0;

  // Call the function with an empty properties map
  int result = getPropertyValue("key1", &value, properties);

  // Verify that the function returns EINVAL
  ASSERT_EQ(result, EINVAL);
}

TEST_F(AltRsmiTest, GetPropertyValueKeyNotFound) {
  std::map<std::string, uint64_t> properties = {{"key1", 12345}};
  uint64_t value = 0;

  // Call the function with a key that does not exist in the map
  int result = getPropertyValue("key2", &value, properties);

  // Verify that the function returns EINVAL
  ASSERT_EQ(result, EINVAL);
}

TEST_F(AltRsmiTest, FileExistsInvalidPath) {
  const char *invalid_path = "/invalid/path/to/file";
  bool result = fileExists(invalid_path);
  ASSERT_FALSE(result); // Expect false for non-existent file
}

TEST_F(AltRsmiTest, ARSMIReadDevicePropertiesInvalidNode) {
  uint32_t invalid_node_id = 9999; // Node ID that doesn't exist
  std::map<std::string, uint64_t> properties;
  int result =
      ARSMI_readDeviceProperties(invalid_node_id, properties);
  ASSERT_NE(result, 0);            // Expect non-zero error code
  ASSERT_TRUE(properties.empty()); // Properties map should remain empty
}

TEST_F(AltRsmiTest, ARSMI_readDevicePropertiesNotRegularFile) {
  // Clean up
  removeFile("/tmp/test_kfd/topology/nodes/0/properties");

  // Create a directory instead of a regular file
  createDirectory("/tmp/test_kfd/topology/nodes/0/properties");

  std::map<std::string, uint64_t> properties = {{"unique_id", 12345},
                                                {"location_id", 67890}};

  std::ifstream fs;
  int result = ARSMI_readDeviceProperties(0, properties);

  // Verify that the function returns ENOENT
  ASSERT_EQ(result, ENOENT);
}

TEST_F(AltRsmiTest, ARSMI_readDevicePropertiesEmptyFile) {
  createFile("/tmp/test_kfd/topology/nodes/0/properties", "");

  std::map<std::string, uint64_t> properties;

  std::ifstream fs;
  int result = ARSMI_readDeviceProperties(0, properties);

  // Verify that the function handles empty lines correctly
  ASSERT_EQ(result, ENOENT);
}

TEST_F(AltRsmiTest, ARSMI_readDevicePropertiesTrailingEmptyLines) {
  createFile("/tmp/test_kfd/topology/nodes/0/properties", "key1 101\n"
                                                          "key2 102\n"
                                                          "  \n"
                                                          "\n");

  std::map<std::string, uint64_t> properties;

  std::ifstream fs;
  int result = ARSMI_readDeviceProperties(0, properties);

  // Verify that the function handles empty lines correctly
  ASSERT_EQ(result, 0);
}

TEST_F(AltRsmiTest, ARSMIReadLinkPropertiesInvalidLink) {
  uint32_t invalid_node_id = 9999;   // Node ID that doesn't exist
  uint32_t invalid_target_id = 9999; // Target ID that doesn't exist
  std::map<std::string, uint64_t> properties;
  int result = ARSMI_readLinkProperties(
      invalid_node_id, invalid_target_id, properties);
  ASSERT_NE(result, 0);            // Expect non-zero error code
  ASSERT_TRUE(properties.empty()); // Properties map should remain empty
}

TEST_F(AltRsmiTest, ARSMI_readLinkPropertiesNotRegularFile) {
  // Clean up
  removeFile("/tmp/test_kfd/topology/nodes/0/io_links/1/properties");

  // Create a directory instead of a regular file
  createDirectory("/tmp/test_kfd/topology/nodes/0/properties");

  std::map<std::string, uint64_t> properties = {{"unique_id", 12345},
                                                {"location_id", 67890}};

  std::ifstream fs;
  int result = ARSMI_readLinkProperties(0, 1, properties);

  // Verify that the function returns ENOENT
  ASSERT_EQ(result, ENOENT);

  // Clean up
  rmdir("/tmp/test_kfd/0/io_links/1/properties");
}

TEST_F(AltRsmiTest, ARSMI_readLinkPropertiesTrailingEmptyLine) {

  createFile("/tmp/test_kfd/topology/nodes/0/io_links/0/properties",
             "key1 101\n"
             "key2 102\n"
             "  \n");

  std::map<std::string, uint64_t> properties;

  std::ifstream fs;
  int result = ARSMI_readLinkProperties(0, 0, properties);

  // Verify that the function handles empty lines correctly
  ASSERT_EQ(result, 0);
}

TEST_F(AltRsmiTest, ARSMI_readLinkPropertiesEmptyFile) {
  createFile("/tmp/test_kfd/topology/nodes/0/io_links/0/properties", "");

  std::map<std::string, uint64_t> properties;

  std::ifstream fs;
  int result = ARSMI_readLinkProperties(0, 0, properties);

  ASSERT_EQ(result, ENOENT);
}

TEST_F(AltRsmiTest, ReadNodePropertiesInvalidProperty) {
  std::map<std::string, uint64_t> properties = {{"unique_id", 12345},
                                                {"location_id", 67890}};
  uint64_t value = 0;

  // Call the wrapper function with an invalid property name
  int result = read_node_properties(0, "invalid_property",
                                                     &value, properties);

  // Verify that the function fails for an invalid property name
  ASSERT_NE(result, 0);
}

TEST_F(AltRsmiTest, ReadNodePropertiesInvalidNode) {
  uint32_t invalid_node_id = 9999; // Node ID that doesn't exist
  std::map<std::string, uint64_t> properties;
  uint64_t value = 0;
  int result = read_node_properties(
      invalid_node_id, "valid_property", &value, properties);
  ASSERT_NE(result, 0); // Expect non-zero error code
  ASSERT_EQ(value, 0);  // Value should not be modified
}

TEST_F(AltRsmiTest, ReadNodePropertiesInvalidPropertyValue) {
  uint32_t invalid_node_id = 9999; // Node ID that doesn't exist
  std::map<std::string, uint64_t> properties;
  uint64_t *value = nullptr;
  int result = read_node_properties(invalid_node_id, "", value,
                                                     properties);
  ASSERT_EQ(result, EINVAL); // Expect non-zero error code
}

TEST_F(AltRsmiTest, ReadLinkPropertiesInvalidLink) {
  uint32_t invalid_node_id = 9999;   // Node ID that doesn't exist
  uint32_t invalid_target_id = 9999; // Target ID that doesn't exist
  std::map<std::string, uint64_t> properties;
  uint64_t value = 0;
  int result = read_link_properties(
      invalid_node_id, invalid_target_id, "valid_property", &value, properties);
  ASSERT_NE(result, 0); // Expect non-zero error code
  ASSERT_EQ(value, 0);  // Value should not be modified
}

TEST_F(AltRsmiTest, ReadLinkPropertiesInvalidPropertyValue) {
  uint32_t invalid_node_id = 9999;   // Node ID that doesn't exist
  uint32_t invalid_target_id = 9999; // Target ID that doesn't exist
  std::map<std::string, uint64_t> properties;
  uint64_t *value = nullptr;
  int result = read_link_properties(
      invalid_node_id, invalid_target_id, "", value, properties);
  ASSERT_EQ(result, EINVAL); // Expect non-zero error code
}

TEST_F(AltRsmiTest, NullInfoPointer) {
  int result = ARSMI_topo_get_link_info(0, 1, nullptr);
  ASSERT_EQ(result, EINVAL); // Expect EINVAL for null `info` pointer
}

TEST_F(AltRsmiTest, SourceDeviceIndexOutOfRange) {
  ARSMI_linkInfo info;
  ARSMI_num_devices =
      2; // Simulate initialized state with two devices
  int result = ARSMI_topo_get_link_info(999, 1, &info); // Invalid source index
  ASSERT_EQ(result, EINVAL); // Expect EINVAL for out-of-range source index
}

TEST_F(AltRsmiTest, DestinationDeviceIndexOutOfRange) {
  ARSMI_linkInfo info;
  ARSMI_num_devices =
      2; // Simulate initialized state with two devices
  int result =
      ARSMI_topo_get_link_info(0, 999, &info); // Invalid destination index
  ASSERT_EQ(result, EINVAL); // Expect EINVAL for out-of-range destination index
}

TEST_F(AltRsmiTest, UninitializedNumDevices) {

  kKFDNodesPathRoot =
      "/tmp/invalid_path"; // Simulate invalid path

  ARSMI_linkInfo info;
  ARSMI_num_devices = -1; // Simulate uninitialized state
  int result = ARSMI_topo_get_link_info(0, 0, &info);
  ASSERT_NE(result, 0); // Expect non-zero error code for uninitialized state
}

TEST_F(AltRsmiTest, InvalidLinkInfo) {

  // Initialize ARSMI_orderedLinks with data not in order
  ARSMI_orderedLinks = {
      {
          {1, 0, 0, ARSMI_IOLINK_TYPE_UNDEFINED, 0, 0,
           0}, // No link from Device 1 to itself
          {1, 0, 1, ARSMI_IOLINK_TYPE_PCIEXPRESS, 40, 1000, 2000}
          // Link from Device 1 to Device 0
      },
      {
          {0, 1, 1, ARSMI_IOLINK_TYPE_PCIEXPRESS, 40, 1000,
           2000}, // Link from Device 0 to Device 1
          {0, 0, 0, ARSMI_IOLINK_TYPE_UNDEFINED, 0, 0, 0}
          // No link from Device 0 to itself
      }};

  // Leave ARSMI_orderedLinks uninitialized
  ARSMI_linkInfo info;
  int result = ARSMI_topo_get_link_info(0, 1, &info);
  ASSERT_EQ(info.hops, 2); // Expect default values for uninitialized link info
  ASSERT_EQ(info.type, ARSMI_IOLINK_TYPE_PCIEXPRESS);
  ASSERT_EQ(info.weight, 40);
  ASSERT_EQ(info.min_bandwidth, 0);
  ASSERT_EQ(info.max_bandwidth, 0);
}

} // namespace RcclUnitTesting
