// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// kInvalidNumaNode/kInvalidNumaNodeWeight are the sentinels KFDNode::Initialize()
// assigns when no CPU io_link is found. Tests KFDNode::Initialize() against
// a mock KFD topology (via SetKFDNodesPathRootForTesting()) to prove getters are
// actually sentinel-valued when no CPU link exists, and hold real discovered values
// when one does. No GPU required.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "rocm_smi/rocm_smi_kfd.h"

namespace {

// Creates a unique directory on construction, recursively removes it on destruction.
class ScopedTempDir {
 public:
  ScopedTempDir() {
    std::string tmpl = (std::filesystem::temp_directory_path() / "kfd_numa_test_XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (char* result = mkdtemp(buf.data())) {
      path_ = result;
    }
  }
  ~ScopedTempDir() {
    if (!path_.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(path_, ec);
    }
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

// Points KFDNodesPathRoot() at a mock topology on construction, restores the real
// path on destruction.
class ScopedKfdRootOverride {
 public:
  explicit ScopedKfdRootOverride(const std::string& path) : path_(path) {
    amd::smi::SetKFDNodesPathRootForTesting(path_.c_str());
  }
  ~ScopedKfdRootOverride() { amd::smi::SetKFDNodesPathRootForTesting(nullptr); }

 private:
  std::string path_;
};

void WriteFile(const std::filesystem::path& path, const std::string& content) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path) << content;
}

// /sys/class/kfd/kfd/topology/nodes/<node_idx>/gpu_id ; 0 means a CPU node.
void WriteGpuId(const std::filesystem::path& root, uint32_t node_idx, uint64_t gpu_id) {
  WriteFile(root / std::to_string(node_idx) / "gpu_id", std::to_string(gpu_id) + "\n");
}

// Minimal properties KFDNode::Initialize() requires beyond gpu_id: hive_id plus
// three CU-count inputs read after io_link loop.
void WriteNodeProperties(const std::filesystem::path& root, uint32_t node_idx) {
  WriteFile(root / std::to_string(node_idx) / "properties",
            "hive_id 0\nsimd_arrays_per_engine 1\narray_count 1\ncu_per_simd_array 1\n");
}

// /sys/class/kfd/kfd/topology/nodes/<node_idx>/io_links/<link_idx>/properties
void WriteIoLink(const std::filesystem::path& root, uint32_t node_idx, uint32_t link_idx,
                 uint32_t node_to, amd::smi::IO_LINK_TYPE type, uint64_t weight) {
  std::ostringstream props;
  props << "type " << static_cast<uint64_t>(type) << "\n"
        << "node_from " << node_idx << "\n"
        << "node_to " << node_to << "\n"
        << "weight " << weight << "\n"
        << "flags 1\n"
        << "min_bandwidth 100\n"
        << "max_bandwidth 200\n";
  WriteFile(root / std::to_string(node_idx) / "io_links" / std::to_string(link_idx) / "properties",
            props.str());
}

TEST(SystemUnit, InvalidNumaNodeSentinelIsUint32Max) {
  EXPECT_EQ(amd::smi::kInvalidNumaNode, UINT32_MAX);
}

TEST(SystemUnit, InvalidNumaNodeWeightSentinelIsUint64Max) {
  EXPECT_EQ(amd::smi::kInvalidNumaNodeWeight, UINT64_MAX);
}

// Node 0 has a single XGMI io_link to node 1, another GPU (gpu_id != 0); No
// CPU node in topology
TEST(SystemUnit, KfdNodeInitializeSentinelsWhenNoCpuIoLink) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedKfdRootOverride kfd_root_override(root.path().string());

  WriteGpuId(root.path(), 0, 12345);
  WriteNodeProperties(root.path(), 0);
  WriteIoLink(root.path(), 0, 0, /*node_to=*/1, amd::smi::IOLINK_TYPE_XGMI, /*weight=*/15);
  WriteGpuId(root.path(), 1, 999);  // peer GPU, not a CPU node

  amd::smi::KFDNode node(0);
  ASSERT_EQ(node.Initialize(), 0);

  EXPECT_EQ(node.numa_node_number(), amd::smi::kInvalidNumaNode);
  EXPECT_EQ(node.numa_node_weight(), amd::smi::kInvalidNumaNodeWeight);
  EXPECT_EQ(node.numa_node_type(), amd::smi::IOLINK_TYPE_UNDEFINED);
}

// Node 2 has XGMI link to GPU node 3 and PCIe link to CPU node 4 (gpu_id == 0).
// Getters must reflect the discovered CPU node, not the sentinels.
TEST(SystemUnit, KfdNodeInitializeFindsCpuIoLink) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedKfdRootOverride kfd_root_override(root.path().string());

  WriteGpuId(root.path(), 2, 23456);
  WriteNodeProperties(root.path(), 2);
  WriteIoLink(root.path(), 2, 0, /*node_to=*/3, amd::smi::IOLINK_TYPE_XGMI, /*weight=*/15);
  WriteIoLink(root.path(), 2, 1, /*node_to=*/4, amd::smi::IOLINK_TYPE_PCIEXPRESS, /*weight=*/20);
  WriteGpuId(root.path(), 3, 34567);  // peer GPU
  WriteGpuId(root.path(), 4, 0);      // CPU node

  amd::smi::KFDNode node(2);
  ASSERT_EQ(node.Initialize(), 0);

  EXPECT_EQ(node.numa_node_number(), 4u);
  EXPECT_EQ(node.numa_node_weight(), 20u);
  EXPECT_EQ(node.numa_node_type(), amd::smi::IOLINK_TYPE_PCIEXPRESS);
}

}  // namespace
