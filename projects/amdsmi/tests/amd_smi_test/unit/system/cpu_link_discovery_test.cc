// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for amd::smi::DiscoverGpuCpuLinks() (rocm_smi_cpu_link.cc), the
// backing layer for amdsmi_get_gpu_cpu_links(). Exercises it directly against
// a mock KFD topology via SetKFDNodesPathRootForTesting(), the same hook used
// by kfd_numa_node_test.cc in this directory -- no GPU or node_handle/board
// fabrication involved, so there is no WSLGPUBackend short-circuit to guard
// against here (DiscoverGpuCpuLinks and its callees have no
// ENABLE_WSL_BACKEND branch; WSLGPUBackend only intercepts the amd_smi.cc
// public-API layer, which this test does not go through).

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi_cpu_link.h"
#include "rocm_smi/rocm_smi_io_link.h"

namespace {

// Identical in spirit to kfd_numa_node_test.cc's ScopedTempDir/ScopedKfdRootOverride;
// kept local to this file to avoid coupling across unit-test translation units.
class ScopedTempDir {
 public:
  ScopedTempDir() {
    std::string tmpl =
        (std::filesystem::temp_directory_path() / "cpu_link_discovery_test_XXXXXX").string();
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
//
// The standalone get_gpu_id() that DiscoverGpuCpuLinks() calls per io_link
// neighbor first runs a full KFDNode::Initialize() on that neighbor (gated by
// KFDNodeSupported(), which requires a non-empty properties file), unlike
// KFDNode::Initialize()'s own lighter io_link-neighbor gpu_id check (see
// kfd_numa_node_test.cc) which has neither gate. For a non-CPU (gpu_id != 0)
// neighbor, that nested Initialize() additionally requires a hive_id property
// and a readable (even empty) io_links directory, or it throws rather than
// returning an error code.
void WriteGpuId(const std::filesystem::path& root, uint32_t node_idx, uint64_t gpu_id) {
  WriteFile(root / std::to_string(node_idx) / "gpu_id", std::to_string(gpu_id) + "\n");
  // A CPU node's (gpu_id == 0) Initialize() returns before any of these
  // properties are consulted; a GPU node's does not, so it needs the full set
  // (matches WriteNodeProperties() in kfd_numa_node_test.cc).
  std::string properties = "gpu_id " + std::to_string(gpu_id) + "\n";
  if (gpu_id != 0) {
    properties += "hive_id 0\nsimd_arrays_per_engine 1\narray_count 1\ncu_per_simd_array 1\n";
  }
  WriteFile(root / std::to_string(node_idx) / "properties", properties);
  std::filesystem::create_directories(root / std::to_string(node_idx) / "io_links");
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

}  // namespace

// MI450-like topology: GPU node 1 has a direct XGMI io_link to CPU node 0.
TEST(SystemUnit, DiscoverGpuCpuLinksFindsDirectXgmiCpuLink) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedKfdRootOverride kfd_root_override(root.path().string());

  WriteGpuId(root.path(), 0, 0);  // CPU node
  WriteGpuId(root.path(), 1, 12345);
  WriteIoLink(root.path(), 1, 0, /*node_to=*/0, amd::smi::IOLINK_TYPE_XGMI, /*weight=*/15);

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);

  ASSERT_EQ(links.size(), 1u);
  EXPECT_EQ(links[0].numa_node_id, 0u);
  EXPECT_EQ(links[0].type, amd::smi::IOLINK_TYPE_XGMI);
}

// GPU node 1's only io_link neighbor is peer GPU node 2 (not a CPU) -- no CPU
// link exists, which must be reported as an empty, successful result.
TEST(SystemUnit, DiscoverGpuCpuLinksEmptyWhenOnlyPeerGpuNeighbor) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedKfdRootOverride kfd_root_override(root.path().string());

  WriteGpuId(root.path(), 1, 12345);
  WriteGpuId(root.path(), 2, 23456);  // peer GPU, not a CPU node
  WriteIoLink(root.path(), 1, 0, /*node_to=*/2, amd::smi::IOLINK_TYPE_XGMI, /*weight=*/15);

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);
  EXPECT_TRUE(links.empty());
}

// GPU node 1 has both a peer-GPU XGMI neighbor and a direct CPU PCIe
// neighbor -- only the CPU link must be reported.
TEST(SystemUnit, DiscoverGpuCpuLinksIgnoresNonCpuNeighbors) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedKfdRootOverride kfd_root_override(root.path().string());

  WriteGpuId(root.path(), 0, 0);  // CPU node
  WriteGpuId(root.path(), 1, 12345);
  WriteGpuId(root.path(), 2, 23456);  // peer GPU, not a CPU node
  WriteIoLink(root.path(), 1, 0, /*node_to=*/2, amd::smi::IOLINK_TYPE_XGMI, /*weight=*/15);
  WriteIoLink(root.path(), 1, 1, /*node_to=*/0, amd::smi::IOLINK_TYPE_PCIEXPRESS, /*weight=*/20);

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);

  ASSERT_EQ(links.size(), 1u);
  EXPECT_EQ(links[0].numa_node_id, 0u);
  EXPECT_EQ(links[0].type, amd::smi::IOLINK_TYPE_PCIEXPRESS);
}
