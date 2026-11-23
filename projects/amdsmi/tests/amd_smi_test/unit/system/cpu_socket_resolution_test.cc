// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for AMDSmiSystem::get_cpu_sockets_from_numa_node() (amd_smi_system.cc),
// the layer amdsmi_get_gpu_cpu_links() (amd_smi.cc) relies on to turn a raw
// NUMA node id into a physical CPU package id. Exercises it directly against
// a mock "/sys/devices/system" tree via SetSysDevicesSystemRootForTesting(),
// the same hook style cpu_link_discovery_test.cc uses for KFD topology -- no
// GPU or real sysfs involved.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_system.h"

namespace {

class ScopedTempDir {
 public:
  ScopedTempDir() {
    std::string tmpl =
        (std::filesystem::temp_directory_path() / "cpu_socket_resolution_test_XXXXXX").string();
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

class ScopedSysRootOverride {
 public:
  explicit ScopedSysRootOverride(const std::string& path) : path_(path) {
    amd::smi::SetSysDevicesSystemRootForTesting(path_.c_str());
  }
  ~ScopedSysRootOverride() { amd::smi::SetSysDevicesSystemRootForTesting(nullptr); }

 private:
  std::string path_;
};

void WriteFile(const std::filesystem::path& path, const std::string& content) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path) << content;
}

void WriteCpulist(const std::filesystem::path& root, uint32_t numa_node,
                  const std::string& cpulist) {
  WriteFile(root / "node" / ("node" + std::to_string(numa_node)) / "cpulist", cpulist + "\n");
}

void WritePhysicalPackageId(const std::filesystem::path& root, uint32_t cpu_index,
                            uint32_t package_id) {
  WriteFile(root / "cpu" / ("cpu" + std::to_string(cpu_index)) / "topology" / "physical_package_id",
            std::to_string(package_id) + "\n");
}

}  // namespace

TEST(SystemUnit, CpuSocketsFromNumaNodeNegativeIsSentinel) {
  // No sysfs involved: a negative numa_node short-circuits to the
  // "unresolved NUMA affinity" sentinel used elsewhere in the codebase.
  auto sockets = amd::smi::AMDSmiSystem::getInstance().get_cpu_sockets_from_numa_node(-1);

  ASSERT_EQ(sockets.size(), 1u);
  EXPECT_EQ(sockets[0], static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
}

TEST(SystemUnit, CpuSocketsFromNumaNodeDedupsSamePackage) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedSysRootOverride root_override(root.path().string());

  // NUMA node 0 spans two CPUs that both live on physical package 3.
  WriteCpulist(root.path(), 0, "0-1");
  WritePhysicalPackageId(root.path(), 0, 3);
  WritePhysicalPackageId(root.path(), 1, 3);

  auto sockets = amd::smi::AMDSmiSystem::getInstance().get_cpu_sockets_from_numa_node(0);

  ASSERT_EQ(sockets.size(), 1u);
  EXPECT_EQ(sockets[0], 3u);
}

TEST(SystemUnit, CpuSocketsFromNumaNodeParsesRangesAndCommas) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedSysRootOverride root_override(root.path().string());

  // A comma-separated mix of ranges and single indices, as real cpulist files use
  // (e.g. "0-31,64-95"): every member CPU must be visited, not just the first.
  WriteCpulist(root.path(), 0, "0-1,3");
  WritePhysicalPackageId(root.path(), 0, 3);
  WritePhysicalPackageId(root.path(), 1, 5);
  WritePhysicalPackageId(root.path(), 3, 7);

  auto sockets = amd::smi::AMDSmiSystem::getInstance().get_cpu_sockets_from_numa_node(0);

  ASSERT_EQ(sockets.size(), 3u);
  EXPECT_EQ(sockets[0], 3u);
  EXPECT_EQ(sockets[1], 5u);
  EXPECT_EQ(sockets[2], 7u);
}

TEST(SystemUnit, CpuSocketsFromNumaNodeUnresolvedIsEmpty) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedSysRootOverride root_override(root.path().string());

  // NUMA node 0 exists and lists CPU 0, but CPU 0's topology dir is missing
  // (e.g. a masked /sys/devices/system/cpu) -- nothing resolves to a package.
  WriteCpulist(root.path(), 0, "0");

  auto sockets = amd::smi::AMDSmiSystem::getInstance().get_cpu_sockets_from_numa_node(0);

  EXPECT_TRUE(sockets.empty());
}

TEST(SystemUnit, CpuSocketsFromNumaNodeSkipsMalformedTokens) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedSysRootOverride root_override(root.path().string());

  // A corrupt cpulist mixing whitespace, non-numeric, and empty (double-comma)
  // tokens with one real entry: only the real entry should resolve.
  WriteCpulist(root.path(), 0, " ,abc,,1");
  WritePhysicalPackageId(root.path(), 1, 7);

  auto sockets = amd::smi::AMDSmiSystem::getInstance().get_cpu_sockets_from_numa_node(0);

  ASSERT_EQ(sockets.size(), 1u);
  EXPECT_EQ(sockets[0], 7u);
}

TEST(SystemUnit, CpuSocketsFromNumaNodeRejectsHugeOrNegativeRangesWithoutHanging) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedSysRootOverride root_override(root.path().string());

  // Each of these previously drove the first-last loop into billions of
  // iterations (or an outright infinite loop via uint32_t wraparound). They
  // must all resolve to no sockets, and the test itself must complete.
  const char* kMalformedCpulists[] = {"-1", "0--1", "0-4294967295", "99999999999"};
  int numa_node = 0;
  for (const char* cpulist : kMalformedCpulists) {
    WriteCpulist(root.path(), numa_node, cpulist);
    auto sockets = amd::smi::AMDSmiSystem::getInstance().get_cpu_sockets_from_numa_node(numa_node);
    EXPECT_TRUE(sockets.empty()) << "cpulist=\"" << cpulist << "\"";
    ++numa_node;
  }
}

TEST(SystemUnit, CpuSocketsFromNumaNodeRejectsNegativePhysicalPackageId) {
  ScopedTempDir root;
  ASSERT_FALSE(root.path().empty());
  ScopedSysRootOverride root_override(root.path().string());

  // A CPU with no topology yet can report physical_package_id "-1"; that must
  // not wrap to UINT32_MAX and be treated as a real package.
  WriteCpulist(root.path(), 0, "0");
  WriteFile(root.path() / "cpu" / "cpu0" / "topology" / "physical_package_id", "-1\n");

  auto sockets = amd::smi::AMDSmiSystem::getInstance().get_cpu_sockets_from_numa_node(0);

  EXPECT_TRUE(sockets.empty());
}
