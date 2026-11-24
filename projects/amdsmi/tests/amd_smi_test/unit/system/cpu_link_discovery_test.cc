// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for amd::smi::DiscoverGpuCpuLinks() (rocm_smi_cpu_link.cc), the
// backing layer for amdsmi_get_gpu_cpu_links(). Exercises it directly against
// a mock KFD topology via the KfdNodesPathRoot test hook, the same hook used
// by kfd_numa_node_test.cc in this directory -- no GPU or node_handle/board
// fabrication involved, so there is no WSLGPUBackend short-circuit to guard
// against here (DiscoverGpuCpuLinks and its callees have no
// ENABLE_WSL_BACKEND branch; WSLGPUBackend only intercepts the amd_smi.cc
// public-API layer, which this test does not go through).

#include <gtest/gtest.h>

#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "rocm_smi/rocm_smi_cpu_link.h"
#include "rocm_smi/rocm_smi_io_link.h"
#include "test_fixture_utils.h"

namespace {

// TEMP DIAGNOSTIC -- not for commit. Runs the real `amd-smi xgmi` CLI (if
// findable) and dumps its full stdout+stderr, prefixed so it's easy to grep
// out of the CI job log alongside the DiscoverGpuCpuLinks() errno tracing in
// rocm_smi_cpu_link.cc.
std::string RunCommandCaptureOutput(const std::string& cmd) {
  std::string full_cmd = cmd + " 2>&1";
  FILE* pipe = popen(full_cmd.c_str(), "r");
  if (!pipe) return "<popen failed>";
  std::array<char, 4096> buf{};
  std::string out;
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
    out += buf.data();
  }
  int status = pclose(pipe);
  out += "\n[exit status: " + std::to_string(status) + "]";
  return out;
}

TEST(SystemUnit, DebugDumpXgmiCliOutput) {
  std::array<char, 4096> self_path{};
  ssize_t len = readlink("/proc/self/exe", self_path.data(), self_path.size() - 1);
  std::string exe_dir;
  if (len > 0) {
    self_path[static_cast<size_t>(len)] = '\0';
    std::string exe(self_path.data());
    auto pos = exe.find_last_of('/');
    if (pos != std::string::npos) exe_dir = exe.substr(0, pos);
  }

  std::vector<std::string> candidates = {"amd-smi xgmi"};
  if (!exe_dir.empty()) {
    // amdsmitst typically lives at <prefix>/share/amd_smi/tests/amdsmitst.
    candidates.push_back(exe_dir + "/../../../bin/amd-smi xgmi");
    candidates.push_back(exe_dir + "/../../../../bin/amd-smi xgmi");
  }

  for (const auto& cmd : candidates) {
    std::cerr << "[CPU_LINK_DEBUG][xgmi-cli] trying: " << cmd << "\n";
    std::string out = RunCommandCaptureOutput(cmd);
    std::cerr << "[CPU_LINK_DEBUG][xgmi-cli] ---- output begin ----\n"
              << out << "\n[CPU_LINK_DEBUG][xgmi-cli] ---- output end ----\n";
  }
}

// MI450-like topology: GPU node 1 has a direct XGMI io_link to CPU node 0.
TEST(SystemUnit, DiscoverGpuCpuLinksFindsDirectXgmiCpuLink) {
  FakeSysfsTree tree;
  WriteCpuNode(tree, /*node_idx=*/0);
  WriteGpuNode(tree, /*node_idx=*/1, /*card_idx=*/0, /*gpu_id=*/12345);
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/0, /*node_to=*/0, amd::smi::IOLINK_TYPE_XGMI,
              /*weight=*/15);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);

  ASSERT_EQ(links.size(), 1u);
  EXPECT_EQ(links[0].numa_node_id, 0u);
  EXPECT_EQ(links[0].type, amd::smi::IOLINK_TYPE_XGMI);
}

// GPU node 1's only io_link neighbor is peer GPU node 2 (not a CPU) -- no CPU
// link exists, which must be reported as an empty, successful result.
TEST(SystemUnit, DiscoverGpuCpuLinksEmptyWhenOnlyPeerGpuNeighbor) {
  FakeSysfsTree tree;
  WriteGpuNode(tree, /*node_idx=*/1, /*card_idx=*/0, /*gpu_id=*/12345);
  WriteGpuNode(tree, /*node_idx=*/2, /*card_idx=*/1, /*gpu_id=*/23456);  // peer GPU, not a CPU node
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/0, /*node_to=*/2, amd::smi::IOLINK_TYPE_XGMI,
              /*weight=*/15);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);
  EXPECT_TRUE(links.empty());
}

// GPU node 1 has both a peer-GPU XGMI neighbor and a direct CPU PCIe
// neighbor -- only the CPU link must be reported.
TEST(SystemUnit, DiscoverGpuCpuLinksIgnoresNonCpuNeighbors) {
  FakeSysfsTree tree;
  WriteCpuNode(tree, /*node_idx=*/0);
  WriteGpuNode(tree, /*node_idx=*/1, /*card_idx=*/0, /*gpu_id=*/12345);
  WriteGpuNode(tree, /*node_idx=*/2, /*card_idx=*/1, /*gpu_id=*/23456);  // peer GPU, not a CPU node
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/0, /*node_to=*/2, amd::smi::IOLINK_TYPE_XGMI,
              /*weight=*/15);
  WriteIoLink(tree, /*node_idx=*/1, /*link_idx=*/1, /*node_to=*/0, amd::smi::IOLINK_TYPE_PCIEXPRESS,
              /*weight=*/20);
  amd::smi::testing::ScopedOverride<std::string> kfd_root(amd::smi::test_hooks::KfdNodesPathRoot,
                                                          amd::smi::test_hooks::SetKfdNodesPathRoot,
                                                          tree.root());

  std::vector<amd::smi::GpuCpuLink> links;
  ASSERT_EQ(amd::smi::DiscoverGpuCpuLinks(1, &links), 0);

  ASSERT_EQ(links.size(), 1u);
  EXPECT_EQ(links[0].numa_node_id, 0u);
  EXPECT_EQ(links[0].type, amd::smi::IOLINK_TYPE_PCIEXPRESS);
}

}  // namespace
