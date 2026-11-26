// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

#include "amd_smi/amdsmi.h"

namespace {

const std::string kKfdNodes = "/sys/class/kfd/kfd/topology/nodes/";
constexpr int64_t kKfdIoLinkTypeXgmi = 11;

// Value of `key` in a KFD sysfs properties file, or -1 if it is missing.
int64_t KfdProperty(const std::string& file, const std::string& key) {
  std::ifstream in(file);
  std::string name;
  int64_t value = 0;
  while (in >> name >> value) {
    if (name == key) return value;
  }
  return -1;
}

int64_t KfdGpuId(int64_t node) {
  std::ifstream in(kKfdNodes + std::to_string(node) + "/gpu_id");
  int64_t gpu_id = -1;
  in >> gpu_id;
  return gpu_id;
}

// Type of the lowest-weight io_link from KFD node `node` to a CPU node (gpu_id 0), or -1.
// Equal weights go to the lower CPU node, as in the library.
int64_t CpuLinkType(uint32_t node) {
  int64_t type = -1;
  int64_t best_weight = std::numeric_limits<int64_t>::max();
  int64_t best_to = std::numeric_limits<int64_t>::max();
  std::error_code ec;
  for (const auto& link :
       std::filesystem::directory_iterator(kKfdNodes + std::to_string(node) + "/io_links", ec)) {
    const std::string props = link.path().string() + "/properties";
    const int64_t to = KfdProperty(props, "node_to");
    const int64_t weight = KfdProperty(props, "weight");
    if (KfdGpuId(to) != 0) continue;
    if (weight < best_weight || (weight == best_weight && to < best_to)) {
      best_weight = weight;
      best_to = to;
      type = KfdProperty(props, "type");
    }
  }
  return type;
}

}  // namespace

// A GPU that reaches its CPU over xGMI has no PCIe host link, so the static PCIe caps (read
// from an emulated config space) must be reported as unavailable. Other GPUs keep real values.
TEST(GpuFunctionalReadOnly, PcieStaticCapsUnavailableOnlyWithXgmiHostLink) {
  amdsmi_status_t status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  if (status == AMDSMI_STATUS_DRIVER_NOT_LOADED) GTEST_SKIP() << "amdgpu driver not loaded";
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

  std::vector<amdsmi_processor_handle> gpus;
  uint32_t socket_count = 0;
  ASSERT_EQ(amdsmi_get_socket_handles(&socket_count, nullptr), AMDSMI_STATUS_SUCCESS);
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  ASSERT_EQ(amdsmi_get_socket_handles(&socket_count, sockets.data()), AMDSMI_STATUS_SUCCESS);
  for (auto socket : sockets) {
    uint32_t count = 0;
    ASSERT_EQ(amdsmi_get_processor_handles(socket, &count, nullptr), AMDSMI_STATUS_SUCCESS);
    std::vector<amdsmi_processor_handle> handles(count);
    ASSERT_EQ(amdsmi_get_processor_handles(socket, &count, handles.data()), AMDSMI_STATUS_SUCCESS);
    gpus.insert(gpus.end(), handles.begin(), handles.begin() + count);
  }

  int checked = 0;
  for (auto gpu : gpus) {
    amdsmi_kfd_info_t kfd{};
    if (amdsmi_get_gpu_kfd_info(gpu, &kfd) != AMDSMI_STATUS_SUCCESS) continue;
    if (KfdGpuId(kfd.node_id) != static_cast<int64_t>(kfd.kfd_id)) continue;
    const int64_t cpu_link_type = CpuLinkType(kfd.node_id);
    if (cpu_link_type < 0) continue;  // KFD lists no CPU link to compare with
    amdsmi_pcie_info_t pcie{};
    if (amdsmi_get_pcie_info(gpu, &pcie) != AMDSMI_STATUS_SUCCESS) continue;

    const bool caps_unavailable =
        pcie.pcie_static.max_pcie_width == std::numeric_limits<uint16_t>::max() &&
        pcie.pcie_static.max_pcie_speed == std::numeric_limits<uint32_t>::max() &&
        pcie.pcie_static.pcie_interface_version == std::numeric_limits<uint32_t>::max();
    EXPECT_EQ(caps_unavailable, cpu_link_type == kKfdIoLinkTypeXgmi)
        << "KFD node " << kfd.node_id << ", CPU io_link type " << cpu_link_type << ": width "
        << pcie.pcie_static.max_pcie_width << ", speed " << pcie.pcie_static.max_pcie_speed
        << ", version " << pcie.pcie_static.pcie_interface_version;
    ++checked;
  }
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
  if (checked == 0) GTEST_SKIP() << "no GPU with a CPU io_link in KFD";
}
