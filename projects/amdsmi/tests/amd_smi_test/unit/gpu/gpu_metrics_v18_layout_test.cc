// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gpu_metrics v1.8 and xcp_metrics v1.0 are fixed C structs written by amdgpu
// (struct gpu_metrics_v1_8 and struct amdgpu_partition_metrics_v1_0 in
// drivers/gpu/drm/amd/include/kgd_pp_interface.h). These tests write blobs
// laid out at the kernel's byte offsets into a fake sysfs tree and read them
// back through the public rsmi entry points.

#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_gpu_metrics.h"
#include "rocm_smi/rocm_smi_main.h"

namespace {

// Kernel layout of struct amdgpu_xcp_metrics_v1_2 (MAX_XCC 8, NUM_JPEG_ENG_V1 40, NUM_VCN 4).
constexpr std::size_t kXcpGfxBusyInst = 0;
constexpr std::size_t kXcpGfxBelowHostLimitTotalAcc = 376;
constexpr std::size_t kXcpSize = 440;

// Kernel layout of struct gpu_metrics_v1_8.
constexpr std::size_t kV18XcpStats = 344;
constexpr std::size_t kV18PcieLcPerfOtherEndRecovery = 3864;
constexpr std::size_t kV18Size = 3872;

// Kernel layout of struct amdgpu_partition_metrics_v1_0.
constexpr std::size_t kXcpV10GfxBusyInst = 48;
constexpr std::size_t kXcpV10GfxBelowHostLimitTotalAcc = 424;
constexpr std::size_t kXcpV10Size = 488;

constexpr uint16_t kNotAvailable16 = std::numeric_limits<uint16_t>::max();

template <typename T>
void Put(std::vector<uint8_t>& blob, std::size_t offset, T value) {
  ASSERT_LE(offset + sizeof(T), blob.size());
  std::memcpy(blob.data() + offset, &value, sizeof(T));
}

auto MetricsBlob(std::size_t size, uint8_t format_revision, uint8_t content_revision)
    -> std::vector<uint8_t> {
  std::vector<uint8_t> blob(size, 0);
  amd::smi::AMDGpuMetricsHeader_v1_t header{};
  header.m_structure_size = static_cast<uint16_t>(size);
  header.m_format_revision = format_revision;
  header.m_content_revision = content_revision;
  std::memcpy(blob.data(), &header, sizeof(header));
  return blob;
}

class FakeSysfsDevice {
 public:
  explicit FakeSysfsDevice(const std::string& name)
      : root_(std::filesystem::temp_directory_path() /
              ("amdsmi_v18_layout_" + name + "_" + std::to_string(::getpid()) + "_" +
               std::to_string(NextId()))) {
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_ / "device" / "xcp");
  }
  ~FakeSysfsDevice() { std::filesystem::remove_all(root_); }

  FakeSysfsDevice(const FakeSysfsDevice&) = delete;
  FakeSysfsDevice& operator=(const FakeSysfsDevice&) = delete;

  void Write(const std::string& relative, const std::vector<uint8_t>& blob) const {
    std::ofstream stream(root_ / "device" / relative, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(blob.data()),
                 static_cast<std::streamsize>(blob.size()));
  }

  std::string path() const { return root_.string(); }

 private:
  static unsigned NextId() {
    static std::atomic<unsigned> counter{0};
    return counter.fetch_add(1);
  }

  std::filesystem::path root_;
};

class ScopedLibraryDevice {
 public:
  explicit ScopedLibraryDevice(const std::string& device_path)
      : devices_(amd::smi::RocmSMI::getInstance().devices()), original_count_(devices_.size()) {
    devices_.push_back(std::make_shared<amd::smi::Device>(device_path, nullptr));
  }
  ~ScopedLibraryDevice() { devices_.resize(original_count_); }

  ScopedLibraryDevice(const ScopedLibraryDevice&) = delete;
  ScopedLibraryDevice& operator=(const ScopedLibraryDevice&) = delete;

  uint32_t index() const { return static_cast<uint32_t>(original_count_); }

 private:
  std::vector<std::shared_ptr<amd::smi::Device>>& devices_;
  std::size_t original_count_;
};

}  // namespace

TEST(GpuUnit, GPUMetricsV18ReadsKernelLayout) {
  auto blob = MetricsBlob(kV18Size, 1, 8);
  for (std::size_t xcp = 0; xcp < RSMI_MAX_NUM_XCP; ++xcp) {
    const std::size_t base = kV18XcpStats + xcp * kXcpSize;
    Put<uint32_t>(blob, base + kXcpGfxBusyInst, 0x1000 + static_cast<uint32_t>(xcp));
    Put<uint64_t>(blob, base + kXcpGfxBelowHostLimitTotalAcc + 7 * sizeof(uint64_t),
                  0xA000000000000000ULL + xcp);
  }
  Put<uint32_t>(blob, kV18PcieLcPerfOtherEndRecovery, 0xC0FFEE);

  FakeSysfsDevice fake("gpu_v1_8");
  fake.Write("gpu_metrics", blob);
  const ScopedLibraryDevice device(fake.path());

  rsmi_gpu_metrics_t metrics{};
  ASSERT_EQ(rsmi_dev_gpu_metrics_info_get(device.index(), &metrics), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(metrics.common_header.content_revision, 8);

  for (std::size_t xcp = 0; xcp < RSMI_MAX_NUM_XCP; ++xcp) {
    SCOPED_TRACE(testing::Message() << "xcp_stats[" << xcp << "]");
    EXPECT_EQ(metrics.xcp_stats[xcp].gfx_busy_inst[0], 0x1000 + xcp);
    EXPECT_EQ(metrics.xcp_stats[xcp].gfx_below_host_limit_total_acc[7],
              0xA000000000000000ULL + xcp);
    EXPECT_EQ(metrics.xcp_stats[xcp].temperature_xcd[0], kNotAvailable16)
        << "v1.8 carries no XCD temperature";
  }
  EXPECT_EQ(metrics.pcie_lc_perf_other_end_recovery, 0xC0FFEEu);
}

TEST(GpuUnit, XCPMetricsV10ReadsKernelLayout) {
  auto blob = MetricsBlob(kXcpV10Size, 1, 0);
  Put<uint32_t>(blob, kXcpV10GfxBusyInst, 0x1234);
  Put<uint64_t>(blob, kXcpV10GfxBelowHostLimitTotalAcc + 7 * sizeof(uint64_t),
                0xB000000000000007ULL);

  FakeSysfsDevice fake("xcp_v1_0");
  fake.Write("xcp/xcp_metrics", blob);
  const ScopedLibraryDevice device(fake.path());

  rsmi_gpu_metrics_t metrics{};
  ASSERT_EQ(rsmi_dev_gpu_partition_metrics_info_get(device.index(), &metrics), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(metrics.xcp_stats[0].gfx_busy_inst[0], 0x1234u);
  EXPECT_EQ(metrics.xcp_stats[0].gfx_below_host_limit_total_acc[7], 0xB000000000000007ULL);
  EXPECT_EQ(metrics.xcp_stats[0].temperature_xcd[0], kNotAvailable16)
      << "xcp_metrics v1.0 carries no XCD temperature";
}
