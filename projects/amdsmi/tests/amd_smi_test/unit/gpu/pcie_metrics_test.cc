// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_common.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"

namespace {
amd::smi::Device* test_device = nullptr;
amd::smi::AMDSmiGPUDevice* test_gpu = nullptr;
std::string sysfs_prefix;
std::string fixture_prefix;
std::map<std::string, int> open_errors;
int link_read_error = 0;
std::string link_read_attribute = "current_link_speed";
std::string link_read_prefix;
std::size_t link_read_offset = 0;
int metrics_read_error = 0;

class GpuUnit : public ::testing::Test {
 protected:
  void SetUp() override {
    auto pattern = (std::filesystem::temp_directory_path() / "amdsmi-pcie-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const char* directory = mkdtemp(buffer.data());
    ASSERT_NE(directory, nullptr);
    root_ = directory;
    std::filesystem::create_directory(root_ / "device");
    std::ofstream(root_ / "device/gpu_metrics").put('\0');
    auto& smi = amd::smi::RocmSMI::getInstance();
    ASSERT_TRUE(smi.devices().empty());
    device_ = std::make_shared<amd::smi::Device>(root_.string(), &smi.getEnv());
    smi.devices().push_back(device_);
    test_device = device_.get();
    metrics_read_error = ENOTSUP;
    gpu_ = std::make_unique<amd::smi::AMDSmiGPUDevice>(0, root_.filename().string(), amdsmi_bdf_t{},
                                                       drm_);
    test_gpu = gpu_.get();
    sysfs_prefix = "/sys/class/drm/" + root_.filename().string() + "/device/";
    fixture_prefix = root_.string() + "/device/";
    amd::smi::amdsmi_library_init_ref_acquire();
    initialized_ = true;
    Write("max_link_width", "16\n");
    Write("max_link_speed", "32.0 GT/s PCIe\n");
    Write("current_link_width", "8\n");
    Write("current_link_speed", "2.5 GT/s PCIe\n");
  }

  void TearDown() override {
    if (initialized_) amd::smi::amdsmi_library_init_ref_release();
    test_gpu = nullptr;
    gpu_.reset();
    test_device = nullptr;
    metrics_read_error = 0;
    link_read_error = 0;
    link_read_attribute = "current_link_speed";
    link_read_prefix.clear();
    link_read_offset = 0;
    open_errors.clear();
    sysfs_prefix.clear();
    fixture_prefix.clear();
    amd::smi::RocmSMI::getInstance().devices().clear();
    device_.reset();
    if (!root_.empty()) std::filesystem::remove_all(root_);
  }

  void Write(const std::string& name, const std::string& value) {
    std::ofstream(root_ / "device" / name) << value;
  }

  void WriteMetrics(uint16_t width, uint16_t speed, uint64_t replay = 0) {
    amd::smi::AMDGpuMetrics_v14_t metrics{};
    metrics.m_common_header = {sizeof(metrics), 1, 4};
    metrics.m_pcie_link_width = width;
    metrics.m_pcie_link_speed = speed;
    metrics.m_pcie_replay_count_acc = replay;
    std::ofstream stream(root_ / "device/gpu_metrics", std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&metrics), sizeof(metrics));
    metrics_read_error = 0;
  }

  std::filesystem::path root_;
  std::shared_ptr<amd::smi::Device> device_;
  amd::smi::AMDSmiDrm drm_;
  std::unique_ptr<amd::smi::AMDSmiGPUDevice> gpu_;
  bool initialized_ = false;
};

TEST_F(GpuUnit, PcieLoggingPreservesUnsupportedRead) {
  std::ostringstream output;
  EXPECT_EQ(device_->dev_log_gpu_metrics(output), RSMI_STATUS_NOT_SUPPORTED);
}

TEST_F(GpuUnit, PcieMetricsQueryPreservesUnsupportedRead) {
  rsmi_gpu_metrics_t metrics{};
  EXPECT_EQ(rsmi_dev_gpu_metrics_info_get(0, &metrics), RSMI_STATUS_NOT_SUPPORTED);
}

TEST_F(GpuUnit, PcieUnsupportedMetricsPreserveCurrentLink) {
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_static.max_pcie_width, 16);
  EXPECT_EQ(info.pcie_static.max_pcie_speed, 32000);
  EXPECT_EQ(info.pcie_metric.pcie_width, 8);
  EXPECT_EQ(info.pcie_metric.pcie_speed, 2500);
  EXPECT_EQ(info.pcie_metric.pcie_bandwidth, UINT32_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_replay_count, UINT64_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_l0_to_recovery_count, UINT64_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_replay_roll_over_count, UINT64_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_nak_sent_count, UINT64_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_nak_received_count, UINT64_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_lc_perf_other_end_recovery_count, UINT32_MAX);
  for (auto value : info.pcie_metric.reserved) EXPECT_EQ(value, 0);
  for (auto value : info.reserved) EXPECT_EQ(value, 0);
}

TEST_F(GpuUnit, PcieMissingCurrentWidthKeepsSpeed) {
  std::filesystem::remove(root_ / "device/current_link_width");
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_metric.pcie_width, UINT16_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_speed, 2500);
}

TEST_F(GpuUnit, PcieMissingCurrentSpeedKeepsWidth) {
  std::filesystem::remove(root_ / "device/current_link_speed");
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_metric.pcie_width, 8);
  EXPECT_EQ(info.pcie_metric.pcie_speed, UINT32_MAX);
}

TEST_F(GpuUnit, PcieUnsupportedCurrentLinksKeepStaticInfo) {
  open_errors = {{"current_link_width", ENOTSUP}, {"current_link_speed", ENOTSUP}};
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_static.max_pcie_width, 16);
  EXPECT_EQ(info.pcie_static.max_pcie_speed, 32000);
  EXPECT_EQ(info.pcie_metric.pcie_width, UINT16_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_speed, UINT32_MAX);
}

TEST_F(GpuUnit, PcieCurrentLinkOpenFailuresKeepStaticInfo) {
  for (int error : {EACCES, EPERM, EIO}) {
    SCOPED_TRACE(error);
    open_errors = {{"current_link_width", error}, {"current_link_speed", error}};
    amdsmi_pcie_info_t info{};
    ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(info.pcie_static.max_pcie_width, 16);
    EXPECT_EQ(info.pcie_static.max_pcie_speed, 32000);
    EXPECT_EQ(info.pcie_metric.pcie_width, UINT16_MAX);
    EXPECT_EQ(info.pcie_metric.pcie_speed, UINT32_MAX);
  }
}

TEST_F(GpuUnit, PcieUnparsableCurrentWidthStaysUnavailable) {
  for (const char* value : {"", "nan\n", "lanes\n"}) {
    SCOPED_TRACE(value);
    Write("current_link_width", value);
    amdsmi_pcie_info_t info{};
    ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(info.pcie_metric.pcie_width, UINT16_MAX);
    EXPECT_EQ(info.pcie_metric.pcie_speed, 2500);
  }
}

TEST_F(GpuUnit, PcieInvalidCurrentWidthStaysUnavailable) {
  for (const std::string& value :
       std::vector<std::string>{"-8\n", "65535\n", "65536\n", "70000\n", "4294967297\n", "1.5\n",
                                "8 lanes\n", "8\njunk\n", std::string(1024, '9') + "\n"}) {
    SCOPED_TRACE(value);
    Write("current_link_width", value);
    amdsmi_pcie_info_t info{};
    ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(info.pcie_metric.pcie_width, UINT16_MAX);
    EXPECT_EQ(info.pcie_metric.pcie_speed, 2500);
    EXPECT_EQ(info.pcie_static.max_pcie_width, 16);
  }
}

TEST_F(GpuUnit, PcieUnparsableCurrentSpeedStaysUnavailable) {
  for (const char* value : {"", "GT/s\n", "unknown\n"}) {
    SCOPED_TRACE(value);
    Write("current_link_speed", value);
    amdsmi_pcie_info_t info{};
    ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(info.pcie_metric.pcie_speed, UINT32_MAX);
    EXPECT_EQ(info.pcie_metric.pcie_width, 8);
  }
}

TEST_F(GpuUnit, PcieCurrentSpeedsConvertToMTs) {
  for (const auto& value : std::map<std::string, uint32_t>{{"2.5", 2500},
                                                           {"5", 5000},
                                                           {"8", 8000},
                                                           {"16", 16000},
                                                           {"32.0", 32000},
                                                           {"64", 64000},
                                                           {"128", 128000},
                                                           {"0.001", 1},
                                                           {"4294967.294", UINT32_MAX - 1}}) {
    SCOPED_TRACE(value.first);
    for (const char* suffix : {" GT/s PCIe\n", " GT/s\n", " GT/s", " GT/s PCIe \t\n"}) {
      SCOPED_TRACE(suffix);
      Write("current_link_speed", value.first + suffix);
      amdsmi_pcie_info_t info{};
      ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(info.pcie_metric.pcie_speed, value.second);
    }
  }
}

TEST_F(GpuUnit, PcieInvalidCurrentSpeedStaysUnavailable) {
  for (const std::string& value : std::vector<std::string>{
           "nan GT/s\n", "inf GT/s\n", "-inf GT/s\n", "-2.5 GT/s\n", "0 GT/s\n",
           "4294967.295 GT/s\n", "4294967.296 GT/s\n", "1e20 GT/s\n", "0x1p200 GT/s\n", "32\n",
           "32 GB/s\n", "32 GT/s Xyz\n", "32 GT/s garbage\n", "32 GT/s PCIe garbage\n",
           "32 GT/s PCIe\njunk\n", "1.2345 GT/s\n", "2.5.0 GT/s\n",
           std::string(1024, '9') + " GT/s\n"}) {
    SCOPED_TRACE(value);
    Write("current_link_speed", value);
    amdsmi_pcie_info_t info{};
    ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(info.pcie_metric.pcie_speed, UINT32_MAX);
    EXPECT_EQ(info.pcie_metric.pcie_width, 8);
    EXPECT_EQ(info.pcie_static.max_pcie_speed, 32000);
  }
}

TEST_F(GpuUnit, PcieValidMetricsTakePrecedence) {
  WriteMetrics(4, 80, 17);
  open_errors = {{"current_link_width", EACCES}, {"current_link_speed", EACCES}};
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_metric.pcie_width, 4);
  EXPECT_EQ(info.pcie_metric.pcie_speed, 8000);
  EXPECT_EQ(info.pcie_metric.pcie_replay_count, 17);
  EXPECT_EQ(info.pcie_metric.pcie_bandwidth, 0);
  EXPECT_EQ(info.pcie_metric.pcie_nak_sent_count, UINT64_MAX);
}

TEST_F(GpuUnit, PcieValidMetricsPreserveZeroCounters) {
  WriteMetrics(16, 320);
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_metric.pcie_replay_count, 0);
  EXPECT_EQ(info.pcie_metric.pcie_bandwidth, 0);
}

TEST_F(GpuUnit, PcieValidMetricsPreserveLinkNaMarkers) {
  WriteMetrics(UINT16_MAX, UINT16_MAX);
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_metric.pcie_width, UINT16_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_speed, UINT32_MAX);
}

TEST_F(GpuUnit, PcieMetricsDoNotReturnStaleDataAfterReadFailure) {
  WriteMetrics(16, 320);
  rsmi_gpu_metrics_t metrics{};
  ASSERT_EQ(rsmi_dev_gpu_metrics_info_get(0, &metrics), RSMI_STATUS_SUCCESS);
  metrics_read_error = ENOTSUP;
  EXPECT_EQ(rsmi_dev_gpu_metrics_info_get(0, &metrics), RSMI_STATUS_NOT_SUPPORTED);
}

TEST_F(GpuUnit, PcieMetricsErrorsDoNotUseLinkFallback) {
  for (const auto& error : std::map<int, amdsmi_status_t>{{EACCES, AMDSMI_STATUS_NO_PERM},
                                                          {EIO, AMDSMI_STATUS_UNEXPECTED_SIZE},
                                                          {ENXIO, AMDSMI_STATUS_UNEXPECTED_DATA}}) {
    SCOPED_TRACE(error.first);
    metrics_read_error = error.first;
    amdsmi_pcie_info_t info{};
    EXPECT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), error.second);
  }
}

TEST_F(GpuUnit, PcieNullOutputIsRejected) {
  EXPECT_EQ(amdsmi_get_pcie_info(gpu_.get(), nullptr), AMDSMI_STATUS_INVAL);
}

TEST_F(GpuUnit, PcieRequiresInitialization) {
  amd::smi::amdsmi_library_init_ref_release();
  initialized_ = false;
  amdsmi_pcie_info_t info{};
  EXPECT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_NOT_INIT);
}

TEST_F(GpuUnit, PcieUnknownCurrentSpeedKeepsStaticInfo) {
  Write("current_link_speed", "Unknown\n");
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_metric.pcie_speed, UINT32_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_width, 8);
  EXPECT_EQ(info.pcie_static.max_pcie_speed, 32000);
}

TEST_F(GpuUnit, PcieLegacyUnknownSpeedKeepsStaticInfo) {
  Write("current_link_speed", "Unknown speed\n");
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_metric.pcie_speed, UINT32_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_width, 8);
  EXPECT_EQ(info.pcie_static.max_pcie_speed, 32000);
}

TEST_F(GpuUnit, PcieCurrentLinksAcceptEofWithoutNewline) {
  Write("current_link_width", "8");
  Write("current_link_speed", "2.5 GT/s PCIe");
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_metric.pcie_width, 8);
  EXPECT_EQ(info.pcie_metric.pcie_speed, 2500);
}

TEST_F(GpuUnit, PcieZeroNegotiatedWidthIsUnavailable) {
  Write("current_link_width", "0\n");
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_metric.pcie_width, UINT16_MAX);
  EXPECT_EQ(info.pcie_metric.pcie_speed, 2500);
}

TEST_F(GpuUnit, PcieUnsupportedMetricsVersionKeepsLinkInfo) {
  const amd::smi::AMDGpuMetricsHeader_v1_t header{sizeof(header), 99, 1};
  {
    std::ofstream stream(root_ / "device/gpu_metrics", std::ios::binary);
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
  }
  metrics_read_error = 0;
  amdsmi_pcie_info_t info{};
  ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.pcie_metric.pcie_width, 8);
  EXPECT_EQ(info.pcie_metric.pcie_speed, 2500);
  EXPECT_EQ(info.pcie_metric.pcie_bandwidth, UINT32_MAX);
}

TEST_F(GpuUnit, PcieAttributeReadFailuresKeepOtherFields) {
  for (const char* attribute : {"current_link_width", "current_link_speed"}) {
    SCOPED_TRACE(attribute);
    link_read_attribute = attribute;
    for (int error : {EACCES, EPERM, ENOTSUP, EIO, ENOENT}) {
      SCOPED_TRACE(error);
      link_read_error = error;
      amdsmi_pcie_info_t info{};
      ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(info.pcie_metric.pcie_width,
                link_read_attribute == "current_link_width" ? UINT16_MAX : 8);
      EXPECT_EQ(info.pcie_metric.pcie_speed,
                link_read_attribute == "current_link_speed" ? UINT32_MAX : 2500);
    }
  }
}

TEST_F(GpuUnit, PciePartialReadFailuresKeepValuesUnavailable) {
  for (const auto& entry : std::map<std::string, std::string>{
           {"current_link_width", "8\n"}, {"current_link_speed", "2.5 GT/s PCIe\n"}}) {
    SCOPED_TRACE(entry.first);
    link_read_attribute = entry.first;
    link_read_prefix = entry.second;
    for (int error : {EACCES, ENOTSUP, EIO}) {
      SCOPED_TRACE(error);
      link_read_offset = 0;
      link_read_error = error;
      amdsmi_pcie_info_t info{};
      ASSERT_EQ(amdsmi_get_pcie_info(gpu_.get(), &info), AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(info.pcie_metric.pcie_width, entry.first == "current_link_width" ? UINT16_MAX : 8);
      EXPECT_EQ(info.pcie_metric.pcie_speed,
                entry.first == "current_link_speed" ? UINT32_MAX : 2500);
      EXPECT_EQ(info.pcie_static.max_pcie_width, 16);
      EXPECT_EQ(info.pcie_static.max_pcie_speed, 32000);
    }
  }
}
}  // namespace

extern "C" amdsmi_status_t
wrap_gpu_lookup(amdsmi_processor_handle, amd::smi::AMDSmiGPUDevice**) asm(
    "__wrap__Z26get_gpu_device_from_handlePvPPN3amd3smi15AMDSmiGPUDeviceE");

extern "C" amdsmi_status_t wrap_gpu_lookup(amdsmi_processor_handle handle,
                                           amd::smi::AMDSmiGPUDevice** device) {
  if (!device || !handle || handle != test_gpu) return AMDSMI_STATUS_INVAL;
  *device = test_gpu;
  return AMDSMI_STATUS_SUCCESS;
}

extern "C" FILE* __real_fopen(const char* path, const char* mode);
extern "C" FILE* __wrap_fopen(const char* path, const char* mode) {
  const std::string name(path);
  if (!sysfs_prefix.empty() && name.compare(0, sysfs_prefix.size(), sysfs_prefix) == 0) {
    const auto attribute = name.substr(sysfs_prefix.size());
    if (attribute == link_read_attribute && link_read_error != 0) {
      cookie_io_functions_t io{};
      io.read = [](void*, char* buffer, size_t size) -> ssize_t {
        if (link_read_offset < link_read_prefix.size()) {
          const auto count = link_read_prefix.copy(buffer, size, link_read_offset);
          link_read_offset += count;
          return static_cast<ssize_t>(count);
        }
        errno = link_read_error;
        return -1;
      };
      return fopencookie(nullptr, "r", io);
    }
    const auto error = open_errors.find(attribute);
    if (error != open_errors.end()) {
      errno = error->second;
      return nullptr;
    }
    const auto redirected = fixture_prefix + attribute;
    return __real_fopen(redirected.c_str(), mode);
  }
  return __real_fopen(path, mode);
}

// Intercept only the binary device-I/O boundary; metrics setup and conversion remain real.
extern "C" int real_read_metrics(amd::smi::Device*, amd::smi::DevInfoTypes, std::size_t, void*) asm(
    "__real__ZN3amd3smi6Device11readDevInfoENS0_12DevInfoTypesEmPv");
extern "C" int wrap_read_metrics(amd::smi::Device*, amd::smi::DevInfoTypes, std::size_t, void*) asm(
    "__wrap__ZN3amd3smi6Device11readDevInfoENS0_12DevInfoTypesEmPv");

extern "C" int wrap_read_metrics(amd::smi::Device* device, amd::smi::DevInfoTypes type,
                                 std::size_t size, void* data) {
  if (device == test_device && type == amd::smi::kDevGpuMetrics && metrics_read_error != 0) {
    return metrics_read_error;
  }
  return real_read_metrics(device, type, size, data);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
