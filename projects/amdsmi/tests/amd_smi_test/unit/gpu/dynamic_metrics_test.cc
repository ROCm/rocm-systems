// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <amd_smi_test/test_base.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_gpu_metrics.h"
#include "test_common.h"

namespace amd::smi {

// Forward declarations of internal helpers we exercise in this unit-test.
AMDGpuMetricVersionFlags_t translate_header_to_flag_version(
    const AMDGpuMetricsHeader_v1_t& metrics_header, bool is_partition_metrics,
    const std::string& file_path);

GpuMetricsBasePtr amdgpu_metrics_factory(AMDGpuMetricVersionFlags_t gpu_metric_version,
                                         bool is_partition_metrics, const std::string& file_path);

}  // namespace amd::smi

namespace {
// Version helper checker
auto GetExpectedMetricVersionFlag(uint16_t major, uint16_t minor, bool is_partition_metrics)
    -> amd::smi::AMDGpuMetricVersionFlags_t {
  using Flag = amd::smi::AMDGpuMetricVersionFlags_t;
  if (is_partition_metrics) {
    if (major == 1) {
      if (minor == 0) {
        return Flag::kGpuXcpMetricV10;
      } else if (minor >= 1) {
        return Flag::kGpuXcpMetricDynV11Plus;
      } else {
        return Flag::kGpuMetricNone;
      }
    }
  } else {  // GPU metrics
    if (major == 1) {
      switch (minor) {
        case 0:
          return Flag::kGpuMetricV10;
        case 1:
          return Flag::kGpuMetricV11;
        case 2:
          return Flag::kGpuMetricV12;
        case 3:
          return Flag::kGpuMetricV13;
        case 4:
          return Flag::kGpuMetricV14;
        case 5:
          return Flag::kGpuMetricV15;
        case 6:
          return Flag::kGpuMetricV16;
        case 7:
          return Flag::kGpuMetricV17;
        case 8:
          return Flag::kGpuMetricV18;
        default:
          return Flag::kGpuMetricDynV19Plus;
      }
    }
  }
  return Flag::kGpuMetricNone;
}

// pass a header we want to test against
auto BuildFakeMetricsBlob(amd::smi::AMDGpuMetricsHeader_v1_t new_header) -> std::vector<uint8_t> {
  if (new_header.m_structure_size < sizeof(new_header)) {
    throw std::runtime_error("Header size too small");
  }
  amd::smi::AMDGpuMetricsHeader_v1_t header{};
  header.m_structure_size = static_cast<uint16_t>(sizeof(header));
  header.m_format_revision = new_header.m_format_revision;
  header.m_content_revision = new_header.m_content_revision;

  const uint8_t* begin = reinterpret_cast<const uint8_t*>(&header);
  return std::vector<uint8_t>(begin, begin + sizeof(header));
}

auto WriteBlobToTempFile(const std::vector<uint8_t>& blob,
                         const std::string& filename = "amdsmi_fake_metrics.bin")
    -> std::filesystem::path {
  auto temp_dir = std::filesystem::temp_directory_path();
  auto file_path = temp_dir / filename;

  std::ofstream stream(file_path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(blob.data()),
               static_cast<std::streamsize>(blob.size()));
  stream.close();

  return file_path;
}

}  // namespace

TEST(GpuUnit, GPUMetricDynamicVersionSupported) {
  PRINT_VERBOSITY();
  const bool is_partition_metrics = false;
  for (auto ver : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}) {
    std::string test_detail = "[GPUMetric";
    if (ver >= 9) {
      test_detail += "Dynamic] ";
    } else {
      test_detail += "Static] ";
    }
    std::cout << test_detail << "Checking version 1." << ver << std::endl;
    SCOPED_TRACE(testing::Message() << "Subtest for minor version: 1." << ver);
    const auto blob = BuildFakeMetricsBlob(amd::smi::AMDGpuMetricsHeader_v1_t{
        .m_structure_size = sizeof(amd::smi::AMDGpuMetricsHeader_v1_t),
        .m_format_revision = 1,
        .m_content_revision = static_cast<uint8_t>(ver),  // Known minor versions
    });
    const auto fake_path =
        WriteBlobToTempFile(blob, "amdsmi_fake_gpu_metrics_v1" + std::to_string(ver) + ".bin");

    ASSERT_FALSE(blob.empty());
    ASSERT_TRUE(std::filesystem::exists(fake_path));

    const auto* header = reinterpret_cast<const amd::smi::AMDGpuMetricsHeader_v1_t*>(blob.data());
    const auto flag = amd::smi::translate_header_to_flag_version(*header, is_partition_metrics,
                                                                 fake_path.string());
    EXPECT_EQ(flag,
              GetExpectedMetricVersionFlag(1, static_cast<uint16_t>(ver), is_partition_metrics))
        << "Version 1." << ver << " should be treated as supported";

    auto gpu_metrics_ptr =
        amd::smi::amdgpu_metrics_factory(flag, is_partition_metrics, fake_path.string());
    EXPECT_NE(gpu_metrics_ptr, nullptr)
        << "Factory must create metrics object for supported version";

    if (gpu_metrics_ptr) {
      std::cout << test_detail << "Created valid object for version 1." << ver << std::endl;
    } else {
      std::cout << test_detail << "Unsupported Metric Version"
                << " | Failed to create valid object for version 1." << ver << std::endl;
    }

    std::filesystem::remove(fake_path);
  }
}

TEST(GpuUnit, XCPMetricDynamicVersionSupported) {
  PRINT_VERBOSITY();
  const bool is_partition_metrics = true;
  for (auto ver : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}) {
    std::string test_detail = "[XCPMetric";
    if (ver >= 1) {
      test_detail += "Dynamic] ";
    } else {
      test_detail += "Static] ";
    }
    std::cout << test_detail << "Checking version 1." << ver << std::endl;
    SCOPED_TRACE(testing::Message() << "Subtest for minor version: 1." << ver);
    const auto blob = BuildFakeMetricsBlob(amd::smi::AMDGpuMetricsHeader_v1_t{
        .m_structure_size = sizeof(amd::smi::AMDGpuMetricsHeader_v1_t),
        .m_format_revision = 1,
        .m_content_revision = static_cast<uint8_t>(ver),  // Known minor versions
    });
    const auto fake_path =
        WriteBlobToTempFile(blob, "amdsmi_fake_xcp_metrics_v1" + std::to_string(ver) + ".bin");

    ASSERT_FALSE(blob.empty());
    ASSERT_TRUE(std::filesystem::exists(fake_path));

    const auto* header = reinterpret_cast<const amd::smi::AMDGpuMetricsHeader_v1_t*>(blob.data());
    const auto flag = amd::smi::translate_header_to_flag_version(*header, is_partition_metrics,
                                                                 fake_path.string());
    EXPECT_EQ(flag,
              GetExpectedMetricVersionFlag(1, static_cast<uint16_t>(ver), is_partition_metrics))
        << "Version 1." << ver << " should be treated as supported";

    auto xcp_metrics_ptr =
        amd::smi::amdgpu_metrics_factory(flag, is_partition_metrics, fake_path.string());
    EXPECT_NE(xcp_metrics_ptr, nullptr)
        << "Factory must create metrics object for supported version";

    if (xcp_metrics_ptr) {
      std::cout << test_detail << "Created valid object for version 1." << ver << std::endl;
    } else {
      std::cout << test_detail << "Failed to create valid object for version 1." << ver
                << std::endl;
    }

    std::filesystem::remove(fake_path);
  }
}

// gpu_metrics versions we can model must build an object; versions we cannot
// must classify as kGpuMetricNone (no object) so callers get NOT_SUPPORTED
// rather than corrupt data. APUs expose the v2.x family: v2.0-v2.3 are
// byte-prefix subsets read with the v2.4 layout, while v2.4 and v3.0 are exact
// matches -- all are supported. Guard that contract here.
TEST(GpuUnit, GPUMetricVersionSupportClassification) {
  PRINT_VERBOSITY();
  const bool is_partition_metrics = false;

  struct VersionCase {
    uint8_t format;
    uint8_t content;
    bool recognized;
  };
  const VersionCase cases[] = {
      {1, 4, true},                                               // GPU metrics positive control
      {2, 0, true},  {2, 1, true},  {2, 2, true},  {2, 3, true},  // APU v2.x prefix subsets
      {2, 4, true},  {3, 0, true},                                // APU exact matches
      {2, 5, false}, {4, 0, false}, {5, 0, false},                // beyond what we model
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(testing::Message() << "metrics version " << static_cast<int>(c.format) << "."
                                    << static_cast<int>(c.content));
    const auto blob = BuildFakeMetricsBlob(amd::smi::AMDGpuMetricsHeader_v1_t{
        .m_structure_size = sizeof(amd::smi::AMDGpuMetricsHeader_v1_t),
        .m_format_revision = c.format,
        .m_content_revision = c.content,
    });
    const auto fake_path =
        WriteBlobToTempFile(blob, "amdsmi_fake_metrics_v" + std::to_string(c.format) + "_" +
                                      std::to_string(c.content) + ".bin");
    ASSERT_TRUE(std::filesystem::exists(fake_path));

    const auto* header = reinterpret_cast<const amd::smi::AMDGpuMetricsHeader_v1_t*>(blob.data());
    const auto flag = amd::smi::translate_header_to_flag_version(*header, is_partition_metrics,
                                                                 fake_path.string());
    auto metrics_ptr =
        amd::smi::amdgpu_metrics_factory(flag, is_partition_metrics, fake_path.string());

    if (c.recognized) {
      EXPECT_NE(flag, amd::smi::AMDGpuMetricVersionFlags_t::kGpuMetricNone);
      EXPECT_NE(metrics_ptr, nullptr) << "recognized version must build a metrics object";
    } else {
      EXPECT_EQ(flag, amd::smi::AMDGpuMetricVersionFlags_t::kGpuMetricNone)
          << "unrecognized version must not map to a supported flag";
      EXPECT_EQ(metrics_ptr, nullptr) << "unrecognized version must not build a metrics object";
    }

    std::filesystem::remove(fake_path);
  }
}

namespace {

using ApuMetrics = amd::smi::AMDApuMetrics_v24_t;

// Declared table sizes of the APU revisions the v2.4 layout covers: each
// revision stops at a different field, so its declared size is the offset of
// the first field it does not define.
constexpr std::size_t kApuV20Size = offsetof(ApuMetrics, m_padding) + sizeof(uint16_t);
constexpr std::size_t kApuV21Size = offsetof(ApuMetrics, m_indep_throttle_status);
constexpr std::size_t kApuV22Size = offsetof(ApuMetrics, m_average_temperature_gfx);
constexpr std::size_t kApuV23Size = offsetof(ApuMetrics, m_average_cpu_voltage);
constexpr std::size_t kApuV24Size = offsetof(ApuMetrics, m_average_gfx_current) + sizeof(uint16_t);

// Any byte other than the 0xFF not-applicable sentinel works; this one makes
// "the device supplied this" obvious in a failure message.
constexpr uint8_t kFillerByte = 0xAB;
constexpr uint16_t kFilled16 = 0xABAB;
constexpr uint32_t kFilled32 = 0xABABABABU;
constexpr uint64_t kFilled64 = 0xABABABABABABABABULL;

// Fake sysfs tree at <tmp>/<name>/device/gpu_metrics. Binary reads are cached
// per (device path, read size), so each case needs its own directory.
class FakeMetricsDevice {
 public:
  explicit FakeMetricsDevice(const std::string& name)
      : root_(std::filesystem::temp_directory_path() / ("amdsmi_apu_metrics_" + name)) {
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_ / "device");
  }
  ~FakeMetricsDevice() { std::filesystem::remove_all(root_); }

  FakeMetricsDevice(const FakeMetricsDevice&) = delete;
  FakeMetricsDevice& operator=(const FakeMetricsDevice&) = delete;

  void WriteMetrics(const std::vector<uint8_t>& blob) const {
    std::ofstream stream(root_ / "device" / "gpu_metrics", std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(blob.data()),
                 static_cast<std::streamsize>(blob.size()));
  }

  std::string path() const { return root_.string(); }

 private:
  std::filesystem::path root_;
};

// A full v2.4-sized blob whose every metric byte carries the filler, with a
// header declaring only `declared_size` bytes. The reader is expected to honor
// the declared size, so bytes past it must never reach the metrics table.
auto BuildApuMetricsBlob(uint8_t content_revision, uint16_t declared_size) -> std::vector<uint8_t> {
  std::vector<uint8_t> blob(sizeof(ApuMetrics), kFillerByte);
  amd::smi::AMDGpuMetricsHeader_v1_t header{};
  header.m_structure_size = declared_size;
  header.m_format_revision = 2;
  header.m_content_revision = content_revision;
  std::memcpy(blob.data(), &header, sizeof(header));
  return blob;
}

}  // namespace

// A device may report an older, shorter APU revision than the newest layout we
// model, or declare a table longer than it. Reading must pre-fill with the
// not-applicable sentinel, take only the bytes the header declares, and stay
// inside our own buffer. Exercise that end to end against a fake sysfs node.
TEST(GpuUnit, APUMetricsShortRevisionReadsTrailingFieldsAsNotApplicable) {
  PRINT_VERBOSITY();

  struct ReadCase {
    const char* name;
    uint8_t content_revision;
    uint16_t declared_size;
    bool has_indep_throttle_status;  // added in v2.2
    bool has_average_temperature;    // added in v2.3
    bool has_average_voltage;        // added in v2.4
  };
  const ReadCase cases[] = {
      {"v2_0", 0, kApuV20Size, false, false, false},
      {"v2_1", 1, kApuV21Size, false, false, false},
      {"v2_2", 2, kApuV22Size, true, false, false},
      {"v2_3", 3, kApuV23Size, true, true, false},
      {"v2_4", 4, kApuV24Size, true, true, true},
      // Header claims more than the layout we model: the read must clamp to our
      // buffer rather than run past it.
      {"v2_4_oversized", 4, sizeof(ApuMetrics) + 4096, true, true, true},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(testing::Message() << "APU metrics revision " << test_case.name);
    FakeMetricsDevice fake_device(test_case.name);
    fake_device.WriteMetrics(
        BuildApuMetricsBlob(test_case.content_revision, test_case.declared_size));

    amd::smi::Device device(fake_device.path(), nullptr);
    ASSERT_EQ(device.setup_gpu_metrics_reading(), rsmi_status_t::RSMI_STATUS_SUCCESS);

    const auto [status_code, metrics] = device.dev_copy_internal_to_external_metrics();
    ASSERT_EQ(status_code, rsmi_status_t::RSMI_STATUS_SUCCESS);
    ASSERT_NE(metrics.apu_metrics, nullptr);
    const auto& apu = *metrics.apu_metrics;

    EXPECT_EQ(metrics.common_header.format_revision, 2);
    EXPECT_EQ(metrics.common_header.content_revision, test_case.content_revision);

    // Present in every v2.x revision.
    EXPECT_EQ(apu.temperature_gfx, kFilled16);
    EXPECT_EQ(apu.average_gfx_activity, kFilled16);
    EXPECT_EQ(apu.average_socket_power, kFilled16);
    EXPECT_EQ(apu.average_gfxclk_frequency, kFilled16);
    EXPECT_EQ(apu.throttle_status, kFilled32);
    EXPECT_EQ(apu.fan_pwm, kFilled16);

    // Added by later revisions: read back only when the header declares them.
    EXPECT_EQ(apu.indep_throttle_status,
              test_case.has_indep_throttle_status ? kFilled64 : UINT64_MAX);
    EXPECT_EQ(apu.average_temperature_gfx,
              test_case.has_average_temperature ? kFilled16 : UINT16_MAX);
    EXPECT_EQ(apu.average_cpu_voltage, test_case.has_average_voltage ? kFilled16 : UINT16_MAX);
  }
}
