/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <amd_smi_test/test_base.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "rocm_smi/rocm_smi_gpu_metrics.h"

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
        case 0: return Flag::kGpuMetricV10;
        case 1: return Flag::kGpuMetricV11;
        case 2: return Flag::kGpuMetricV12;
        case 3: return Flag::kGpuMetricV13;
        case 4: return Flag::kGpuMetricV14;
        case 5: return Flag::kGpuMetricV15;
        case 6: return Flag::kGpuMetricV16;
        case 7: return Flag::kGpuMetricV17;
        case 8: return Flag::kGpuMetricV18;
        default: return Flag::kGpuMetricDynV19Plus;
      }
    }
  }
  return Flag::kGpuMetricNone;
}

// pass a header we want to test against
auto BuildFakeMetricsBlob(amd::smi::AMDGpuMetricsHeader_v1_t new_header)
  -> std::vector<uint8_t> {
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

TEST(AmdSmiDynamicMetricTest, GPUMetricDynamicVersionSupported) {
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
    EXPECT_EQ(flag, GetExpectedMetricVersionFlag(1, static_cast<uint16_t>(ver), is_partition_metrics))
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

TEST(AmdSmiDynamicMetricTest, XCPMetricDynamicVersionSupported) {
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
    EXPECT_EQ(flag, GetExpectedMetricVersionFlag(1, static_cast<uint16_t>(ver), is_partition_metrics))
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
