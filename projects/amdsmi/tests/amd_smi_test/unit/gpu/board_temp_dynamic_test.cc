// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "rocm_smi/rocm_smi_board_temp.h"
#include "rocm_smi/rocm_smi_dyn_gpu_metrics.h"

namespace {

using AttrId = amd::smi::details::AMDGpuMetricAttributeId_t;
using AttrType = amd::smi::details::AMDGpuMetricAttributeType_t;

// Writes a dynamic board temperature table with one signed 16-bit value per attribute, as the
// driver emits it, and returns its path.
std::filesystem::path WriteBoardTempTable(const std::vector<std::pair<AttrId, int16_t>>& attrs,
                                          const std::string& filename) {
  std::vector<std::byte> blob;
  auto put = [&blob](const auto& v) {
    const auto* p = reinterpret_cast<const std::byte*>(&v);
    blob.insert(blob.end(), p, p + sizeof(v));
  };
  put(amd::smi::details::AMDGpuDynamicMetricsHeader_v1_t{0, 1, 1});
  put(static_cast<uint32_t>(attrs.size()));
  for (const auto& [id, value] : attrs) {
    put(amd::smi::details::amdgpu_metrics_encode_attr(
        0, static_cast<uint64_t>(AttrType::TYPE_INT16), static_cast<uint64_t>(id), 1));
    put(value);
  }
  const auto size = static_cast<uint16_t>(blob.size());
  std::memcpy(blob.data(), &size, sizeof(size));

  const auto path = std::filesystem::temp_directory_path() / filename;
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(blob.data()),
               static_cast<std::streamsize>(blob.size()));
  return path;
}

TEST(GpuUnit, DynamicBaseboardTempUnavailableSensorIsNotSupported) {
  const auto path = WriteBoardTempTable({{AttrId::SYSTEM_TEMP_UBB_FPGA, 55},
                                         {AttrId::SYSTEM_TEMP_UBB_FRONT, -5},
                                         {AttrId::SYSTEM_TEMP_UBB_OAM1, int16_t(-1)}},
                                        "amdsmi_fake_baseboard_temp.bin");

  int64_t value = 0;
  EXPECT_EQ(amd::smi::get_baseboard_temp_value_dynamic(path.c_str(),
                                                       RSMI_TEMP_TYPE_BASEBOARD_UBB_FPGA, &value),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(value, 55000);
  EXPECT_EQ(amd::smi::get_baseboard_temp_value_dynamic(path.c_str(),
                                                       RSMI_TEMP_TYPE_BASEBOARD_UBB_FRONT, &value),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(value, -5000);
  EXPECT_EQ(amd::smi::get_baseboard_temp_value_dynamic(path.c_str(),
                                                       RSMI_TEMP_TYPE_BASEBOARD_UBB_OAM1, &value),
            RSMI_STATUS_NOT_SUPPORTED);

  std::filesystem::remove(path);
}

TEST(GpuUnit, DynamicGpuboardTempUnavailableSensorIsNotSupported) {
  const auto path =
      WriteBoardTempTable({{AttrId::NODE_TEMP_RETIMER, 28}, {AttrId::NODE_TEMP_IBC, int16_t(-1)}},
                          "amdsmi_fake_gpuboard_temp.bin");

  int64_t value = 0;
  EXPECT_EQ(amd::smi::get_gpuboard_temp_value_dynamic(
                path.c_str(), RSMI_TEMP_TYPE_GPUBOARD_NODE_RETIMER_X, &value),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(value, 28000);
  EXPECT_EQ(amd::smi::get_gpuboard_temp_value_dynamic(
                path.c_str(), RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_IBC, &value),
            RSMI_STATUS_NOT_SUPPORTED);

  std::filesystem::remove(path);
}

}  // namespace
