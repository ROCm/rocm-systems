// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "rocm_smi/rocm_smi_board_temp.h"
#include "rocm_smi/rocm_smi_dyn_gpu_metrics.h"
#include "test_fixture_utils.h"

namespace {

using AttrId = amd::smi::details::AMDGpuMetricAttributeId_t;
using AttrType = amd::smi::details::AMDGpuMetricAttributeType_t;

template <typename T>
constexpr AttrType AttrTypeOf() {
  static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
  constexpr bool is_signed = std::is_signed_v<T>;
  if constexpr (sizeof(T) == 1) {
    return is_signed ? AttrType::TYPE_INT8 : AttrType::TYPE_UINT8;
  } else if constexpr (sizeof(T) == 2) {
    return is_signed ? AttrType::TYPE_INT16 : AttrType::TYPE_UINT16;
  } else if constexpr (sizeof(T) == 4) {
    return is_signed ? AttrType::TYPE_INT32 : AttrType::TYPE_UINT32;
  } else {
    static_assert(sizeof(T) == 8);
    return is_signed ? AttrType::TYPE_INT64 : AttrType::TYPE_UINT64;
  }
}

// Writes a dynamic board temperature table with one T value per attribute, as the driver emits
// it, into tree and returns its path.
template <typename T>
std::string WriteBoardTempTable(FakeSysfsTree& tree,
                                const std::vector<std::pair<AttrId, T>>& attrs) {
  std::string blob;
  auto put = [&blob](const auto& v) { blob.append(reinterpret_cast<const char*>(&v), sizeof(v)); };
  put(amd::smi::details::AMDGpuDynamicMetricsHeader_v1_t{0, 1, 1});
  put(static_cast<uint32_t>(attrs.size()));
  for (const auto& [id, value] : attrs) {
    put(amd::smi::details::amdgpu_metrics_encode_attr(0, static_cast<uint64_t>(AttrTypeOf<T>()),
                                                      static_cast<uint64_t>(id), 1));
    put(value);
  }
  const auto size = static_cast<uint16_t>(blob.size());
  std::memcpy(blob.data(), &size, sizeof(size));
  tree.WriteFile("board_temp_metrics", blob);
  return tree.root() + "/board_temp_metrics";
}

// The driver marks a sensor the board does not have with 0xFF bytes: -1 for a signed type, the
// max value for an unsigned one. kReal must still read as a temperature. Unsigned types use 0:
// UINT32 values above 0xFFFFFF are decoded as encoded temperatures instead.
template <typename T>
struct Readings {
  static constexpr T kMissing = std::is_signed_v<T> ? T(-1) : std::numeric_limits<T>::max();
  static constexpr T kReal = std::is_signed_v<T> ? T(-2) : T(0);
  static constexpr int64_t kRealMillidegrees = static_cast<int64_t>(kReal) * 1000;
};

template <typename T>
::testing::Message WidthName() {
  return ::testing::Message() << (std::is_signed_v<T> ? "int" : "uint") << sizeof(T) * 8
                              << "_t values";
}

template <typename T>
void ExpectBaseboardTemps() {
  SCOPED_TRACE(WidthName<T>());
  FakeSysfsTree tree;
  const auto path =
      WriteBoardTempTable<T>(tree, {{AttrId::SYSTEM_TEMP_UBB_FPGA, T{55}},
                                    {AttrId::SYSTEM_TEMP_UBB_FRONT, Readings<T>::kReal},
                                    {AttrId::SYSTEM_TEMP_UBB_OAM1, Readings<T>::kMissing}});

  int64_t value = 0;
  EXPECT_EQ(amd::smi::get_baseboard_temp_value_dynamic(path.c_str(),
                                                       RSMI_TEMP_TYPE_BASEBOARD_UBB_FPGA, &value),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(value, 55000);
  EXPECT_EQ(amd::smi::get_baseboard_temp_value_dynamic(path.c_str(),
                                                       RSMI_TEMP_TYPE_BASEBOARD_UBB_FRONT, &value),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(value, Readings<T>::kRealMillidegrees);
  EXPECT_EQ(amd::smi::get_baseboard_temp_value_dynamic(path.c_str(),
                                                       RSMI_TEMP_TYPE_BASEBOARD_UBB_OAM1, &value),
            RSMI_STATUS_NOT_SUPPORTED);
}

template <typename T>
void ExpectGpuboardTemps() {
  SCOPED_TRACE(WidthName<T>());
  FakeSysfsTree tree;
  const auto path = WriteBoardTempTable<T>(tree, {{AttrId::NODE_TEMP_RETIMER, T{28}},
                                                  {AttrId::NODE_TEMP_IBC_2, Readings<T>::kReal},
                                                  {AttrId::NODE_TEMP_IBC, Readings<T>::kMissing}});

  int64_t value = 0;
  EXPECT_EQ(amd::smi::get_gpuboard_temp_value_dynamic(
                path.c_str(), RSMI_TEMP_TYPE_GPUBOARD_NODE_RETIMER_X, &value),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(value, 28000);
  EXPECT_EQ(amd::smi::get_gpuboard_temp_value_dynamic(
                path.c_str(), RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_IBC_2, &value),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(value, Readings<T>::kRealMillidegrees);
  EXPECT_EQ(amd::smi::get_gpuboard_temp_value_dynamic(
                path.c_str(), RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_IBC, &value),
            RSMI_STATUS_NOT_SUPPORTED);
}

TEST(GpuUnit, DynamicBaseboardTempUnavailableSensorIsNotSupported) {
  ExpectBaseboardTemps<int8_t>();
  ExpectBaseboardTemps<int16_t>();
  ExpectBaseboardTemps<int32_t>();
  ExpectBaseboardTemps<int64_t>();
  ExpectBaseboardTemps<uint8_t>();
  ExpectBaseboardTemps<uint16_t>();
  ExpectBaseboardTemps<uint32_t>();
  ExpectBaseboardTemps<uint64_t>();
}

TEST(GpuUnit, DynamicGpuboardTempUnavailableSensorIsNotSupported) {
  ExpectGpuboardTemps<int8_t>();
  ExpectGpuboardTemps<int16_t>();
  ExpectGpuboardTemps<int32_t>();
  ExpectGpuboardTemps<int64_t>();
  ExpectGpuboardTemps<uint8_t>();
  ExpectGpuboardTemps<uint16_t>();
  ExpectGpuboardTemps<uint32_t>();
  ExpectGpuboardTemps<uint64_t>();
}

}  // namespace
