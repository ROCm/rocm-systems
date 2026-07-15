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

#include "rocm_smi/rocm_smi_board_temp.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <vector>

#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi_dyn_gpu_metrics.h"
#include "rocm_smi/rocm_smi_logger.h"
#include "rocm_smi/rocm_smi_utils.h"

using amd::smi::getRSMIStatusString;

namespace amd::smi {

// Static mapping tables for temperature type conversions
static const std::map<int, rsmi_temperature_type_t> vr_temp_map = {
    {AMDGPU_VDDCR_VDD0_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_VDD0},
    {AMDGPU_VDDCR_VDD1_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_VDD1},
    {AMDGPU_VDDCR_VDD2_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_VDD2},
    {AMDGPU_VDDCR_VDD3_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_VDD3},
    {AMDGPU_VDDCR_SOC_A_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_SOC_A},
    {AMDGPU_VDDCR_SOC_C_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_SOC_C},
    {AMDGPU_VDDCR_SOCIO_A_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_SOCIO_A},
    {AMDGPU_VDDCR_SOCIO_C_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_SOCIO_C},
    {AMDGPU_VDD_085_HBM_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDD_085_HBM},
    {AMDGPU_VDDCR_11_HBM_B_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_11_HBM_B},
    {AMDGPU_VDDCR_11_HBM_D_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_11_HBM_D},
    {AMDGPU_VDD_USR_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDD_USR},
    {AMDGPU_VDDIO_11_E32_TEMP, RSMI_TEMP_TYPE_GPUBOARD_VDDIO_11_E32}};

static const std::map<int, rsmi_temperature_type_t> node_temp_map = {
    {AMDGPU_RETIMER_X_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_RETIMER_X},
    {AMDGPU_OAM_X_IBC_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_IBC},
    {AMDGPU_OAM_X_IBC_2_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_IBC_2},
    {AMDGPU_OAM_X_VDD18_VR_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_VDD18_VR},
    {AMDGPU_OAM_X_04_HBM_B_VR_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_B_VR},
    {AMDGPU_OAM_X_04_HBM_D_VR_TEMP, RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_D_VR}};

static const std::map<int, rsmi_temperature_type_t> system_temp_map = {
    {AMDGPU_UBB_FPGA_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_FPGA},
    {AMDGPU_UBB_FRONT_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_FRONT},
    {AMDGPU_UBB_BACK_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_BACK},
    {AMDGPU_UBB_OAM7_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_OAM7},
    {AMDGPU_UBB_IBC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_IBC},
    {AMDGPU_UBB_UFPGA_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_UFPGA},
    {AMDGPU_UBB_OAM1_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_OAM1},
    {AMDGPU_OAM_0_1_HSC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_0_1_HSC},
    {AMDGPU_OAM_2_3_HSC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_2_3_HSC},
    {AMDGPU_OAM_4_5_HSC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_4_5_HSC},
    {AMDGPU_OAM_6_7_HSC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_6_7_HSC},
    {AMDGPU_UBB_FPGA_0V72_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_FPGA_0V72_VR},
    {AMDGPU_UBB_FPGA_3V3_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_UBB_FPGA_3V3_VR},
    {AMDGPU_RETIMER_0_1_2_3_1V2_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_0_1_2_3_1V2_VR},
    {AMDGPU_RETIMER_4_5_6_7_1V2_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_4_5_6_7_1V2_VR},
    {AMDGPU_RETIMER_0_1_0V9_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_0_1_0V9_VR},
    {AMDGPU_RETIMER_4_5_0V9_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_4_5_0V9_VR},
    {AMDGPU_RETIMER_2_3_0V9_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_2_3_0V9_VR},
    {AMDGPU_RETIMER_6_7_0V9_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_RETIMER_6_7_0V9_VR},
    {AMDGPU_OAM_0_1_2_3_3V3_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_0_1_2_3_3V3_VR},
    {AMDGPU_OAM_4_5_6_7_3V3_VR_TEMP, RSMI_TEMP_TYPE_BASEBOARD_OAM_4_5_6_7_3V3_VR},
    {AMDGPU_IBC_HSC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_IBC_HSC},
    {AMDGPU_IBC_TEMP, RSMI_TEMP_TYPE_BASEBOARD_IBC}};

struct DynamicBoardTempMetrics {
  metrics_table_header_t common_header{};
  std::uint32_t attr_count{};
  std::map<details::AMDGpuMetricAttributeId_t, int64_t> metric_values{};
};

static const std::map<details::AMDGpuMetricAttributeId_t, rsmi_temperature_type_t>
    dynamic_gpuboard_temp_map = {
        {details::AMDGpuMetricAttributeId_t::NODE_TEMP_RETIMER,
         RSMI_TEMP_TYPE_GPUBOARD_NODE_RETIMER_X},
        {details::AMDGpuMetricAttributeId_t::NODE_TEMP_IBC, RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_IBC},
        {details::AMDGpuMetricAttributeId_t::NODE_TEMP_IBC_2,
         RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_IBC_2},
        {details::AMDGpuMetricAttributeId_t::NODE_TEMP_VDD18_VR,
         RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_VDD18_VR},
        {details::AMDGpuMetricAttributeId_t::NODE_TEMP_04_HBM_B_VR,
         RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_B_VR},
        {details::AMDGpuMetricAttributeId_t::NODE_TEMP_04_HBM_D_VR,
         RSMI_TEMP_TYPE_GPUBOARD_NODE_OAM_X_04_HBM_D_VR},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDCR_SOCIO_A,
         RSMI_TEMP_TYPE_GPUBOARD_VDDCR_SOCIO_A},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDCR_SOCIO_C,
         RSMI_TEMP_TYPE_GPUBOARD_VDDCR_SOCIO_C},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDCR_X0, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_VDD0},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDCR_X1, RSMI_TEMP_TYPE_GPUBOARD_VDDCR_VDD1},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDCR_HBM_B,
         RSMI_TEMP_TYPE_GPUBOARD_VDDCR_11_HBM_B},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDCR_HBM_D,
         RSMI_TEMP_TYPE_GPUBOARD_VDDCR_11_HBM_D},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDIO_04_HBM_B,
         RSMI_TEMP_TYPE_GPUBOARD_VDDIO_04_HBM_B},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDIO_04_HBM_D,
         RSMI_TEMP_TYPE_GPUBOARD_VDDIO_04_HBM_D},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDCR_075_HBM_B,
         RSMI_TEMP_TYPE_GPUBOARD_VDDCR_075_HBM_B},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDCR_075_HBM_D,
         RSMI_TEMP_TYPE_GPUBOARD_VDDCR_075_HBM_D},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDIO_11_GTA_A,
         RSMI_TEMP_TYPE_GPUBOARD_VDDIO_11_GTA_A},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDIO_11_GTA_C,
         RSMI_TEMP_TYPE_GPUBOARD_VDDIO_11_GTA_C},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDAN_075_GTA_A,
         RSMI_TEMP_TYPE_GPUBOARD_VDDAN_075_GTA_A},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDAN_075_GTA_C,
         RSMI_TEMP_TYPE_GPUBOARD_VDDAN_075_GTA_C},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDCR_075_UCIE,
         RSMI_TEMP_TYPE_GPUBOARD_VDDCR_075_UCIE},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDIO_065_UCIEAA,
         RSMI_TEMP_TYPE_GPUBOARD_VDDIO_065_UCIEAA},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDIO_065_UCIEAM_A,
         RSMI_TEMP_TYPE_GPUBOARD_VDDIO_065_UCIEAM_A},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDIO_065_UCIEAM_C,
         RSMI_TEMP_TYPE_GPUBOARD_VDDIO_065_UCIEAM_C},
        {details::AMDGpuMetricAttributeId_t::VR_TEMP_VDDAN_075, RSMI_TEMP_TYPE_GPUBOARD_VDDAN_075},
};

static const std::map<details::AMDGpuMetricAttributeId_t, rsmi_temperature_type_t>
    dynamic_baseboard_temp_map = {
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_UBB_FPGA,
         RSMI_TEMP_TYPE_BASEBOARD_UBB_FPGA},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_UBB_FRONT,
         RSMI_TEMP_TYPE_BASEBOARD_UBB_FRONT},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_UBB_BACK,
         RSMI_TEMP_TYPE_BASEBOARD_UBB_BACK},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_UBB_OAM7,
         RSMI_TEMP_TYPE_BASEBOARD_UBB_OAM7},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_UBB_IBC, RSMI_TEMP_TYPE_BASEBOARD_UBB_IBC},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_UBB_UFPGA,
         RSMI_TEMP_TYPE_BASEBOARD_UBB_UFPGA},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_UBB_OAM1,
         RSMI_TEMP_TYPE_BASEBOARD_UBB_OAM1},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_OAM_0_1_HSC,
         RSMI_TEMP_TYPE_BASEBOARD_OAM_0_1_HSC},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_OAM_2_3_HSC,
         RSMI_TEMP_TYPE_BASEBOARD_OAM_2_3_HSC},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_OAM_4_5_HSC,
         RSMI_TEMP_TYPE_BASEBOARD_OAM_4_5_HSC},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_OAM_6_7_HSC,
         RSMI_TEMP_TYPE_BASEBOARD_OAM_6_7_HSC},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_UBB_FPGA_0V72_VR,
         RSMI_TEMP_TYPE_BASEBOARD_UBB_FPGA_0V72_VR},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_UBB_FPGA_3V3_VR,
         RSMI_TEMP_TYPE_BASEBOARD_UBB_FPGA_3V3_VR},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_RETIMER_0_1_2_3_1V2_VR,
         RSMI_TEMP_TYPE_BASEBOARD_RETIMER_0_1_2_3_1V2_VR},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_RETIMER_4_5_6_7_1V2_VR,
         RSMI_TEMP_TYPE_BASEBOARD_RETIMER_4_5_6_7_1V2_VR},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_RETIMER_0_1_0V9_VR,
         RSMI_TEMP_TYPE_BASEBOARD_RETIMER_0_1_0V9_VR},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_RETIMER_4_5_0V9_VR,
         RSMI_TEMP_TYPE_BASEBOARD_RETIMER_4_5_0V9_VR},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_RETIMER_2_3_0V9_VR,
         RSMI_TEMP_TYPE_BASEBOARD_RETIMER_2_3_0V9_VR},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_RETIMER_6_7_0V9_VR,
         RSMI_TEMP_TYPE_BASEBOARD_RETIMER_6_7_0V9_VR},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_OAM_0_1_2_3_3V3_VR,
         RSMI_TEMP_TYPE_BASEBOARD_OAM_0_1_2_3_3V3_VR},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_OAM_4_5_6_7_3V3_VR,
         RSMI_TEMP_TYPE_BASEBOARD_OAM_4_5_6_7_3V3_VR},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_IBC_HSC, RSMI_TEMP_TYPE_BASEBOARD_IBC_HSC},
        {details::AMDGpuMetricAttributeId_t::SYSTEM_TEMP_IBC, RSMI_TEMP_TYPE_BASEBOARD_IBC},
};

static bool read_dynamic_scalar_value(const std::byte* data,
                                      details::AMDGpuMetricAttributeType_t attr_type,
                                      int64_t* value) {
  if (!data || !value) {
    return false;
  }

  switch (attr_type) {
    case details::AMDGpuMetricAttributeType_t::TYPE_UINT8: {
      auto v = std::uint8_t(0);
      std::memcpy(&v, data, sizeof(v));
      if (v == std::numeric_limits<uint8_t>::max()) return false;
      *value = static_cast<int64_t>(v);
      return true;
    }
    case details::AMDGpuMetricAttributeType_t::TYPE_INT8: {
      auto v = std::int8_t(0);
      std::memcpy(&v, data, sizeof(v));
      *value = static_cast<int64_t>(v);
      return true;
    }
    case details::AMDGpuMetricAttributeType_t::TYPE_UINT16: {
      auto v = std::uint16_t(0);
      std::memcpy(&v, data, sizeof(v));
      if (v == std::numeric_limits<uint16_t>::max()) return false;
      *value = static_cast<int64_t>(v);
      return true;
    }
    case details::AMDGpuMetricAttributeType_t::TYPE_INT16: {
      auto v = std::int16_t(0);
      std::memcpy(&v, data, sizeof(v));
      *value = static_cast<int64_t>(v);
      return true;
    }
    case details::AMDGpuMetricAttributeType_t::TYPE_UINT32: {
      auto v = std::uint32_t(0);
      std::memcpy(&v, data, sizeof(v));
      if (v == std::numeric_limits<uint32_t>::max()) return false;
      *value = static_cast<int64_t>(v);
      return true;
    }
    case details::AMDGpuMetricAttributeType_t::TYPE_INT32: {
      auto v = std::int32_t(0);
      std::memcpy(&v, data, sizeof(v));
      *value = static_cast<int64_t>(v);
      return true;
    }
    case details::AMDGpuMetricAttributeType_t::TYPE_UINT64: {
      auto v = std::uint64_t(0);
      std::memcpy(&v, data, sizeof(v));
      if (v == std::numeric_limits<uint64_t>::max()) return false;
      *value = static_cast<int64_t>(v);
      return true;
    }
    case details::AMDGpuMetricAttributeType_t::TYPE_INT64: {
      auto v = std::int64_t(0);
      std::memcpy(&v, data, sizeof(v));
      *value = v;
      return true;
    }
    default:
      return false;
  }
}

static int32_t decode_temperature_value(uint32_t encoded, uint8_t* sensor_id);

static int64_t dynamic_temp_to_millicelsius(int64_t raw_value,
                                            details::AMDGpuMetricAttributeType_t attr_type) {
  if (attr_type == details::AMDGpuMetricAttributeType_t::TYPE_UINT32) {
    const uint64_t uvalue = static_cast<uint64_t>(raw_value);
    if (uvalue > 0xFFFFFFULL && uvalue <= std::numeric_limits<uint32_t>::max()) {
      return static_cast<int64_t>(decode_temperature_value(static_cast<uint32_t>(uvalue), nullptr));
    }
  }

  return raw_value * 1000;
}

static rsmi_status_t read_board_temp_header(const char* filename, metrics_table_header_t& header) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    return ErrnoToRsmiStatus(errno);
  }
  file.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (file.gcount() < static_cast<std::streamsize>(sizeof(header))) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  return RSMI_STATUS_SUCCESS;
}

static rsmi_status_t read_dynamic_board_temp_metrics(const char* filename,
                                                     DynamicBoardTempMetrics& metrics) {
  if (!filename) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  metrics_table_header_t header{};
  if (const rsmi_status_t hdr_status = read_board_temp_header(filename, header);
      hdr_status != RSMI_STATUS_SUCCESS) {
    return hdr_status;
  }
  constexpr auto kMaxBoardTempBlobSize = std::size_t(1 << 20);  // 1 MB

  auto reserve_size =
      (static_cast<std::size_t>(header.structure_size) >= sizeof(metrics_table_header_t))
          ? static_cast<std::size_t>(header.structure_size)
          : sizeof(amdgpu_gpuboard_temp_metrics_v1_1);
  if (reserve_size > kMaxBoardTempBlobSize) {
    return RSMI_STATUS_UNEXPECTED_SIZE;
  }

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    return ErrnoToRsmiStatus(errno);
  }

  auto blob = std::vector<std::byte>(reserve_size);
  file.read(reinterpret_cast<char*>(blob.data()), static_cast<std::streamsize>(reserve_size));
  if (file.bad()) {
    return ErrnoToRsmiStatus(errno);
  }
  blob.resize(static_cast<size_t>(file.gcount()));
  if (blob.size() < sizeof(metrics_table_header_t) + sizeof(uint32_t)) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }

  auto offset = std::size_t(0);
  std::memcpy(&metrics.common_header, blob.data(), sizeof(metrics_table_header_t));
  offset += sizeof(metrics_table_header_t);

  if (metrics.common_header.content_revision < 1) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  }

  auto attr_count = std::uint32_t(0);
  std::memcpy(&attr_count, blob.data() + offset, sizeof(attr_count));
  offset += sizeof(attr_count);

  if (attr_count == 0) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  }

  metrics.attr_count = attr_count;
  metrics.metric_values.clear();

  for (uint32_t idx = 0; idx < attr_count; ++idx) {
    if (sizeof(uint64_t) > (blob.size() - offset)) {
      return RSMI_STATUS_UNEXPECTED_SIZE;
    }

    auto encoded_attr = std::uint64_t(0);
    std::memcpy(&encoded_attr, blob.data() + offset, sizeof(encoded_attr));
    offset += sizeof(encoded_attr);

    const auto decoded = details::amdgpu_metrics_decode_attr(encoded_attr);
    const auto attr_type = static_cast<details::AMDGpuMetricAttributeType_t>(decoded.m_attr_type);
    const auto attr_id = static_cast<details::AMDGpuMetricAttributeId_t>(decoded.m_attr_id);
    const auto attr_instance = decoded.m_attr_instance;

    const auto attr_type_size = details::get_metric_bytes(attr_type);
    if (attr_type_size == 0 || attr_instance == 0) {
      return RSMI_STATUS_UNEXPECTED_DATA;
    }

    const auto total_attr_size = (attr_instance * attr_type_size);
    if (total_attr_size > (blob.size() - offset)) {
      return RSMI_STATUS_UNEXPECTED_SIZE;
    }

    auto value = std::int64_t(0);
    if (read_dynamic_scalar_value(blob.data() + offset, attr_type, &value)) {
      // attr_id is the primary key (not a composed key).
      // duplicate attr_ids across attributes will overwrite each other.
      metrics.metric_values[attr_id] = dynamic_temp_to_millicelsius(value, attr_type);
    } else {
      std::ostringstream ss;
      ss << __PRETTY_FUNCTION__ << " | read_dynamic_scalar_value failed"
         << " | idx=" << idx << " | offset=0x" << std::hex << offset
         << " | attr_id=" << static_cast<uint32_t>(attr_id)
         << " | attr_type=" << static_cast<uint32_t>(attr_type) << " | attr_instance=" << std::dec
         << attr_instance << " | next_offset=0x" << std::hex
         << (offset + static_cast<std::size_t>(total_attr_size));
      LOG_DEBUG(ss);
    }

    offset += static_cast<size_t>(total_attr_size);
  }

  return RSMI_STATUS_SUCCESS;
}

// Helper function to create hex dump string
static std::string createHexDump(const void* data, size_t size, const std::string& description) {
  std::ostringstream ss;
  const unsigned char* bytes = static_cast<const unsigned char*>(data);

  ss << "=== " << description << " (size: " << size << " bytes) ===" << std::endl;

  for (size_t i = 0; i < size; i += 16) {
    // Print offset
    ss << std::hex << std::setfill('0') << std::setw(8) << i << ": ";

    // Print hex bytes
    for (size_t j = 0; j < 16; ++j) {
      if (i + j < size) {
        ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(bytes[i + j])
           << " ";
      } else {
        ss << "   ";
      }
    }

    ss << " | ";

    // Print ASCII representation
    for (size_t j = 0; j < 16 && i + j < size; ++j) {
      unsigned char c = bytes[i + j];
      ss << (std::isprint(c) ? static_cast<char>(c) : '.');
    }

    ss << std::endl;
  }

  ss << "=== End " << description << " ===" << std::endl;
  return ss.str();
}

rsmi_status_t read_gpuboard_temp_metrics(const char* filename,
                                         amdgpu_gpuboard_temp_metrics_v1_0& metrics) {
  if (!filename) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
       << " | Fail | filename is null | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
    LOG_INFO(ss);
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
     << " | filename: " << filename;
  LOG_INFO(ss);

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::ostringstream ess;
    ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Fail | Could not open file: " << filename << " | errno: " << errno << " ("
        << std::strerror(errno) << ")"
        << " | Returning = " << getRSMIStatusString(ErrnoToRsmiStatus(errno)) << " |";
    LOG_INFO(ess);
    return ErrnoToRsmiStatus(errno);
  }

  // Clear the metrics structure
  std::memset(&metrics, 0, sizeof(metrics));

  // Read the entire structure
  file.read(reinterpret_cast<char*>(&metrics), sizeof(metrics));
  if (file.bad()) {
    std::ostringstream ess;
    ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Fail | File read error | errno: " << errno << " (" << std::strerror(errno) << ")"
        << " | Returning = " << getRSMIStatusString(ErrnoToRsmiStatus(errno)) << " |";
    LOG_INFO(ess);
    return ErrnoToRsmiStatus(errno);
  }
  // Always create hex dump for debugging, using the number of bytes actually read
  std::string hexDump = createHexDump(&metrics, file.gcount(), "GPU Board Temperature Metrics");
  LOG_DEBUG(hexDump);

  if (file.gcount() != sizeof(metrics)) {
    std::ostringstream ess;
    ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Fail | Insufficient data read"
        << " | Expected: " << sizeof(metrics) << " bytes"
        << " | Actual: " << file.gcount() << " bytes"
        << " | Returning = " << getRSMIStatusString(RSMI_STATUS_INSUFFICIENT_SIZE) << " |";
    LOG_INFO(ess);
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  if (metrics.common_header.content_revision != 0) {
    std::ostringstream ess;
    ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Fail | Unexpected content revision for v1.0 parser: "
        << static_cast<unsigned>(metrics.common_header.content_revision)
        << " | Returning = " << getRSMIStatusString(RSMI_STATUS_UNEXPECTED_DATA) << " |";
    LOG_INFO(ess);
    return RSMI_STATUS_UNEXPECTED_DATA;
  }

  std::ostringstream oss;
  oss << __PRETTY_FUNCTION__ << " | ======= end ======= "
      << " | Success | File: " << filename << " | Bytes read: " << sizeof(metrics)
      << " | Header format: " << static_cast<unsigned>(metrics.common_header.format_revision)
      << " | Header content: " << static_cast<unsigned>(metrics.common_header.content_revision)
      << " | Node ID: " << metrics.node_id
      << " | Returning = " << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |";
  LOG_INFO(oss);

  return RSMI_STATUS_SUCCESS;
}

rsmi_status_t read_baseboard_temp_metrics(const char* filename,
                                          amdgpu_baseboard_temp_metrics_v1_0& metrics) {
  if (!filename) {
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
       << " | Fail | filename is null | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
    LOG_INFO(ss);
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
     << " | filename: " << filename;
  LOG_INFO(ss);

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::ostringstream ess;
    ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Fail | Could not open file: " << filename << " | errno: " << errno << " ("
        << std::strerror(errno) << ")"
        << " | Returning = " << getRSMIStatusString(ErrnoToRsmiStatus(errno)) << " |";
    LOG_INFO(ess);
    return ErrnoToRsmiStatus(errno);
  }

  // Clear the metrics structure
  std::memset(&metrics, 0, sizeof(metrics));

  // Read the entire structure
  file.read(reinterpret_cast<char*>(&metrics), sizeof(metrics));
  if (file.bad()) {
    std::ostringstream ess;
    ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Fail | File read error | errno: " << errno << " (" << std::strerror(errno) << ")"
        << " | Returning = " << getRSMIStatusString(ErrnoToRsmiStatus(errno)) << " |";
    LOG_INFO(ess);
    return ErrnoToRsmiStatus(errno);
  }
  // Always create hex dump for debugging, using the number of bytes actually read
  std::string hexDump = createHexDump(&metrics, file.gcount(), "Baseboard Temperature Metrics");
  LOG_DEBUG(hexDump);

  if (file.gcount() != sizeof(metrics)) {
    std::ostringstream ess;
    ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Fail | Insufficient data read"
        << " | Expected: " << sizeof(metrics) << " bytes"
        << " | Actual: " << file.gcount() << " bytes"
        << " | Returning = " << getRSMIStatusString(RSMI_STATUS_INSUFFICIENT_SIZE) << " |";
    LOG_INFO(ess);
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  if (metrics.common_header.content_revision != 0) {
    std::ostringstream ess;
    ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
        << " | Fail | Unexpected content revision for v1.0 parser: "
        << static_cast<unsigned>(metrics.common_header.content_revision)
        << " | Returning = " << getRSMIStatusString(RSMI_STATUS_UNEXPECTED_DATA) << " |";
    LOG_INFO(ess);
    return RSMI_STATUS_UNEXPECTED_DATA;
  }

  std::ostringstream oss;
  oss << __PRETTY_FUNCTION__ << " | ======= end ======= "
      << " | Success | File: " << filename << " | Bytes read: " << sizeof(metrics)
      << " | Header format: " << static_cast<unsigned>(metrics.common_header.format_revision)
      << " | Header content: " << static_cast<unsigned>(metrics.common_header.content_revision)
      << " | Node ID: " << metrics.node_id
      << " | Returning = " << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |";
  LOG_INFO(oss);

  return RSMI_STATUS_SUCCESS;
}

// Decode encoded temperature value: bits 24-31 = sensor id, bits 0-23 = signed temperature
// (Celsius)
static int32_t decode_temperature_value(uint32_t encoded, uint8_t* sensor_id = nullptr) {
  if (sensor_id) {
    *sensor_id = static_cast<uint8_t>((encoded >> 24) & 0xFF);
  }
  // Extract signed 24-bit temperature value
  int32_t temp = static_cast<int32_t>(encoded & 0xFFFFFF);
  // Sign-extend if negative
  if (temp & 0x800000) {
    temp |= ~0xFFFFFF;
  }

  temp *= 1000;  // Convert Celsius to milli-Celsius
  return temp;
}

rsmi_status_t get_gpuboard_temp_value(const amdgpu_gpuboard_temp_metrics_v1_0& metrics,
                                      rsmi_temperature_type_t temperature_type, int64_t* value) {
  if (!value) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
     << " | Node ID: " << metrics.node_id
     << " | Temperature type: " << static_cast<int>(temperature_type);
  LOG_INFO(ss);

  *value = 0;  // Initialize to 0
  const uint32_t INVALID_VALUE = std::numeric_limits<uint32_t>::max();

  // Check VR (Voltage Regulator) temperatures first
  for (int i = 0; i < AMDGPU_VR_MAX_TEMP_ENTRIES; ++i) {
    if (metrics.vr_temp[i] != INVALID_VALUE) {
      auto it = vr_temp_map.find(i);
      if (it != vr_temp_map.end() && it->second == temperature_type) {
        *value = decode_temperature_value(metrics.vr_temp[i]);
        std::ostringstream oss;
        oss << __PRETTY_FUNCTION__ << " | ======= end ======= "
            << " | Success | VR temp found at index: " << i << " | Raw value: " << *value
            << " | Returning = " << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |";
        LOG_INFO(oss);
        return RSMI_STATUS_SUCCESS;
      }
    }
  }

  // Check node temperatures if not found in VR
  for (int i = 0; i < AMDGPU_NODE_MAX_TEMP_ENTRIES; ++i) {
    if (metrics.node_temp[i] != INVALID_VALUE) {  // Max int indicates invalid temperature reading
      auto it = node_temp_map.find(i);
      if (it != node_temp_map.end() && it->second == temperature_type) {
        *value = decode_temperature_value(metrics.node_temp[i]);

        std::ostringstream oss;
        oss << __PRETTY_FUNCTION__ << " | ======= end ======= "
            << " | Success | Node temp found at index: " << i << " | Raw value: " << *value
            << " | Returning = " << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |";
        LOG_INFO(oss);
        return RSMI_STATUS_SUCCESS;
      }
    }
  }

  // Temperature type not found in metrics
  std::ostringstream ess;
  ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
      << " | Fail | Temperature type not found in GPU board metrics"
      << " | Temperature type: " << static_cast<int>(temperature_type)
      << " | Returning = " << getRSMIStatusString(RSMI_STATUS_NOT_SUPPORTED) << " |";
  LOG_ERROR(ess);
  return RSMI_STATUS_NOT_SUPPORTED;
}

rsmi_status_t get_baseboard_temp_value(const amdgpu_baseboard_temp_metrics_v1_0& metrics,
                                       rsmi_temperature_type_t temperature_type, int64_t* value) {
  if (!value) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start ======= "
     << " | Node ID: " << metrics.node_id
     << " | Temperature type: " << static_cast<int>(temperature_type);
  LOG_INFO(ss);

  *value = 0;  // Initialize to 0
  const uint32_t INVALID_VALUE = std::numeric_limits<uint32_t>::max();

  // Check system temperatures
  for (int i = 0; i < AMDGPU_SYSTEM_MAX_TEMP_ENTRIES; ++i) {
    if (metrics.system_temp[i] != INVALID_VALUE) {  // Max int indicates invalid temperature reading
      auto it = system_temp_map.find(i);
      if (it != system_temp_map.end() && it->second == temperature_type) {
        *value = decode_temperature_value(metrics.system_temp[i]);
        std::ostringstream oss;
        oss << __PRETTY_FUNCTION__ << " | ======= end ======= "
            << " | Success | System temp found at index: " << i << " | Raw value: " << *value
            << " | Returning = " << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |";
        LOG_INFO(oss);
        return RSMI_STATUS_SUCCESS;
      }
    }
  }

  // Temperature type not found in metrics
  std::ostringstream ess;
  ess << __PRETTY_FUNCTION__ << " | ======= end ======= "
      << " | Fail | Temperature type not found in baseboard metrics"
      << " | Temperature type: " << static_cast<int>(temperature_type)
      << " | Returning = " << getRSMIStatusString(RSMI_STATUS_NOT_SUPPORTED) << " |";
  LOG_ERROR(ess);
  return RSMI_STATUS_NOT_SUPPORTED;
}

rsmi_status_t get_gpuboard_temp_value_dynamic(const char* filename,
                                              rsmi_temperature_type_t temperature_type,
                                              int64_t* value) {
  if (!value) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DynamicBoardTempMetrics metrics{};
  rsmi_status_t status = read_dynamic_board_temp_metrics(filename, metrics);
  if (status != RSMI_STATUS_SUCCESS) {
    return status;
  }

  for (const auto& [attr_id, rsmi_type] : dynamic_gpuboard_temp_map) {
    if (rsmi_type != temperature_type) {
      continue;
    }

    auto metric_it = metrics.metric_values.find(attr_id);
    if (metric_it != metrics.metric_values.end()) {
      *value = metric_it->second;
      return RSMI_STATUS_SUCCESS;
    }
  }

  return RSMI_STATUS_NOT_SUPPORTED;
}

rsmi_status_t get_baseboard_temp_value_dynamic(const char* filename,
                                               rsmi_temperature_type_t temperature_type,
                                               int64_t* value) {
  if (!value) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DynamicBoardTempMetrics metrics{};
  rsmi_status_t status = read_dynamic_board_temp_metrics(filename, metrics);
  if (status != RSMI_STATUS_SUCCESS) {
    return status;
  }

  for (const auto& [attr_id, rsmi_type] : dynamic_baseboard_temp_map) {
    if (rsmi_type != temperature_type) {
      continue;
    }

    auto metric_it = metrics.metric_values.find(attr_id);
    if (metric_it != metrics.metric_values.end()) {
      *value = metric_it->second;
      return RSMI_STATUS_SUCCESS;
    }
  }

  return RSMI_STATUS_NOT_SUPPORTED;
}

}  // namespace amd::smi
