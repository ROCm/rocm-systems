/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2023, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include "rocm_smi/rocm_smi_gpu_metrics.h"
#include "rocm_smi/rocm_smi_common.h"  // Should go before rocm_smi.h
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_logger.h"

#include <dirent.h>
#include <pthread.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>  // NOLINT
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace amd::smi;

#define TRY try {
#define CATCH } catch (...) {return amd::smi::handleException();}

// Put definitions of old gpu_metrics formats here
typedef struct {
  struct metrics_table_header_t common_header;

  /* Driver attached timestamp (in ns) */
  uint64_t      system_clock_counter;

/* Temperature */
  uint16_t      temperature_edge;
  uint16_t      temperature_hotspot;
  uint16_t      temperature_mem;
  uint16_t      temperature_vrgfx;
  uint16_t      temperature_vrsoc;
  uint16_t      temperature_vrmem;

/* Utilization */
  uint16_t      average_gfx_activity;
  uint16_t      average_umc_activity;  // memory controller
  uint16_t      average_mm_activity;   // UVD or VCN

/* Power/Energy */
  uint16_t      average_socket_power;
  uint32_t      energy_accumulator;

/* Average clocks */
  uint16_t      average_gfxclk_frequency;
  uint16_t      average_socclk_frequency;
  uint16_t      average_uclk_frequency;
  uint16_t      average_vclk0_frequency;
  uint16_t      average_dclk0_frequency;
  uint16_t      average_vclk1_frequency;
  uint16_t      average_dclk1_frequency;

/* Current clocks */
  uint16_t      current_gfxclk;
  uint16_t      current_socclk;
  uint16_t      current_uclk;
  uint16_t      current_vclk0;
  uint16_t      current_dclk0;
  uint16_t      current_vclk1;
  uint16_t      current_dclk1;

/* Throttle status */
  uint32_t      throttle_status;

/* Fans */
  uint16_t      current_fan_speed;

/* Link width/speed */
  uint8_t       pcie_link_width;
  uint8_t       pcie_link_speed;  // in 0.1 GT/s
} rsmi_gpu_metrics_v_1_0_t;

typedef struct {
  rsmi_gpu_metrics_t base;
  uint64_t           firmware_timestamp;
} rsmi_gpu_metrics_v_1_2;

typedef struct {
  rsmi_gpu_metrics_t base;
  /* PMFW attached timestamp (10ns resolution) */
  uint64_t			firmware_timestamp;

  /* Voltage (mV) */
  uint16_t			voltage_soc;
  uint16_t			voltage_gfx;
  uint16_t			voltage_mem;

  uint16_t          padding1;

  /* Throttle status (ASIC independent) */
  uint64_t			indep_throttle_status;

} rsmi_gpu_metrics_v_1_3;


// log current gpu_metrics file content read
// any metrics value can be a nullptr
void log_gpu_metrics(const metrics_table_header_t *gpu_metrics_table_header,
                     const rsmi_gpu_metrics_v_1_2 *rsmi_gpu_metrics_v_1_2,
                     const rsmi_gpu_metrics_v_1_3 *gpu_metrics_v_1_3,
                     const rsmi_gpu_metrics_t *rsmi_gpu_metrics) {
  if (!RocmSMI::getInstance().isLoggingOn()) {
    return;
  }
  std::ostringstream ss;
  if (gpu_metrics_table_header != nullptr) {
    ss
        /* Common Header */
        << print_unsigned_hex_and_int(
            gpu_metrics_table_header->structure_size,
            "gpu_metrics_table_header->structure_size")
        << print_unsigned_hex_and_int(
            gpu_metrics_table_header->format_revision,
            "gpu_metrics_table_header->format_revision")
        << print_unsigned_hex_and_int(
            gpu_metrics_table_header->content_revision,
            "gpu_metrics_table_header->content_revision");
    LOG_DEBUG(ss);
  }
  if (rsmi_gpu_metrics == nullptr) {
    return;
  }

  ss
    /* Common Header */
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->common_header.structure_size,
        "rsmi_gpu_metrics->common_header.structure_size")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->common_header.format_revision,
        "rsmi_gpu_metrics->common_header.format_revision")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->common_header.content_revision,
        "rsmi_gpu_metrics->common_header.content_revision")
    /* Temperature */
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->temperature_edge,
        "rsmi_gpu_metrics->temperature_edge")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->temperature_hotspot,
        "rsmi_gpu_metrics->temperature_hotspot")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->temperature_mem,
        "rsmi_gpu_metrics->temperature_mem")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->temperature_vrgfx,
        "rsmi_gpu_metrics->temperature_vrgfx")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->temperature_vrsoc,
        "rsmi_gpu_metrics->temperature_vrsoc")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->temperature_vrmem,
        "rsmi_gpu_metrics->temperature_vrmem")
    /* Utilization */
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->average_gfx_activity,
        "rsmi_gpu_metrics->average_gfx_activity")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->average_umc_activity,
        "rsmi_gpu_metrics->average_umc_activity")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->average_mm_activity,
        "rsmi_gpu_metrics->average_mm_activity")
    /* Power/Energy */
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->average_socket_power,
        "rsmi_gpu_metrics->average_socket_power")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->energy_accumulator,
        "rsmi_gpu_metrics->energy_accumulator")
    /* Driver attached timestamp (in ns) */
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->system_clock_counter,
        "rsmi_gpu_metrics->system_clock_counter")
    /* Average clocks */
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->average_gfxclk_frequency,
        "rsmi_gpu_metrics->average_gfxclk_frequency")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->average_socclk_frequency,
        "rsmi_gpu_metrics->average_socclk_frequency")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->average_uclk_frequency,
        "rsmi_gpu_metrics->average_uclk_frequency")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->average_vclk0_frequency,
        "rsmi_gpu_metrics->average_vclk0_frequency")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->average_dclk0_frequency,
        "rsmi_gpu_metrics->average_dclk0_frequency")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->average_vclk1_frequency,
        "rsmi_gpu_metrics->average_vclk1_frequency")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->average_dclk1_frequency,
        "rsmi_gpu_metrics->average_dclk1_frequency")
    /* Current clocks */
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->current_gfxclk,
        "rsmi_gpu_metrics->current_gfxclk")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->current_socclk,
        "rsmi_gpu_metrics->current_socclk")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->current_uclk,
        "rsmi_gpu_metrics->current_uclk")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->current_vclk0,
        "rsmi_gpu_metrics->current_vclk0")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->current_dclk0,
        "rsmi_gpu_metrics->current_dclk0")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->current_vclk1,
        "rsmi_gpu_metrics->current_vclk1")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->current_dclk1,
        "rsmi_gpu_metrics->current_dclk1")
    /* Throttle status */
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->throttle_status,
        "rsmi_gpu_metrics->throttle_status")
    /* Fans */
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->current_fan_speed,
        "rsmi_gpu_metrics->current_fan_speed")
    /* Link width/speed */
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->pcie_link_width,
        "rsmi_gpu_metrics->pcie_link_width")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->pcie_link_speed,
        "rsmi_gpu_metrics->pcie_link_speed")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->padding,
        "rsmi_gpu_metrics->padding")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->gfx_activity_acc,
        "rsmi_gpu_metrics->gfx_activity_acc")
    << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->mem_activity_acc,
        "rsmi_gpu_metrics->mem_activity_acc");
    for (int i=0; i < RSMI_NUM_HBM_INSTANCES; i++) {
      ss << print_unsigned_hex_and_int(
        rsmi_gpu_metrics->temperature_hbm[i],
        "rsmi_gpu_metrics->temperature_hbm[" + std::to_string(i) + "]");
    }

    if (rsmi_gpu_metrics_v_1_2 != nullptr) {
      /* PMFW attached timestamp (10ns resolution) */
      ss
      << print_unsigned_hex_and_int(
          rsmi_gpu_metrics_v_1_2->firmware_timestamp,
          "rsmi_gpu_metrics_v_1_2->firmware_timestamp");
    }

    if (gpu_metrics_v_1_3 != nullptr) {
      /* PMFW attached timestamp (10ns resolution) */
      ss
      << print_unsigned_hex_and_int(
          gpu_metrics_v_1_3->firmware_timestamp,
          "gpu_metrics_v_1_3->firmware_timestamp")
      /* Voltage (mV) */
      << print_unsigned_hex_and_int(
          gpu_metrics_v_1_3->voltage_soc,
          "gpu_metrics_v_1_3->voltage_soc")
      << print_unsigned_hex_and_int(
          gpu_metrics_v_1_3->voltage_gfx,
          "gpu_metrics_v_1_3->voltage_gfx")
      << print_unsigned_hex_and_int(
          gpu_metrics_v_1_3->voltage_mem,
          "gpu_metrics_v_1_3->voltage_mem")
      << print_unsigned_hex_and_int(
          gpu_metrics_v_1_3->padding1,
          "gpu_metrics_v_1_3->padding1")
      /* Throttle status (ASIC independent) */
      << print_unsigned_hex_and_int(
          gpu_metrics_v_1_3->indep_throttle_status,
          "gpu_metrics_v_1_3->indep_throttle_status");
    }
    LOG_DEBUG(ss);
}

static rsmi_status_t GetGPUMetricsFormat1(uint32_t dv_ind,
                                rsmi_gpu_metrics_t *data, uint8_t content_v) {
  assert(content_v != RSMI_GPU_METRICS_API_CONTENT_VER_1 &&
         content_v != RSMI_GPU_METRICS_API_CONTENT_VER_2 &&
         content_v != RSMI_GPU_METRICS_API_CONTENT_VER_3 );
  if (content_v == RSMI_GPU_METRICS_API_CONTENT_VER_1 ||
      content_v == RSMI_GPU_METRICS_API_CONTENT_VER_2 ||
      content_v == RSMI_GPU_METRICS_API_CONTENT_VER_3 ) {
    // This function shouldn't be called if content version is
    // RSMI_GPU_METRICS_API_CONTENT_VER_1 or RSMI_GPU_METRICS_API_CONTENT_VER_2
    // or RSMI_GPU_METRICS_API_CONTENT_VER_3
    return RSMI_STATUS_INVALID_ARGS;
  }
  void *metric_data = nullptr;
  size_t data_size;
  rsmi_status_t ret;

  rsmi_gpu_metrics_v_1_0_t metric_data_v_1_0;

  if (content_v == 0) {
    metric_data = &metric_data_v_1_0;
    data_size = sizeof(rsmi_gpu_metrics_v_1_0_t);
  }  // else { ...   handle other conversions to v1

  assert(metric_data != nullptr && "Unexpected conversion attempted.");
  ret = GetDevBinaryBlob(amd::smi::kDevGpuMetrics, dv_ind, data_size,
                                                                 metric_data);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

#define ASSIGN_DATA_FIELD(FIELD, SRC) \
  data->FIELD = (SRC)->FIELD;

#define ASSIGN_COMMON_FORMATS(SRC) \
    ASSIGN_DATA_FIELD(common_header, (SRC)) \
    ASSIGN_DATA_FIELD(temperature_edge, (SRC)) \
    ASSIGN_DATA_FIELD(temperature_hotspot, (SRC)) \
    ASSIGN_DATA_FIELD(temperature_mem, (SRC)) \
    ASSIGN_DATA_FIELD(temperature_vrgfx, (SRC)) \
    ASSIGN_DATA_FIELD(temperature_vrsoc, (SRC)) \
    ASSIGN_DATA_FIELD(temperature_vrmem, (SRC)) \
    ASSIGN_DATA_FIELD(average_gfx_activity, (SRC)) \
    ASSIGN_DATA_FIELD(average_umc_activity, (SRC)) \
    ASSIGN_DATA_FIELD(average_mm_activity, (SRC)) \
    ASSIGN_DATA_FIELD(average_socket_power, (SRC)) \
    ASSIGN_DATA_FIELD(system_clock_counter, (SRC)) \
    ASSIGN_DATA_FIELD(average_gfxclk_frequency, (SRC)) \
    ASSIGN_DATA_FIELD(average_socclk_frequency, (SRC)) \
    ASSIGN_DATA_FIELD(average_uclk_frequency, (SRC)) \
    ASSIGN_DATA_FIELD(average_vclk0_frequency, (SRC)) \
    ASSIGN_DATA_FIELD(average_dclk0_frequency, (SRC)) \
    ASSIGN_DATA_FIELD(average_vclk1_frequency, (SRC)) \
    ASSIGN_DATA_FIELD(average_dclk1_frequency, (SRC)) \
    ASSIGN_DATA_FIELD(current_gfxclk, (SRC)) \
    ASSIGN_DATA_FIELD(current_socclk, (SRC)) \
    ASSIGN_DATA_FIELD(current_uclk, (SRC)) \
    ASSIGN_DATA_FIELD(current_vclk0, (SRC)) \
    ASSIGN_DATA_FIELD(current_dclk0, (SRC)) \
    ASSIGN_DATA_FIELD(current_vclk1, (SRC)) \
    ASSIGN_DATA_FIELD(current_dclk1, (SRC)) \
    ASSIGN_DATA_FIELD(throttle_status, (SRC)) \
    ASSIGN_DATA_FIELD(current_fan_speed, (SRC))

  // Now handle differences from format 1
  if (content_v == 0) {
    // First handle all data that is common to Format1 and other formats
    ASSIGN_COMMON_FORMATS(
                    reinterpret_cast<rsmi_gpu_metrics_v_1_0_t *>(metric_data))

    // Then, the differences:
    data->energy_accumulator = static_cast<uint64_t>(
        reinterpret_cast<rsmi_gpu_metrics_v_1_0_t *>(
                                            metric_data)->energy_accumulator);
    data->pcie_link_width = static_cast<uint16_t>(
        reinterpret_cast<rsmi_gpu_metrics_v_1_0_t *>(
                                               metric_data)->pcie_link_width);
    data->pcie_link_speed = static_cast<uint16_t>(
        reinterpret_cast<rsmi_gpu_metrics_v_1_0_t *>(
                                               metric_data)->pcie_link_speed);

    // These fields didn't exist in v0
    data->gfx_activity_acc = 0;
    data->mem_activity_acc  = 0;
    (void)memset(data->temperature_hbm, 0,
                                   RSMI_NUM_HBM_INSTANCES * sizeof(uint16_t));
  }  // else handle other conversions to format 1
#undef ASSIGN_DATA_FIELD
#undef ASSIGN_COMMON_FORMATS
  return RSMI_STATUS_SUCCESS;
}

// Translate gpu_metrics version 1.2 to rsmi_gpu_metrics_t. gpu_metrics
// version 1.2 provides timestamp provided by the firmware. This timestamp
// is sampled atomically along when gpu_metric information. Use this
// timestamp instead of system_clock_counter

static void map_gpu_metrics_1_2_to_rsmi_gpu_metrics_t(
        const rsmi_gpu_metrics_v_1_2 *gpu_metrics_v_1_2,
        rsmi_gpu_metrics_t *rsmi_gpu_metrics)
{
    memcpy(rsmi_gpu_metrics, &gpu_metrics_v_1_2->base,
           sizeof(rsmi_gpu_metrics_t));
    // firmware_timestamp is at 10ns resolution
    rsmi_gpu_metrics->system_clock_counter =
          gpu_metrics_v_1_2->firmware_timestamp * 10;
}

static void map_gpu_metrics_1_3_to_rsmi_gpu_metrics_t(
        const rsmi_gpu_metrics_v_1_3 *gpu_metrics_v_1_3,
        rsmi_gpu_metrics_t *rsmi_gpu_metrics)
{
    memcpy(rsmi_gpu_metrics, &gpu_metrics_v_1_3->base,
           sizeof(rsmi_gpu_metrics_t));
    // firmware_timestamp is at 10ns resolution
    rsmi_gpu_metrics->system_clock_counter =
          gpu_metrics_v_1_3->firmware_timestamp * 10;

}

rsmi_status_t
rsmi_dev_gpu_metrics_info_get(uint32_t dv_ind, rsmi_gpu_metrics_t *smu) {
  TRY
  DEVICE_MUTEX
  CHK_SUPPORT_NAME_ONLY(smu)
  rsmi_gpu_metrics_v_1_2 smu_v_1_2;
  rsmi_gpu_metrics_v_1_3 smu_v_1_3;
  rsmi_status_t ret;

  std::ostringstream ss;
  if (!dev->gpu_metrics_ver().structure_size) {
    ret = GetDevBinaryBlob(amd::smi::kDevGpuMetrics, dv_ind,
              sizeof(struct metrics_table_header_t), &dev->gpu_metrics_ver());
    log_gpu_metrics(&dev->gpu_metrics_ver(), nullptr, nullptr, nullptr);

    if (ret != RSMI_STATUS_SUCCESS) {
      ss << "Returning = " << getRSMIStatusString(ret)
         << ",\ndev->gpu_metrics_ver().structure_size = "
         << print_unsigned_int(dev->gpu_metrics_ver().structure_size)
         << ", could not read common header";
      LOG_ERROR(ss);
      return ret;
    }
  }
  // only supports gpu_metrics_v1_x version
  if (dev->gpu_metrics_ver().format_revision != 1) {
    ss << "Returning = " << getRSMIStatusString(RSMI_STATUS_NOT_SUPPORTED)
         << ",\ndev->gpu_metrics_ver().format_revision = "
         << print_unsigned_int(dev->gpu_metrics_ver().format_revision)
         << " was not equal to 1";
    LOG_ERROR(ss);
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  // Initialize the smu fields to zero as some of them only valid in
  // a specific version.
  *smu = {};

  if (dev->gpu_metrics_ver().content_revision ==
                                        RSMI_GPU_METRICS_API_CONTENT_VER_1) {
        ret = GetDevBinaryBlob(amd::smi::kDevGpuMetrics, dv_ind,
                                           sizeof(rsmi_gpu_metrics_t), smu);
        ss << __PRETTY_FUNCTION__ << " | RSMI_GPU_METRICS_API_CONTENT_VER_1";
        LOG_DEBUG(ss);
        log_gpu_metrics(nullptr, nullptr, nullptr, smu);
  } else if (dev->gpu_metrics_ver().content_revision ==
                                        RSMI_GPU_METRICS_API_CONTENT_VER_2) {
        ret = GetDevBinaryBlob(amd::smi::kDevGpuMetrics, dv_ind,
                                sizeof(rsmi_gpu_metrics_v_1_2), &smu_v_1_2);
        map_gpu_metrics_1_2_to_rsmi_gpu_metrics_t(&smu_v_1_2, smu);
        ss << __PRETTY_FUNCTION__ << " | RSMI_GPU_METRICS_API_CONTENT_VER_2";
        LOG_DEBUG(ss);
        log_gpu_metrics(nullptr, &smu_v_1_2, nullptr, smu);
  } else if (dev->gpu_metrics_ver().content_revision ==
                                        RSMI_GPU_METRICS_API_CONTENT_VER_3) {
        ret = GetDevBinaryBlob(amd::smi::kDevGpuMetrics, dv_ind,
                                sizeof(rsmi_gpu_metrics_v_1_3), &smu_v_1_3);
        map_gpu_metrics_1_3_to_rsmi_gpu_metrics_t(&smu_v_1_3, smu);
        ss << __PRETTY_FUNCTION__ << " | RSMI_GPU_METRICS_API_CONTENT_VER_3";
        LOG_DEBUG(ss);
        log_gpu_metrics(nullptr, nullptr, &smu_v_1_3, smu);
  } else {
            ret = GetGPUMetricsFormat1(dv_ind, smu,
                                     dev->gpu_metrics_ver().content_revision);
            ss << __PRETTY_FUNCTION__ << " | GetGPUMetricsFormat1";
            LOG_DEBUG(ss);
            log_gpu_metrics(nullptr, nullptr, nullptr, smu);
  }

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  return ret;
  CATCH
}


namespace amd::smi
{

constexpr uint16_t join_metrics_version(uint8_t format_rev, uint8_t content_rev)
{
  return (format_rev << 8 | content_rev);
}

constexpr uint16_t join_metrics_version(const AMDGpuMetricsHeader_v1_t& metrics_header)
{
  return join_metrics_version(metrics_header.m_format_revision, metrics_header.m_content_revision);
}

AMDGpuMetricsHeader_v1_t disjoin_metrics_version(uint16_t version)
{
  AMDGpuMetricsHeader_v1_t metrics_header;

  metrics_header.m_format_revision  = static_cast<uint8_t>((version & 0xFF00) >> 8);
  metrics_header.m_content_revision = static_cast<uint8_t>(version & 0x00FF);

  return metrics_header;
}

uint64_t actual_timestamp_in_secs()
{
  using namespace std::chrono;
  return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

std::string stringfy_metrics_header(const AMDGpuMetricsHeader_v1_t& metrics_header)
{
  std::stringstream metrics_header_info;
  metrics_header_info
    << "Format: " << print_unsigned_hex_and_int(metrics_header.m_format_revision)
    << "." << print_unsigned_hex_and_int(metrics_header.m_content_revision)
    << " Size: " << print_unsigned_hex_and_int(metrics_header.m_structure_size);

  return metrics_header_info.str();
}

//
//  version 1,0: 256
//  version 1,1: 257
//  version 1,2: 258
//  version 1,3: 259
//  version 1,4: 260
//  version 1,5: 261
//
const AMDGpuMetricVersionTranslationTbl_t amdgpu_metric_version_translation_table
{
  {join_metrics_version(1, 1), AMDGpuMetricVersionFlags_t::kGpuMetricV11},
  {join_metrics_version(1, 2), AMDGpuMetricVersionFlags_t::kGpuMetricV12},
  {join_metrics_version(1, 3), AMDGpuMetricVersionFlags_t::kGpuMetricV13},
  {join_metrics_version(1, 4), AMDGpuMetricVersionFlags_t::kGpuMetricV14},
};

/**
 *
*/
const AMDGpuMetricsClassIdTranslationTbl_t amdgpu_metrics_class_id_translation_table
{
  {AMDGpuMetricsClassId_t::kGpuMetricHeader, "Header"},
  {AMDGpuMetricsClassId_t::kGpuMetricTemperature, "Temperature"},
  {AMDGpuMetricsClassId_t::kGpuMetricUtilization, "Utilization"},
  {AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy, "Power/Energy"},
  {AMDGpuMetricsClassId_t::kGpuMetricSystemClockCounter, "System Clock"},
  {AMDGpuMetricsClassId_t::kGpuMetricAverageClock, "Average Clock"},
  {AMDGpuMetricsClassId_t::kGpuMetricCurrentClock, "Current Clock"},
  {AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus, "Throttle"},
  {AMDGpuMetricsClassId_t::kGpuMetricGfxClkLockStatus, "Gfx Clock Lock"},
  {AMDGpuMetricsClassId_t::kGpuMetricCurrentFanSpeed, "Current Fan Speed"},
  {AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed, "Link/Bandwidth/Speed"},
  {AMDGpuMetricsClassId_t::kGpuMetricVoltage, "Voltage"},
  {AMDGpuMetricsClassId_t::kGpuMetricTimestamp, "Timestamp"},
};

const AMDGpuMetricsUnitTypeTranslationTbl_t amdgpu_metrics_unit_type_translation_table
{
  // kGpuMetricTemperature counters
  {AMDGpuMetricsUnitType_t::kMetricTempEdge, "TempEdge"},
  {AMDGpuMetricsUnitType_t::kMetricTempHotspot, "TempHotspot"},
  {AMDGpuMetricsUnitType_t::kMetricTempMem, "TempMem"},
  {AMDGpuMetricsUnitType_t::kMetricTempVrGfx, "TempVrGfx"},
  {AMDGpuMetricsUnitType_t::kMetricTempVrSoc, "TempVrSoc"},
  {AMDGpuMetricsUnitType_t::kMetricTempVrMem, "TempVrMem"},
  {AMDGpuMetricsUnitType_t::kMetricTempHbm, "TempHbm"},

  // kGpuMetricUtilization counters
  {AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity, "AvgGfxActivity"},
  {AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity, "AvgUmcActivity"},
  {AMDGpuMetricsUnitType_t::kMetricAvgMmActivity, "AvgMmActivity"},
  {AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator, "GfxActivityAcc"},
  {AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator, "MemActivityAcc"},
  {AMDGpuMetricsUnitType_t::kMetricVcnActivity, "VcnActivity"},

  // kGpuMetricAverageClock counters
  {AMDGpuMetricsUnitType_t::kMetricAvgGfxClockFrequency, "AvgGfxClockFrequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgSocClockFrequency, "AvgSocClockFrequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgUClockFrequency, "AvgUClockFrequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgVClock0Frequency, "AvgVClock0Frequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgDClock0Frequency, "AvgDClock0Frequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgVClock1Frequency, "AvgVClock1Frequency"},
  {AMDGpuMetricsUnitType_t::kMetricAvgDClock1Frequency, "AvgDClock1Frequency"},

  // kGpuMetricCurrentClock counters
  {AMDGpuMetricsUnitType_t::kMetricCurrGfxClock, "CurrGfxClock"},
  {AMDGpuMetricsUnitType_t::kMetricCurrSocClock, "CurrSocClock"},
  {AMDGpuMetricsUnitType_t::kMetricCurrUClock, "CurrUClock"},
  {AMDGpuMetricsUnitType_t::kMetricCurrVClock0, "CurrVClock0"},
  {AMDGpuMetricsUnitType_t::kMetricCurrDClock0, "CurrDClock0"},
  {AMDGpuMetricsUnitType_t::kMetricCurrVClock1, "CurrVClock1"},
  {AMDGpuMetricsUnitType_t::kMetricCurrDClock1, "CurrDClock1"},

  // kGpuMetricThrottleStatus counters
  {AMDGpuMetricsUnitType_t::kMetricThrottleStatus, "ThrottleStatus"},
  {AMDGpuMetricsUnitType_t::kMetricIndepThrottleStatus, "IndepThrottleStatus"},

  // kGpuMetricGfxClkLockStatus counters
  {AMDGpuMetricsUnitType_t::kMetricGfxClkLockStatus, "GfxClkLockStatus"},

  // kGpuMetricCurrentFanSpeed counters
  {AMDGpuMetricsUnitType_t::kMetricCurrFanSpeed, "CurrFanSpeed"},

  // kGpuMetricLinkWidthSpeed counters
  {AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth, "PcieLinkWidth"},
  {AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed, "PcieLinkSpeed"},
  {AMDGpuMetricsUnitType_t::kMetricPcieBandwidthAccumulator, "PcieBandwidthAcc"},
  {AMDGpuMetricsUnitType_t::kMetricPcieBandwidthInst, "PcieBandwidthInst"},
  {AMDGpuMetricsUnitType_t::kMetricXgmiLinkWidth, "XgmiLinkWidth"},
  {AMDGpuMetricsUnitType_t::kMetricXgmiLinkSpeed, "XgmiLinkSpeed"},
  {AMDGpuMetricsUnitType_t::kMetricXgmiReadDataAccumulator, "XgmiReadDataAcc"},
  {AMDGpuMetricsUnitType_t::kMetricXgmiWriteDataAccumulator, "XgmiWriteDataAcc"},

  // kGpuMetricPowerEnergy counters
  {AMDGpuMetricsUnitType_t::kMetricAvgSocketPower, "AvgSocketPower"},
  {AMDGpuMetricsUnitType_t::kMetricCurrSocketPower, "CurrSocketPower"},
  {AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator, "EnergyAcc"},

  // kGpuMetricVoltage counters
  {AMDGpuMetricsUnitType_t::kMetricVoltageSoc, "VoltageSoc"},
  {AMDGpuMetricsUnitType_t::kMetricVoltageGfx, "VoltageGfx"},
  {AMDGpuMetricsUnitType_t::kMetricVoltageMem, "VoltageMem"},

  // kGpuMetricTimestamp counters
  {AMDGpuMetricsUnitType_t::kMetricTSClockCounter, "TSClockCounter"},
  {AMDGpuMetricsUnitType_t::kMetricTSFirmware, "TSFirmware"},
};


AMDGpuMetricVersionFlags_t translate_header_to_flag_version(const AMDGpuMetricsHeader_v1_t& metrics_header)
{
  const auto flag_version = join_metrics_version(metrics_header);
  if (amdgpu_metric_version_translation_table.find(flag_version) != amdgpu_metric_version_translation_table.end()) {
    return amdgpu_metric_version_translation_table.at(flag_version);
  }

  return AMDGpuMetricVersionFlags_t::kGpuMetricNone;
}

rsmi_status_t is_gpu_metrics_version_supported(const AMDGpuMetricsHeader_v1_t& metrics_header)
{
  const auto flag_version = join_metrics_version(metrics_header);
  return (amdgpu_metric_version_translation_table.find(flag_version) !=
          amdgpu_metric_version_translation_table.end())
         ? rsmi_status_t::RSMI_STATUS_SUCCESS : rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED;
}

struct AMDGpuMetricsLogInfo_t
{
  rsmi_status_t m_status_code;
  std::string m_title;
  std::string m_pretty_function;
};

class AMDGpuMetricsLogger_t
{
  public:
    enum class LogInfoType_t
    {
      kLogError,
      kLogAlarm,
      kLogInfo,
      kLogBuffer,
      kLogTrace,
      kLogDebug,
    };

    void operator()(const AMDGpuMetricsLogInfo_t& log_info)
    {
      m_ostrstream << log_info.m_pretty_function << log_info.m_title;
      LOG_TRACE(m_ostrstream);

      m_ostrstream << log_info.m_pretty_function
                   << " | ======= end ======= "
                   << " | Fail "
                   << " | Device #: "
                   << " | Metric Version: "
                   << " | Cause: Couldn't get a valid metric object"
                   << " | Returning = "
                   << getRSMIStatusString(log_info.m_status_code)
                   << " |";

      LOG_ERROR(m_ostrstream);
    }

  private:
    std::ostringstream m_ostrstream;

};


AMDGpuMetricFactories_t amd_gpu_metrics_factory_table
{
  {AMDGpuMetricVersionFlags_t::kGpuMetricV11, std::make_unique<GpuMetricsBase_v11_t>(GpuMetricsBase_v11_t{})},
  {AMDGpuMetricVersionFlags_t::kGpuMetricV12, std::make_unique<GpuMetricsBase_v12_t>(GpuMetricsBase_v12_t{})},
  {AMDGpuMetricVersionFlags_t::kGpuMetricV13, std::make_unique<GpuMetricsBase_v13_t>(GpuMetricsBase_v13_t{})},
  {AMDGpuMetricVersionFlags_t::kGpuMetricV14, std::make_unique<GpuMetricsBase_v14_t>(GpuMetricsBase_v14_t{})},
};

GpuMetricsBasePtr amdgpu_metrics_factory(AMDGpuMetricVersionFlags_t gpu_metric_version)
{
  auto contains = [](const AMDGpuMetricVersionFlags_t metric_version) {
    return (amd_gpu_metrics_factory_table.find(metric_version) != amd_gpu_metrics_factory_table.end());
  };

  if (contains(gpu_metric_version)) {
    return std::move(amd_gpu_metrics_factory_table[gpu_metric_version]);
  }

  return nullptr;
}

template<typename T>
AMDGpuDynamicMetricTblValues_t format_metric_row(const T& metric, const std::string& value_title)
{
  auto multi_values = AMDGpuDynamicMetricTblValues_t{};

  auto get_data_type_info = [&]() {
    auto data_type(AMDGpuMetricsDataType_t::kUInt64);
    if constexpr (std::is_array_v<T>) {
      const uint8_t  check_uint8[]={1};
      const uint16_t check_uint16[]={2};
      const uint32_t check_uint32[]={3};
      const uint64_t check_uint64[]={4};
      if constexpr (std::is_same_v<decltype(metric[0]), decltype(check_uint8[0])>) {
        data_type = AMDGpuMetricsDataType_t::kUInt8;
      }
      if constexpr (std::is_same_v<decltype(metric[0]), decltype(check_uint16[0])>) {
        data_type = AMDGpuMetricsDataType_t::kUInt16;
      }
      if constexpr (std::is_same_v<decltype(metric[0]), decltype(check_uint32[0])>) {
        data_type = AMDGpuMetricsDataType_t::kUInt32;
      }
      if constexpr (std::is_same_v<decltype(metric[0]), decltype(check_uint64[0])>) {
        data_type = AMDGpuMetricsDataType_t::kUInt64;
      }
      return std::make_tuple(data_type, static_cast<uint16_t>(std::end(metric) - std::begin(metric)));
    }

    const uint16_t kSingleValue(1);
    if constexpr (std::is_same_v<T, std::uint8_t>) {
      data_type = AMDGpuMetricsDataType_t::kUInt8;
    }
    if constexpr (std::is_same_v<T, std::uint16_t>) {
      data_type = AMDGpuMetricsDataType_t::kUInt16;
    }
    if constexpr (std::is_same_v<T, std::uint32_t>) {
      data_type = AMDGpuMetricsDataType_t::kUInt32;
    }
    if constexpr (std::is_same_v<T, std::uint64_t>) {
      data_type = AMDGpuMetricsDataType_t::kUInt64;
    }
    return std::make_tuple(data_type, kSingleValue);
  };

  const auto [data_type, num_values] = get_data_type_info();
  for (auto idx = uint16_t(0); idx < num_values; ++idx) {
    auto value = uint64_t(0);
    if constexpr (std::is_array_v<T>) {
      value = (metric[idx]);
    }
    else {
      value = (metric);
    }

    auto amdgpu_dynamic_metric_value = [&]() {
      AMDGpuDynamicMetricsValue_t amdgpu_dynamic_metric_value_init{};
      amdgpu_dynamic_metric_value_init.m_value = value;
      amdgpu_dynamic_metric_value_init.m_info  = (value_title + " : " + std::to_string(idx));
      amdgpu_dynamic_metric_value_init.m_original_type = data_type;
      return amdgpu_dynamic_metric_value_init;
    }();

    multi_values.emplace_back(amdgpu_dynamic_metric_value);
  }

  return multi_values;
}

rsmi_status_t GpuMetricsBase_v14_t::populate_metrics_dynamic_tbl()
{
  // Temperature Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHotspot,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hotspot,
                                "temperature_hotspot"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHotspot,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hotspot,
                               "temperature_hotspot"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_mem,
                                "temperature_mem"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrsoc,
                                "temperature_vrsoc"))
           );

  // Power/Energy Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocketPower,
              format_metric_row(m_gpu_metrics_tbl.m_curr_socket_power,
                                "curr_socket_power"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_energy_accumulator,
                                "energy_acc"))
           );

  // Utilization Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfx_activity,
                                "average_gfx_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_umc_activity,
                                "average_umc_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVcnActivity,
              format_metric_row(m_gpu_metrics_tbl.m_vcn_activity,
                                "[average_vcn_activity]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_gfx_activity_acc,
              "gfx_activity_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_mem_activity_acc,
                                "mem_activity_acc"))
           );

  // Timestamp Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSFirmware,
              format_metric_row(m_gpu_metrics_tbl.m_firmware_timestamp,
                                "firmware_timestamp"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
              format_metric_row(m_gpu_metrics_tbl.m_system_clock_counter,
                                "system_clock_counter"))
           );

  // Throttle Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricThrottleStatus,
              format_metric_row(m_gpu_metrics_tbl.m_throttle_status,
                                "throttle_status"))
           );

  // GfxLock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricGfxClkLockStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxClkLockStatus,
              format_metric_row(m_gpu_metrics_tbl.m_gfxclk_lock_status,
                                "gfxclk_lock_status"))
           );

  // Link/Width/Speed Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_width,
                                "pcie_link_width"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_speed,
                                "pcie_link_speed"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_link_width,
                                "xgmi_link_width"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_link_speed,
                                "xgmi_link_speed"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieBandwidthAccumulator,\
              format_metric_row(m_gpu_metrics_tbl.m_pcie_bandwidth_acc,
                                "pcie_bandwidth_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieBandwidthInst,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_bandwidth_acc,
                                "pcie_bandwidth_inst"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiReadDataAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_read_data_acc,
                                "[xgmi_read_data_acc]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricXgmiWriteDataAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_xgmi_write_data_acc,
                                "[xgmi_write_data_acc]"))
           );

  // CurrentClock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_gfxclk,
                                "[current_gfxclk]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_socclk,
                                "[current_socclk]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk0,
                                "[current_vclk0]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk0,
                                "[current_dclk0]"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrUClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_uclk,
                                "current_uclk"))
           );

  return rsmi_status_t::RSMI_STATUS_SUCCESS;
}

rsmi_status_t GpuMetricsBase_v13_t::populate_metrics_dynamic_tbl()
{
  // Temperature Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempEdge,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_edge,
                                "temperature_edge"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHotspot,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hotspot,
                                "temperature_hotspot"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_mem,
                                "temperature_mem"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrGfx,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrgfx,
                                "temperature_vrgfx"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrSoc,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrsoc,
                                "temperature_vrsoc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempVrMem,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_vrmem,
                                "temperature_vrmem"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTemperature]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTempHbm,
              format_metric_row(m_gpu_metrics_tbl.m_temperature_hbm,
                                "[temperature_hbm]"))
           );

  // Power/Energy Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgSocketPower,
              format_metric_row(m_gpu_metrics_tbl.m_average_socket_power,
                                "average_socket_power"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricPowerEnergy]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricEnergyAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_energy_accumulator,
                                "energy_acc"))
           );

  // Utilization Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfx_activity,
                                "average_gfx_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUmcActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_umc_activity,
                                "average_umc_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgMmActivity,
              format_metric_row(m_gpu_metrics_tbl.m_average_mm_activity,
                                "average_mm_activity"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricGfxActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_gfx_activity_acc,
                                "gfx_activity_acc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricUtilization]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricMemActivityAccumulator,
              format_metric_row(m_gpu_metrics_tbl.m_mem_activity_acc,
                                "mem_activity_acc"))
           );

  // Timestamp Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSFirmware,
              format_metric_row(m_gpu_metrics_tbl.m_firmware_timestamp,
                                "firmware_timestamp"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricTimestamp]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricTSClockCounter,
              format_metric_row(m_gpu_metrics_tbl.m_system_clock_counter,
                                "system_clock_counter"))
           );

  // Fan Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentFanSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrFanSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_current_fan_speed,
                                "current_fan_speed"))
           );

  // Throttle Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricThrottleStatus,
              format_metric_row(m_gpu_metrics_tbl.m_throttle_status,
                                "throttle_status"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricThrottleStatus]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricIndepThrottleStatus,
              format_metric_row(m_gpu_metrics_tbl.m_indep_throttle_status,
                                "indep_throttle_status"))
           );

  // Average Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgGfxClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_gfxclk_frequency,
                                "average_gfxclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgSocClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_socclk_frequency,
                                "average_socclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgUClockFrequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_uclk_frequency,
                                "average_uclk_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgVClock0Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_vclk0_frequency,
                                "average_vclk0_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgDClock0Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_dclk0_frequency,
                                "average_dclk0_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgVClock1Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_vclk1_frequency,
                                "average_vclk1_frequency"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricAverageClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricAvgDClock1Frequency,
              format_metric_row(m_gpu_metrics_tbl.m_average_dclk1_frequency,
                                "average_dclk1_frequency"))
           );

  // CurrentClock Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrGfxClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_gfxclk,
                                "current_gfxclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrSocClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_socclk,
                                "current_socclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrUClock,
              format_metric_row(m_gpu_metrics_tbl.m_current_uclk,
                                "current_uclk"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk0,
                                "current_vclk0"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock0,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk0,
                                "current_dclk0"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrVClock1,
              format_metric_row(m_gpu_metrics_tbl.m_current_vclk1,
                                "current_vclk1"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricCurrentClock]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricCurrDClock1,
              format_metric_row(m_gpu_metrics_tbl.m_current_dclk1,
                                "current_dclk1"))
           );

  // Link/Width/Speed Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkWidth,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_width,
                                "pcie_link_width"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricLinkWidthSpeed]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricPcieLinkSpeed,
              format_metric_row(m_gpu_metrics_tbl.m_pcie_link_speed,
                                "pcie_link_speed"))
           );

  // Voltage Info
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricVoltage]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVoltageSoc,
              format_metric_row(m_gpu_metrics_tbl.m_voltage_soc,
                                "voltage_soc"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricVoltage]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVoltageGfx,
              format_metric_row(m_gpu_metrics_tbl.m_voltage_gfx,
                                "voltage_gfx"))
           );
  m_metrics_dynamic_tbl[AMDGpuMetricsClassId_t::kGpuMetricVoltage]
    .insert(std::make_pair(AMDGpuMetricsUnitType_t::kMetricVoltageMem,
              format_metric_row(m_gpu_metrics_tbl.m_voltage_mem,
                                "voltage_mem"))
           );

  return rsmi_status_t::RSMI_STATUS_SUCCESS;
}

rsmi_status_t GpuMetricsBase_v12_t::populate_metrics_dynamic_tbl()
{
  // TODO: Implement these;
  return rsmi_status_t::RSMI_STATUS_NOT_YET_IMPLEMENTED;
}

rsmi_status_t GpuMetricsBase_v11_t::populate_metrics_dynamic_tbl()
{
  // TODO: Implement these;
  return rsmi_status_t::RSMI_STATUS_NOT_YET_IMPLEMENTED;
}


rsmi_status_t Device::dev_read_gpu_metrics_header_data()
{
  std::ostringstream ostrstream;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ostrstream << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ostrstream);

  // Check if/when metrics table needs to be refreshed.
  auto now_ts = actual_timestamp_in_secs();
  if (((!m_gpu_metrics_header.m_structure_size)    ||
       (!m_gpu_metrics_header.m_format_revision)   ||
       (!m_gpu_metrics_header.m_content_revision)) ||
      ((now_ts - m_gpu_metrics_updated_timestamp) >=
        kRSMI_GPU_METRICS_EXPIRATION_SECS)) {
    auto op_result = readDevInfo(DevInfoTypes::kDevGpuMetrics,
                                 sizeof(AMDGpuMetricsHeader_v1_t),
                                 &m_gpu_metrics_header);
    if ((status_code = ErrnoToRsmiStatus(op_result)) !=
        rsmi_status_t::RSMI_STATUS_SUCCESS) {
      ostrstream << __PRETTY_FUNCTION__
                 << " | ======= end ======= "
                 << " | Fail "
                 << " | Device #: " << index()
                 << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header)
                 << " | Cause: readDevInfo(kDevGpuMetrics)"
                 << " | Returning = "
                 << getRSMIStatusString(status_code)
                 << " Could not read Metrics Header: "
                 << print_unsigned_int(m_gpu_metrics_header.m_structure_size)
                 << " |";
      LOG_ERROR(ostrstream);
      return status_code;
    }
    if ((status_code = is_gpu_metrics_version_supported(m_gpu_metrics_header)) ==
        rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED) {
      ostrstream << __PRETTY_FUNCTION__
                 << " | ======= end ======= "
                 << " | Fail "
                 << " | Device #: " << index()
                 << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header)
                 << " | Cause: gpu metric file version is not supported: "
                 << " | Returning = "
                 << getRSMIStatusString(status_code)
                 << " Could not read Metrics Header: "
                 << print_unsigned_int(m_gpu_metrics_header.m_structure_size)
                 << " |";
      LOG_ERROR(ostrstream);
      return status_code;
    }

    m_gpu_metrics_updated_timestamp = actual_timestamp_in_secs();
  }

  ostrstream << __PRETTY_FUNCTION__
             << " | ======= end ======= "
             << " | Success "
             << " | Device #: " << index()
             << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header)
             << " | Update Timestamp: " << m_gpu_metrics_updated_timestamp
             << " | Returning = "
             << getRSMIStatusString(status_code)
             << " |";
  LOG_TRACE(ostrstream);
  return status_code;
}

rsmi_status_t Device::dev_read_gpu_metrics_all_data()
{
  std::ostringstream ostrstream;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ostrstream << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ostrstream);

  // At this point we should have a valid gpu_metrics pointer.
  if (!m_gpu_metrics_ptr) {
    status_code = rsmi_status_t::RSMI_STATUS_UNEXPECTED_DATA;
    ostrstream << __PRETTY_FUNCTION__
                << " | ======= end ======= "
                << " | Fail "
                << " | Device #: " << index()
                << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header())
                << " | Cause: Couldn't get a valid metric object"
                << " | Returning = "
                << getRSMIStatusString(status_code)
                << " |";
    LOG_ERROR(ostrstream);
    return status_code;
  }

  auto op_result = readDevInfo(DevInfoTypes::kDevGpuMetrics,
                               m_gpu_metrics_ptr->sizeof_metric_table(),
                               &m_gpu_metrics_ptr->get_metrics_table());
  if ((status_code = ErrnoToRsmiStatus(op_result)) !=
      rsmi_status_t::RSMI_STATUS_SUCCESS) {
    ostrstream << __PRETTY_FUNCTION__
                << " | ======= end ======= "
                << " | Fail "
                << " | Device #: " << index()
                << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header)
                << " | Cause: readDevInfo(kDevGpuMetrics)"
                << " | Returning = "
                << getRSMIStatusString(status_code)
                << " Could not read Metrics Header: "
                << print_unsigned_int(m_gpu_metrics_header.m_structure_size)
                << " |";
    LOG_ERROR(ostrstream);
    return status_code;
  }

  m_gpu_metrics_updated_timestamp = actual_timestamp_in_secs();
  ostrstream << __PRETTY_FUNCTION__
             << " | ======= end ======= "
             << " | Success "
             << " | Device #: " << index()
             << " | Metric Version: " << stringfy_metrics_header(m_gpu_metrics_header)
             << " | Update Timestamp: " << m_gpu_metrics_updated_timestamp
             << " | Returning = "
             << getRSMIStatusString(status_code)
             << " |";
  LOG_TRACE(ostrstream);
  return status_code;
}

rsmi_status_t Device::setup_gpu_metrics_reading()
{
  std::ostringstream ostrstream;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ostrstream << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ostrstream);

  status_code = dev_read_gpu_metrics_header_data();
  if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    return status_code;
  }

  const auto gpu_metrics_flag_version = translate_header_to_flag_version(dev_get_metrics_header());
  if (gpu_metrics_flag_version == AMDGpuMetricVersionFlags_t::kGpuMetricNone) {
    status_code = rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED;
    ostrstream << __PRETTY_FUNCTION__
                << " | ======= end ======= "
                << " | Fail "
                << " | Device #: " << index()
                << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header())
                << " | [Translates to: " << join_metrics_version(dev_get_metrics_header())
                << " ] "
                << " | Cause: Metric version found is not supported!"
                << " | Returning = "
                << getRSMIStatusString(status_code)
                << " |";
    LOG_ERROR(ostrstream);
    return status_code;
  }

  auto gpu_metrics_ptr = amdgpu_metrics_factory(gpu_metrics_flag_version);
  if (!gpu_metrics_ptr) {
    status_code = rsmi_status_t::RSMI_STATUS_UNEXPECTED_DATA;
    ostrstream << __PRETTY_FUNCTION__
                << " | ======= end ======= "
                << " | Fail "
                << " | Device #: " << index()
                << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header())
                << " | Cause: amdgpu_metrics_factory() couldn't get a valid metric object"
                << " | Returning = "
                << getRSMIStatusString(status_code)
                << " |";
    LOG_ERROR(ostrstream);
    return status_code;
  }

  /// gpu_metrics_ptr has the pointer to the proper object type/version.
  dev_set_gpu_metric(gpu_metrics_ptr);
  status_code = dev_read_gpu_metrics_all_data();
  if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    ostrstream << __PRETTY_FUNCTION__
                << " | ======= end ======= "
                << " | Fail "
                << " | Device #: " << index()
                << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header())
                << " | Cause: dev_read_gpu_metrics_all_data() couldn't read gpu metric data!"
                << " | Returning = "
                << getRSMIStatusString(status_code)
                << " |";
    LOG_ERROR(ostrstream);
    return status_code;
  }

  return status_code;
}

rsmi_status_t Device::dev_log_gpu_metrics()
{
  std::ostringstream ostrstream;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ostrstream << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ostrstream);

  // At this point we should have a valid gpu_metrics pointer.
  if (!m_gpu_metrics_ptr) {
    status_code = rsmi_status_t::RSMI_STATUS_UNEXPECTED_DATA;
    ostrstream << __PRETTY_FUNCTION__
                << " | ======= end ======= "
                << " | Fail "
                << " | Device #: " << index()
                << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header())
                << " | Cause: Couldn't get a valid metric object"
                << " | Returning = "
                << getRSMIStatusString(status_code)
                << " |";
    LOG_ERROR(ostrstream);
    return status_code;
  }

  //  Header info
  auto header_output = [&]() {
    const auto gpu_metrics_header = dev_get_metrics_header();
    ostrstream << "GPU Metrics Header: \n";
    ostrstream << "Timestamp: " << m_gpu_metrics_updated_timestamp << "\n";
    ostrstream << "Based on:  " << static_cast<uint32_t>(m_gpu_metrics_ptr->get_gpu_metrics_version_used()) << "\n";
    ostrstream << print_unsigned_hex_and_int(gpu_metrics_header.m_structure_size,   " ->structure_size ");
    ostrstream << print_unsigned_hex_and_int(gpu_metrics_header.m_format_revision,  " ->format_revision ");
    ostrstream << print_unsigned_hex_and_int(gpu_metrics_header.m_content_revision, " ->content_revision ");
    LOG_DEBUG(ostrstream);
    return;
  };

  auto table_content_output = [&]() {
    const auto gpu_metrics_tbl = m_gpu_metrics_ptr->get_metrics_dynamic_tbl();

    ostrstream << "GPU Metrics Data: \n";
    for (const auto& [metric_class, metric_data] : gpu_metrics_tbl) {
      ostrstream << amdgpu_metrics_class_id_translation_table.at(metric_class) << "\n";

      for (const auto& [metric_unit, metric_values] : metric_data) {
        for (const auto& metric_value : metric_values) {
          ostrstream << print_unsigned_hex_and_int(metric_value.m_value, metric_value.m_info);
        }
      }
    }
    return;
  };

  //
  header_output();
  table_content_output();

  return status_code;
}

rsmi_status_t Device::run_internal_gpu_metrics_query(AMDGpuMetricsUnitType_t metric_counter, AMDGpuDynamicMetricTblValues_t& values)
{
  std::ostringstream ostrstream;
  auto status_code(rsmi_status_t::RSMI_STATUS_NOT_FOUND);
  ostrstream << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ostrstream);

  status_code = setup_gpu_metrics_reading();
  if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    return status_code;
  }

  if (!m_gpu_metrics_ptr) {
    status_code = rsmi_status_t::RSMI_STATUS_UNEXPECTED_DATA;
    ostrstream << __PRETTY_FUNCTION__
                << " | ======= end ======= "
                << " | Fail "
                << " | Device #: " << index()
                << " | Metric Version: " << stringfy_metrics_header(dev_get_metrics_header())
                << " | Cause: Couldn't get a valid metric object"
                << " | Returning = "
                << getRSMIStatusString(status_code)
                << " |";
    LOG_ERROR(ostrstream);
    return status_code;
  }

  // Lookup the dynamic table
  const auto gpu_metrics_tbl = m_gpu_metrics_ptr->get_metrics_dynamic_tbl();
  for (const auto& [metric_class, metric_data] : gpu_metrics_tbl) {
    for (const auto& [metric_unit, metric_values] : metric_data) {
      if (metric_unit == metric_counter) {
        values = metric_values;
        return rsmi_status_t::RSMI_STATUS_SUCCESS;
      }
    }
  }

  return status_code;
}


template<typename> struct is_vector : std::false_type {};
template <typename... Ts> struct is_vector<std::vector<Ts...>> : std::true_type {};
template <typename T> inline constexpr bool is_vector_v = is_vector<T>::value;

template <typename> struct is_array : std::false_type {};
template <typename T, std::size_t N> struct is_array<std::array<T, N>> : std::true_type {};

template <class> struct is_bounded_uint8_array : std::false_type {};
template <class> struct is_bounded_uint16_array : std::false_type {};
template <class> struct is_bounded_uint32_array : std::false_type {};
template <class> struct is_bounded_uint64_array : std::false_type {};

template <size_t N>
struct is_bounded_uint8_array<uint8_t[N]> : std::true_type {};

template <size_t N>
struct is_bounded_uint16_array<uint16_t[N]> : std::true_type {};

template <size_t N>
struct is_bounded_uint32_array<uint32_t[N]> : std::true_type {};

template <size_t N>
struct is_bounded_uint64_array<uint64_t[N]> : std::true_type {};

template <class> struct is_bounded_array : std::false_type {};

template <class T, size_t N>
struct is_bounded_array<T[N]> : std::true_type {};


template<typename T>
rsmi_status_t rsmi_dev_gpu_metrics_info_query(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, T& metric_value)
{
  std::ostringstream ostrstream;
  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  ostrstream << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ostrstream);

  AMDGpuDynamicMetricTblValues_t tmp_values{};
  GET_DEV_FROM_INDX
  status_code = dev->run_internal_gpu_metrics_query(metric_counter, tmp_values);
  if ((status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) || tmp_values.empty()) {
    ostrstream << __PRETTY_FUNCTION__
                << " | ======= end ======= "
                << " | Fail "
                << " | Device #: " << dv_ind
                << " | Metric Version: " << stringfy_metrics_header(dev->dev_get_metrics_header())
                << " | Cause: Couldn't find metric/counter requested"
                << " | Metric Type: " << static_cast<uint32_t>(metric_counter)
                << " " << amdgpu_metrics_unit_type_translation_table.at(metric_counter)
                << " | Values: " << tmp_values.size()
                << " | Returning = "
                << getRSMIStatusString(status_code)
                << " |";
    LOG_ERROR(ostrstream);
    return status_code;
  }

  const auto num_stored_elems = (std::end(tmp_values) - std::begin(tmp_values));
  if constexpr (std::is_array_v<T>) {
    std::variant<uint8_t, uint16_t, uint32_t, uint64_t> tmp_value;
    auto idx = uint16_t(0);
    for (const auto& value : tmp_values) {
      tmp_value = value.m_value;
      idx++;
      switch (value.m_original_type) {
        case AMDGpuMetricsDataType_t::kUInt8:
          metric_value[idx] = std::get<uint8_t>(tmp_value);
          break;

        case AMDGpuMetricsDataType_t::kUInt16:
          metric_value[idx] = std::get<uint16_t>(tmp_value);
          break;

        case AMDGpuMetricsDataType_t::kUInt32:
          metric_value[idx] = std::get<uint32_t>(tmp_value);
          break;

        case AMDGpuMetricsDataType_t::kUInt64:
          metric_value[idx] = std::get<uint64_t>(tmp_value);
          break;

        default:
          break;
      }

      metric_value[idx++] = tmp_value;
    }
  }
  if constexpr ((std::is_same_v<T, uint8_t>)  || (std::is_same_v<T, uint16_t>) ||
                (std::is_same_v<T, uint32_t>) || (std::is_same_v<T, uint64_t>)) {
    T tmp_value(0);
    tmp_value = static_cast<decltype(tmp_value)>(tmp_values[0].m_value);
    metric_value = tmp_value;
  }

  return status_code;
}


template
rsmi_status_t rsmi_dev_gpu_metrics_info_query<uint8_t>(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, uint8_t& metric_value);

template
rsmi_status_t rsmi_dev_gpu_metrics_info_query<uint16_t>(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, uint16_t& metric_value);

template
rsmi_status_t rsmi_dev_gpu_metrics_info_query<uint32_t>(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, uint32_t& metric_value);

template
rsmi_status_t rsmi_dev_gpu_metrics_info_query<uint64_t>(uint32_t dv_ind, AMDGpuMetricsUnitType_t metric_counter, uint64_t& metric_value);


} //namespace amd::smi
