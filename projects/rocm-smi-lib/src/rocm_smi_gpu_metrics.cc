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

#include <dirent.h>
#include <pthread.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>  // NOLINT
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi_common.h"  // Should go before rocm_smi.h
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_logger.h"

using namespace ROCmLogging;
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
