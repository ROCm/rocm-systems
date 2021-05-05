/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2021, Advanced Micro Devices, Inc.
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

#include <assert.h>
#include <dirent.h>

#include <fstream>
#include <string>
#include <cstdint>
#include <map>
#include <iostream>
#include <algorithm>
#include <regex>  // NOLINT
#include <vector>
#include <pthread.h>

#include "rocm_smi/rocm_smi_common.h"  // Should go before rocm_smi.h
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_monitor.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"

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


static rsmi_status_t GetGPUMetricsFormat1(uint32_t dv_ind,
                                rsmi_gpu_metrics_t *data, uint8_t content_v) {
  assert(content_v != RSMI_GPU_METRICS_API_CONTENT_VER);
  if (content_v == RSMI_GPU_METRICS_API_CONTENT_VER) {
    // This function shouldn't be called if content version is
    // RSMI_GPU_METRICS_API_CONTENT_VER.
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
  data->FIELD = SRC->FIELD;

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
    data->mem_actvity_acc  = 0;
    (void)memset(data->temperature_hbm, 0,
                                   RSMI_NUM_HBM_INSTANCES * sizeof(uint16_t));
  }  // else handle other conversions to format 1
#undef ASSIGN_DATA_FIELD
#undef ASSIGN_COMMON_FORMATS
  return RSMI_STATUS_SUCCESS;
}

rsmi_status_t
rsmi_dev_gpu_metrics_info_get(uint32_t dv_ind, rsmi_gpu_metrics_t *smu) {
  TRY
  DEVICE_MUTEX
  CHK_SUPPORT_NAME_ONLY(smu)
  rsmi_status_t ret;

  if (!dev->gpu_metrics_ver().structure_size) {
    ret = GetDevBinaryBlob(amd::smi::kDevGpuMetrics, dv_ind,
              sizeof(struct metrics_table_header_t), &dev->gpu_metrics_ver());

    if (ret != RSMI_STATUS_SUCCESS) {
      return ret;
    }
  }
  // only supports gpu_metrics_v1_x version
  if (dev->gpu_metrics_ver().format_revision != 1) {
    return RSMI_STATUS_NOT_SUPPORTED;
  } else {  // format_revision == 1
    if (dev->gpu_metrics_ver().content_revision ==
                                           RSMI_GPU_METRICS_API_CONTENT_VER) {
      ret = GetDevBinaryBlob(amd::smi::kDevGpuMetrics, dv_ind,
                                           sizeof(rsmi_gpu_metrics_t), smu);
    } else {
      ret = GetGPUMetricsFormat1(dv_ind, smu,
                                     dev->gpu_metrics_ver().content_revision);
    }
  }

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  return ret;
  CATCH
}
