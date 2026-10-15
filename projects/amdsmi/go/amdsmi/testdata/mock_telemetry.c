// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <string.h>

#include "mock.h"

amdsmi_status_t amdsmi_get_temp_metric(amdsmi_processor_handle processor,
                                       amdsmi_temperature_type_t sensor,
                                       amdsmi_temperature_metric_t metric, int64_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (sensor != AMDSMI_TEMPERATURE_TYPE_HOTSPOT || metric != AMDSMI_TEMP_CURRENT)
    return mock_finish(AMDSMI_STATUS_INVAL);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  *out = -17;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_power_info(amdsmi_processor_handle processor, amdsmi_power_info_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  out->socket_power = UINT64_MAX;
  out->current_socket_power = UINT32_MAX;
  out->average_socket_power = 275;
  out->gfx_voltage = UINT64_MAX;
  out->soc_voltage = 900;
  out->mem_voltage = 850;
  out->power_limit = 350000000;
  out->ubb_power = UINT32_MAX;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_power_cap_info(amdsmi_processor_handle processor, uint32_t sensor,
                                          amdsmi_power_cap_info_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (sensor != 7) return mock_finish(AMDSMI_STATUS_INVAL);
  uint32_t mode = mock_mode(__func__);
  if (status != AMDSMI_STATUS_SUCCESS && mode != 2) return mock_finish(status);
  out->power_cap = 300000000;
  if (mode == 1) return mock_finish(AMDSMI_STATUS_SUCCESS);
  out->default_power_cap = 325000000;
  out->dpm_cap = 3;
  out->min_power_cap = 100000000;
  out->max_power_cap = 400000000;
  return mock_finish(status);
}

amdsmi_status_t amdsmi_get_clock_info(amdsmi_processor_handle processor, amdsmi_clk_type_t clock,
                                      amdsmi_clk_info_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (clock != AMDSMI_CLK_TYPE_GFX) return mock_finish(AMDSMI_STATUS_INVAL);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  out->clk = UINT32_MAX;
  out->min_clk = 500;
  out->max_clk = 2100;
  out->clk_locked = UINT8_MAX;
  out->clk_deep_sleep = 222;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_clk_freq(amdsmi_processor_handle processor, amdsmi_clk_type_t clock,
                                    amdsmi_frequencies_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (clock != AMDSMI_CLK_TYPE_MEM) return mock_finish(AMDSMI_STATUS_INVAL);
  uint32_t mode = mock_mode(__func__);
  if (status != AMDSMI_STATUS_SUCCESS && mode != 4) return mock_finish(status);
  if (mode == 1) {
    out->num_supported = AMDSMI_MAX_NUM_FREQUENCIES + 1;
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  if (mode == 2) {
    out->has_deep_sleep = true;
    out->num_supported = 0;
    out->current = UINT32_MAX;
    out->frequency[0] = UINT64_MAX;
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  if (mode == 3) {
    out->num_supported = AMDSMI_MAX_NUM_FREQUENCIES;
    out->current = AMDSMI_MAX_NUM_FREQUENCIES - 1;
    for (uint32_t i = 0; i < AMDSMI_MAX_NUM_FREQUENCIES; ++i)
      out->frequency[i] = (uint64_t)(i + 1) * 1000000;
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  out->has_deep_sleep = true;
  out->num_supported = 2;
  out->current = UINT32_MAX;
  out->frequency[0] = 500000000;
  out->frequency[1] = UINT64_C(2400000000);
  return mock_finish(status);
}

amdsmi_status_t amdsmi_get_gpu_activity(amdsmi_processor_handle processor,
                                        amdsmi_engine_usage_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  out->gfx_activity = UINT16_MAX;
  out->umc_activity = 43;
  out->mm_activity = 7;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_memory_total(amdsmi_processor_handle processor,
                                            amdsmi_memory_type_t memory, uint64_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (memory != AMDSMI_MEM_TYPE_VRAM) return mock_finish(AMDSMI_STATUS_INVAL);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  *out = UINT64_C(1) << 40;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_memory_usage(amdsmi_processor_handle processor,
                                            amdsmi_memory_type_t memory, uint64_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (memory != AMDSMI_MEM_TYPE_GTT) return mock_finish(AMDSMI_STATUS_INVAL);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  *out = (UINT64_C(1) << 39) + 7;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_vram_info(amdsmi_processor_handle processor,
                                         amdsmi_vram_info_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  uint32_t mode = mock_mode(__func__);
  out->vram_type = mode == 2 ? (amdsmi_vram_type_t)9999 : AMDSMI_VRAM_TYPE_LPDDR5;
  if (mode == 1) {
    memset(out->vram_vendor, 'V', sizeof(out->vram_vendor));
  } else {
    memcpy(out->vram_vendor, "vendor", sizeof("vendor"));
  }
  out->vram_size = 196608;
  out->vram_bit_width = UINT32_MAX;
  out->vram_max_bandwidth = 5300;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}
