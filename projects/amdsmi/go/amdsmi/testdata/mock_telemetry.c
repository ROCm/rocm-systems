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

amdsmi_status_t amdsmi_is_gpu_power_management_enabled(amdsmi_processor_handle processor,
                                                       bool* enabled) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!enabled || !mock_is_zero(enabled, sizeof(*enabled)))
    return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (status != AMDSMI_STATUS_SUCCESS && mock_mode(__func__) != 2) return mock_finish(status);
  *enabled = mock_mode(__func__) != 1;
  return mock_finish(status);
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

amdsmi_status_t amdsmi_get_pcie_info(amdsmi_processor_handle processor, amdsmi_pcie_info_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (status != AMDSMI_STATUS_SUCCESS && mock_mode(__func__) != 2) return mock_finish(status);
  if (mock_mode(__func__) == 1) {
    memset(out, 0xff, sizeof(*out));
    out->pcie_static.slot_type = (amdsmi_card_form_factor_t)9999;
    return mock_finish(AMDSMI_STATUS_SUCCESS);
  }
  out->pcie_static.max_pcie_width = 16;
  out->pcie_static.max_pcie_speed = 32000;
  out->pcie_static.pcie_interface_version = 4;
  out->pcie_static.slot_type = AMDSMI_CARD_FORM_FACTOR_OAM;
  out->pcie_static.max_pcie_interface_version = 5;
  out->pcie_metric.pcie_width = 8;
  out->pcie_metric.pcie_speed = 16000;
  out->pcie_metric.pcie_bandwidth = 123456;
  out->pcie_metric.pcie_replay_count = (UINT64_C(1) << 40) + 1;
  out->pcie_metric.pcie_l0_to_recovery_count = (UINT64_C(1) << 41) + 2;
  out->pcie_metric.pcie_replay_roll_over_count = (UINT64_C(1) << 42) + 3;
  out->pcie_metric.pcie_nak_sent_count = (UINT64_C(1) << 43) + 4;
  out->pcie_metric.pcie_nak_received_count = (UINT64_C(1) << 44) + 5;
  out->pcie_metric.pcie_lc_perf_other_end_recovery_count = 12345;
  return mock_finish(status);
}

amdsmi_status_t amdsmi_get_link_metrics(amdsmi_processor_handle processor,
                                        amdsmi_link_metrics_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  if (status != AMDSMI_STATUS_SUCCESS && mock_mode(__func__) != 5) return mock_finish(status);
  out->num_links = 2;
  out->links[0].bdf.bdf.domain_number = UINT64_C(0xabcde1234567);
  out->links[0].bdf.bdf.bus_number = 0x89;
  out->links[0].bdf.bdf.device_number = 0x15;
  out->links[0].bdf.bdf.function_number = 3;
  out->links[0].bit_rate = 32;
  out->links[0].max_bandwidth = 64;
  out->links[0].link_type = AMDSMI_LINK_TYPE_XGMI;
  out->links[0].read = (UINT64_C(1) << 40) + 11;
  out->links[0].write = (UINT64_C(1) << 41) + 13;
  out->links[0].link_status = AMDSMI_LINK_STATUS_ENABLED;
  memset(&out->links[1], 0xff, sizeof(out->links[1]));
  out->links[1].link_type = (amdsmi_link_type_t)9999;
  out->links[1].link_status = (amdsmi_link_status_t)8888;
  uint32_t mode = mock_mode(__func__);
  if (mode == 1) out->num_links = 0;
  if (mode == 2) {
    out->num_links = AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK;
    for (uint32_t index = 1; index < out->num_links; ++index) {
      out->links[index] = out->links[0];
      out->links[index].read += index;
    }
  }
  if (mode == 3) out->num_links = AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK + 1;
  if (mode == 4) out->num_links = UINT32_MAX;
  return mock_finish(status);
}

amdsmi_status_t amdsmi_get_soc_pstate(amdsmi_processor_handle processor, amdsmi_dpm_policy_t* out) {
  amdsmi_status_t status = mock_begin(__func__, processor);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  out->num_supported = 2;
  out->current = 1;
  out->policies[0].policy_id = 3;
  memcpy(out->policies[0].policy_description, "balanced", sizeof("balanced"));
  out->policies[1].policy_id = 9;
  memcpy(out->policies[1].policy_description, "performance", sizeof("performance"));
  uint32_t mode = mock_mode(__func__);
  if (mode == 1) {
    out->num_supported = 0;
    out->current = UINT32_MAX;
  }
  if (mode == 2) {
    out->num_supported = AMDSMI_MAX_NUM_PM_POLICIES;
    out->current = AMDSMI_MAX_NUM_PM_POLICIES - 1;
    for (uint32_t index = 0; index < out->num_supported; ++index) {
      out->policies[index] = out->policies[0];
      out->policies[index].policy_id = 0x100 + index;
    }
  }
  if (mode == 3) out->num_supported = AMDSMI_MAX_NUM_PM_POLICIES + 1;
  if (mode == 4) out->num_supported = UINT32_MAX;
  if (mode == 5 || mode == 6) {
    out->current = mode == 5 ? UINT32_MAX : 0x12345678;
    out->policies[0].policy_id = UINT32_MAX;
    out->policies[0].policy_description[0] = '\0';
  }
  if (mode == 7) {
    memset(out->policies[0].policy_description, 'P', sizeof(out->policies[0].policy_description));
    out->policies[1].policy_id = UINT32_MAX;
    memcpy(out->policies[1].policy_description, "eco\0ignored", sizeof("eco\0ignored"));
  }
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
  uint32_t mode = mock_mode(__func__);
  if (mode == 1 || mode == 2) {
    out->clk_locked = (uint8_t)(mode - 1);
    out->clk_deep_sleep = (uint8_t)(2 - mode);
  }
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
