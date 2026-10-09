// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#ifndef ROCDXG_SMI_H_
#define ROCDXG_SMI_H_

#include <stdint.h>

#include "hsakmt/hsakmt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROCDXG_SMI_MAX_STRING_LENGTH 256

typedef struct rocdxg_smi_bdf_info {
  uint32_t domain_number;
  uint32_t bus_number;
  uint32_t device_number;
  uint32_t function_number;
} rocdxg_smi_bdf_info_t;

typedef struct rocdxg_smi_asic_info {
  uint64_t device_id;
  uint32_t vendor_id;
  uint32_t subvendor_id;
  uint32_t subsystem_id;
  uint32_t rev_id;
  uint64_t asic_serial;
  char market_name[ROCDXG_SMI_MAX_STRING_LENGTH];
  uint32_t num_of_compute_units;
  uint64_t target_graphics_version;
} rocdxg_smi_asic_info_t;

typedef struct rocdxg_smi_board_info {
  char product_name[ROCDXG_SMI_MAX_STRING_LENGTH];
  char manufacturer_name[ROCDXG_SMI_MAX_STRING_LENGTH];
} rocdxg_smi_board_info_t;

typedef struct rocdxg_smi_vram_info {
  uint32_t vram_type;
  uint32_t vram_bit_width;
  uint64_t vram_size_mb;
} rocdxg_smi_vram_info_t;

typedef struct rocdxg_smi_vram_usage {
  uint64_t vram_used_mb;
  uint64_t vram_total_mb;
} rocdxg_smi_vram_usage_t;

typedef struct rocdxg_smi_power_info {
  uint32_t current_socket_power;
  uint32_t gfx_voltage;
  uint32_t soc_voltage;
  uint32_t mem_voltage;
  uint32_t power_limit;
} rocdxg_smi_power_info_t;

typedef struct rocdxg_smi_clock_info {
  uint32_t clk;
  uint32_t min_clk;
  uint32_t max_clk;
  uint8_t clk_locked;
  uint8_t clk_deep_sleep;
} rocdxg_smi_clock_info_t;

typedef struct rocdxg_smi_pcie_info {
  uint16_t max_pcie_width;
  uint32_t max_pcie_speed;
  uint32_t pcie_interface_version;
  uint32_t slot_type;
  uint16_t pcie_width;
  uint32_t pcie_speed;
  uint32_t pcie_bandwidth;
  uint64_t pcie_replay_count;
  uint64_t pcie_l0_to_recovery_count;
  uint64_t pcie_replay_roll_over_count;
  uint64_t pcie_nak_sent_count;
  uint64_t pcie_nak_received_count;
} rocdxg_smi_pcie_info_t;

typedef struct rocdxg_smi_driver_info {
  char driver_version[ROCDXG_SMI_MAX_STRING_LENGTH];
  char driver_date[ROCDXG_SMI_MAX_STRING_LENGTH];
  char driver_name[ROCDXG_SMI_MAX_STRING_LENGTH];
} rocdxg_smi_driver_info_t;

typedef struct rocdxg_smi_vbios_info {
  char name[ROCDXG_SMI_MAX_STRING_LENGTH];
  char build_date[ROCDXG_SMI_MAX_STRING_LENGTH];
  char part_number[ROCDXG_SMI_MAX_STRING_LENGTH];
  char version[ROCDXG_SMI_MAX_STRING_LENGTH];
  char boot_firmware[ROCDXG_SMI_MAX_STRING_LENGTH];
} rocdxg_smi_vbios_info_t;

typedef struct rocdxg_smi_gpu_metrics_info {
  uint32_t temperature_edge;
  uint32_t temperature_hotspot;
  uint32_t temperature_mem;
  uint32_t average_gfx_activity;
  uint32_t average_umc_activity;
  uint32_t current_socket_power;
  uint32_t current_gfxclk;
  uint32_t current_socclk;
  uint32_t current_fan_speed;
  uint32_t current_fan_speed_percent;
  uint32_t voltage_soc;
  uint32_t voltage_gfx;
  uint32_t voltage_mem;
} rocdxg_smi_gpu_metrics_info_t;

typedef struct rocdxg_smi_process_info {
  uint32_t process_id;
  uint64_t vram_usage_bytes;
  uint64_t sdma_usage;
  uint64_t cu_occupancy;
  uint64_t engine_usage;
  uint64_t evicted_time;
} rocdxg_smi_process_info_t;

#define ROCDXG_SMI_MAX_CACHE_TYPES 10

typedef struct rocdxg_smi_cache_entry {
  uint32_t cache_size_kb;
  uint32_t cache_level;
  uint32_t cache_properties;
  uint32_t max_num_cu_shared;
  uint32_t num_cache_instance;
} rocdxg_smi_cache_entry_t;

typedef struct rocdxg_smi_cache_info {
  uint32_t num_cache_types;
  rocdxg_smi_cache_entry_t cache[ROCDXG_SMI_MAX_CACHE_TYPES];
} rocdxg_smi_cache_info_t;

#define ROCDXG_SMI_MAX_FW_ENTRIES 32

typedef struct rocdxg_smi_fw_entry {
  uint32_t fw_id;
  uint64_t fw_version;
} rocdxg_smi_fw_entry_t;

typedef struct rocdxg_smi_fw_info {
  rocdxg_smi_fw_entry_t entries[ROCDXG_SMI_MAX_FW_ENTRIES];
  uint32_t num_fw_info;
} rocdxg_smi_fw_info_t;

HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_device_count(uint32_t* count);
HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_vram_usage(uint32_t node_id,
                                                 rocdxg_smi_vram_usage_t* usage);
HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_power_info(uint32_t node_id,
                                                 rocdxg_smi_power_info_t* info);
HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_temperature(uint32_t node_id,
                                                  uint32_t sensor_type,
                                                  uint32_t metric,
                                                  int64_t* temperature);
HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_clock_info(uint32_t node_id,
                                                 uint32_t clk_type,
                                                 rocdxg_smi_clock_info_t* info);
HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_pcie_info(uint32_t node_id,
                                                rocdxg_smi_pcie_info_t* info);
HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_gpu_metrics_info(
    uint32_t node_id, rocdxg_smi_gpu_metrics_info_t* info);
HSAKMT_STATUS HSAKMTAPI rocdxg_smi_enum_processes(uint32_t node_id,
                                                  uint32_t* num_processes,
                                                  rocdxg_smi_process_info_t* processes);

#ifdef __cplusplus
}  // extern "C"
#endif


typedef struct rocdxg_smi_device_info {
  rocdxg_smi_bdf_info_t   bdf;
  rocdxg_smi_asic_info_t  asic;
  rocdxg_smi_board_info_t board;
  rocdxg_smi_vram_info_t  vram;
  rocdxg_smi_driver_info_t driver;
  rocdxg_smi_vbios_info_t  vbios;
  rocdxg_smi_cache_info_t  cache;
  rocdxg_smi_fw_info_t     fw;
} rocdxg_smi_device_info_t;

#ifdef __cplusplus
extern "C" {
#endif
HSAKMT_STATUS HSAKMTAPI rocdxg_smi_get_device_info(uint32_t node_id,
                                                   rocdxg_smi_device_info_t* info);
#ifdef __cplusplus
}
#endif
#endif  // ROCDXG_SMI_H_
