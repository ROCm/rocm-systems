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

// amd-smi's private view of the librocdxg ABI, which is dlopen'd and never
// linked. Mirroring it here keeps the build free of any path into another
// project's source tree; layout must match rocr-runtime's
// libhsakmt/include/hsakmt/{rocdxg_smi.h,hsakmttypes.h} exactly, because
// librocdxg writes into structs allocated here. VERIFY_ROCDXG_ABI checks that.

#ifndef AMD_SMI_INCLUDE_IMPL_WSL_ROCDXG_ABI_H_
#define AMD_SMI_INCLUDE_IMPL_WSL_ROCDXG_ABI_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// hsakmt core types (mirrors hsakmttypes.h)
// ---------------------------------------------------------------------------

typedef uint32_t HSAuint32;

typedef enum _HSAKMT_STATUS {
  HSAKMT_STATUS_SUCCESS = 0,
  HSAKMT_STATUS_ERROR = 1,
  HSAKMT_STATUS_DRIVER_MISMATCH = 2,
  HSAKMT_STATUS_INVALID_PARAMETER = 3,
  HSAKMT_STATUS_INVALID_HANDLE = 4,
  HSAKMT_STATUS_INVALID_NODE_UNIT = 5,
  HSAKMT_STATUS_NO_MEMORY = 6,
  HSAKMT_STATUS_BUFFER_TOO_SMALL = 7,
  HSAKMT_STATUS_NOT_IMPLEMENTED = 10,
  HSAKMT_STATUS_NOT_SUPPORTED = 11,
  HSAKMT_STATUS_UNAVAILABLE = 12,
  HSAKMT_STATUS_OUT_OF_RESOURCES = 13,
  HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED = 20,
  HSAKMT_STATUS_KERNEL_COMMUNICATION_ERROR = 21,
  HSAKMT_STATUS_KERNEL_ALREADY_OPENED = 22,
  HSAKMT_STATUS_HSAMMU_UNAVAILABLE = 23,
  HSAKMT_STATUS_WAIT_FAILURE = 30,
  HSAKMT_STATUS_WAIT_TIMEOUT = 31,
  HSAKMT_STATUS_MEMORY_ALREADY_REGISTERED = 35,
  HSAKMT_STATUS_MEMORY_NOT_REGISTERED = 36,
  HSAKMT_STATUS_MEMORY_ALIGNMENT = 37,
} HSAKMT_STATUS;

// hsaKmtAcquireSystemProperties() output. amd-smi calls that function for its
// side effect (it builds the WDDM device list every rocdxg_smi_* call needs);
// the returned counts themselves are unused.
typedef struct _HsaSystemProperties {
  HSAuint32 NumNodes;
  HSAuint32 PlatformOem;
  HSAuint32 PlatformId;
  HSAuint32 PlatformRev;
} HsaSystemProperties;

// ---------------------------------------------------------------------------
// rocdxg_smi types (mirrors rocdxg_smi.h)
// ---------------------------------------------------------------------------

#define ROCDXG_SMI_MAX_STRING_LENGTH 256
#define ROCDXG_SMI_MAX_CACHE_TYPES 10
#define ROCDXG_SMI_MAX_FW_ENTRIES 32

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
  uint32_t num_xcc;
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

typedef struct rocdxg_smi_fw_entry {
  uint32_t fw_id;
  uint64_t fw_version;
} rocdxg_smi_fw_entry_t;

typedef struct rocdxg_smi_fw_info {
  rocdxg_smi_fw_entry_t entries[ROCDXG_SMI_MAX_FW_ENTRIES];
  uint32_t num_fw_info;
} rocdxg_smi_fw_info_t;

// struct_size is set by the caller to sizeof(rocdxg_smi_device_info_t).
// librocdxg rejects a mismatch with BUFFER_TOO_SMALL rather than writing past
// the end of this struct, which is what makes the mirrored layout below safe
// against a librocdxg built from different headers. Keep it first.
typedef struct rocdxg_smi_device_info {
  uint32_t struct_size;
  rocdxg_smi_bdf_info_t bdf;
  rocdxg_smi_asic_info_t asic;
  rocdxg_smi_board_info_t board;
  rocdxg_smi_vram_info_t vram;
  rocdxg_smi_driver_info_t driver;
  rocdxg_smi_vbios_info_t vbios;
  rocdxg_smi_cache_info_t cache;
  rocdxg_smi_fw_info_t fw;
} rocdxg_smi_device_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMD_SMI_INCLUDE_IMPL_WSL_ROCDXG_ABI_H_
