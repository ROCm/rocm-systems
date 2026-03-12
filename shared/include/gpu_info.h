#ifndef _WSL_INC_GPU_INFO_H_
#define _WSL_INC_GPU_INFO_H_

#include <cstdint>

namespace wsl {
namespace thunk {

// GPU VRAM static properties (from KMD adapter info).
// vram_type holds a raw VIDEO_MEMORY_TYPE integer value.
struct VramInfo {
  uint32_t vram_type;      // raw VIDEO_MEMORY_TYPE value
  uint32_t vram_bit_width;
  uint64_t vram_size_mb;   // LocalVisible + LocalInvisible in MB
};

// GPU VRAM dynamic usage (from KMD escape).
struct VramUsage {
  uint64_t vram_used_mb;   // current VRAM usage in MB
  uint64_t vram_total_mb;  // total VRAM in MB
};

// RAS (Reliability, Availability, Serviceability) feature flags
struct RasFeature {
  uint32_t dram_ecc     : 1;  // DRAM ECC enabled
  uint32_t sram_ecc     : 1;  // SRAM ECC enabled
  uint32_t poisoning    : 1;  // data poisoning enabled
  uint32_t rsvd         : 29;
  bool     needs_reboot;      // immediate reboot required after feature change
};

// PCI Bus/Device/Function location.
struct BdfInfo {
  uint32_t domain_number;
  uint32_t bus_number;
  uint32_t device_number;
  uint32_t function_number;
};

// Power and voltage sensor readings from KMD PMLog escape.
struct PowerInfo {
  uint32_t current_socket_power;  // ASIC_POWER sensor in W (UINT32_MAX if unavailable)
  uint32_t gfx_voltage;           // GFX_VOLTAGE sensor in mV
  uint32_t soc_voltage;           // SOC_VOLTAGE sensor in mV
  uint32_t mem_voltage;           // MEM_VOLTAGE sensor in mV
  uint32_t power_limit;           // max power limit from sensor limits
};

// Firmware version entry: fw_id matches amdsmi_fw_block_t integer values.
struct FwEntry {
  uint32_t fw_id;       // amdsmi_fw_block_t cast to uint32_t
  uint64_t fw_version;
};

// Firmware version table read from KMD adapter info.
// num_fw_info entries are valid; the rest are zero-initialised.
constexpr uint32_t kMaxFwEntries = 32;
struct FwInfo {
  FwEntry  entries[kMaxFwEntries];
  uint32_t num_fw_info;
};

// GPU engine activity readings from KMD PMLog escape.
struct GpuActivity {
  uint32_t gfx_activity;  // GFX engine utilization in %
  uint32_t umc_activity;  // Memory controller utilization in % (not available via PMLog, always 0)
  uint32_t mm_activity;   // Multimedia engine utilization in %
};

// Clock frequency readings from KMD PMLog escape.
struct ClockInfo {
  uint32_t clk;            // current clock in MHz
  uint32_t min_clk;        // minimum clock in MHz
  uint32_t max_clk;        // maximum clock in MHz
  uint8_t  clk_locked;     // clock locked flag
  uint8_t  clk_deep_sleep; // deep-sleep capable flag
};

// VBIOS version and identification information from KMD CWDDE escape.
struct VBiosInfo {
  char name[256];          // product name from VBIOS ROM image
  char build_date[256];    // BIOS date info (yyyy/mm/dd hh:mm)
  char part_number[256];   // BIOS part number
  char version[256];       // BIOS version (XXX.YYY.MMM.NNN)
  char boot_firmware[256]; // (not filled via KMD escape)
};

// ASIC static information sourced from KMD adapter info.
struct AsicInfo {
  uint64_t device_id;          // PCI device ID
  uint32_t vendor_id;          // PCI vendor ID
  uint32_t subvendor_id;       // subsystem vendor ID (high 16 bits of ulSubsystemID)
  uint32_t subsystem_id;       // subsystem device ID (low  16 bits of ulSubsystemID)
  uint32_t rev_id;             // silicon revision
  uint64_t asic_serial;        // ProductSerialNumber (unique ID)
  char     market_name[256];   // adapter string from registry
  uint32_t num_of_compute_units;      // WGP*2
  uint64_t target_graphics_version;   // major<<16|minor<<8|stepping
};

} // namespace thunk
} // namespace wsl

#endif // _WSL_INC_GPU_INFO_H_
