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
