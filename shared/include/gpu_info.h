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

} // namespace thunk
} // namespace wsl

#endif // _WSL_INC_GPU_INFO_H_
