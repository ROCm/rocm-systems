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

} // namespace thunk
} // namespace wsl

#endif // _WSL_INC_GPU_INFO_H_
