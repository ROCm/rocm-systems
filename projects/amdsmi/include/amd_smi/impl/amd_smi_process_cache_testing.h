// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace amd::smi {

struct ComputeProcessCache;

// Library-local test seam for the per-GPU cache behind the compute-process list.
// Returns the cache for gpu_id, creating it on first use; the registry owns it.
// Shared by the definition (src/amd_smi/amd_smi_gpu_device.cc) and the unit
// tests so the signature stays in sync.
ComputeProcessCache* get_compute_process_cache(uint32_t gpu_id);

}  // namespace amd::smi
