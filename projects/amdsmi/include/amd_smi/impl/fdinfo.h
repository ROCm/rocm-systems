// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef __FDINFO__
#define __FDINFO__

#include <vector>

#include "amd_smi/amdsmi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Determine, via KFD, whether process `pid` uses the GPU identified by `bdf`.
// `known_kfd_gpu_id` is this device's KFD gpu id; supplying it lets the lookup
// skip rebuilding the entire KFD topology just to translate the BDF. Pass 0 or
// UINT64_MAX to force the topology-discovery fallback that re-derives the id.
amdsmi_status_t gpu_is_in_kfd_pid(const amdsmi_bdf_t& bdf, long pid, uint64_t known_kfd_gpu_id);

// Populate `info` with per-process GPU usage for `pid` on the device `bdf`.
// `kfd_gpu_id` is this device's KFD gpu id, forwarded to gpu_is_in_kfd_pid() so
// the lookup need not rebuild the KFD topology. Pass 0 or UINT64_MAX to fall
// back to topology discovery.
amdsmi_status_t gpuvsmi_get_pid_info(const amdsmi_bdf_t& bdf, long int pid,
                                     amdsmi_proc_info_t& info, uint64_t kfd_gpu_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
