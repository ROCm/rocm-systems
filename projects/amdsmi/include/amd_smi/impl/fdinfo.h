// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef __FDINFO__
#define __FDINFO__

#include <sys/types.h>

#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"

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

// Reads the fdinfo text of the calling process's descriptor `fd` into `text`.
// Returns AMDSMI_STATUS_NOT_FOUND if the descriptor is gone.
using SelfFdinfoReader = amdsmi_status_t (*)(int fd, std::string* text);

namespace amd::smi::test_hooks {
// Reader amdsmi_get_gpu_process_info_self() uses for the caller's fdinfo files. Defaults to null,
// which reads /proc/self/fdinfo; a test supplies fake DRM client text for descriptors it opened.
AMDSMI_TEST_HOOK const SelfFdinfoReader& SelfFdinfoReaderHook();
}  // namespace amd::smi::test_hooks

// Populate `info` with the calling process's usage of the DRM render node whose
// device number is `render_dev`. Only amdgpu clients open on that character
// device are counted. Compute partitions share one PCI BDF in fdinfo
// `drm-pdev` but have their own render node, so they stay separate. Matching
// the device number rather than the link name also holds when a container
// exposes the node under a different name. Needs no host PID. KFD-only fields
// are set to their unavailable sentinels. A null `read_fdinfo` reads
// /proc/self/fdinfo.
amdsmi_status_t gpuvsmi_get_self_info(dev_t render_dev, amdsmi_proc_info_t& info,
                                      SelfFdinfoReader read_fdinfo = nullptr);

#endif
