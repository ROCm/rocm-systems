// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Collector behind amdsmi_get_gpu_process_info_self(). Build-only: it declares
// test hooks, so the dev package does not install it.

#ifndef AMD_SMI_INCLUDE_AMD_SMI_SELF_FDINFO_H_
#define AMD_SMI_INCLUDE_AMD_SMI_SELF_FDINFO_H_

#include <sys/types.h>

#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"

// Reads the fdinfo text of the calling process's descriptor `fd` into `text`.
// Returns AMDSMI_STATUS_NOT_FOUND if the descriptor is gone.
using SelfFdinfoReader = amdsmi_status_t (*)(int fd, std::string* text);

namespace amd::smi {
// Reads all of `fdinfo_fd`, an open /proc/self/fdinfo/<fd> file, into `text`.
// Returns AMDSMI_STATUS_NOT_FOUND if <fd> was closed after the file was opened.
amdsmi_status_t ReadFdinfoFile(int fdinfo_fd, std::string* text);
}  // namespace amd::smi

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

#endif  // AMD_SMI_INCLUDE_AMD_SMI_SELF_FDINFO_H_
