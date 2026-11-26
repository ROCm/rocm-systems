/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_AMD_DMEM_CGROUP_H_
#define HSA_RUNTIME_CORE_INC_AMD_DMEM_CGROUP_H_

#include "hsakmt/hsakmt.h"

namespace rocr {
namespace AMD {
namespace dmem_cgroup {

/// @brief The VRAM cap the kernel's dmem cgroup controller imposes on the calling process for
/// one GPU.
struct VramLimit {
  HSAuint64 max;        // Tightest dmem.max between the process's cgroup and the hierarchy root
  HSAuint64 available;  // Least dmem.max minus dmem.current at any of those levels
};

/// @brief Looks up the dmem cgroup VRAM cap applying to the calling process for the given GPU.
/// Inside a cgroup namespace only the levels within the namespace are visible, so a cap set above
/// it is not observed.
/// @param[in] drm_render_minor Render node minor of the GPU, from HsaNodeProperties::DrmRenderMinor
/// @param[out] limit The cap, written only when a finite cap applies
/// @return true if a finite cap applies, false if none does or the cgroup files cannot be read.
bool QueryVramLimit(HSAuint32 drm_render_minor, VramLimit* limit);

}  // namespace dmem_cgroup
}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_DMEM_CGROUP_H_
