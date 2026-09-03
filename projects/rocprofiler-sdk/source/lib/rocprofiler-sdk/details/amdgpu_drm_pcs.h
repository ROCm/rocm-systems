// Draft KGD PC sampling uAPI — sync ioctl number/struct with the kernel team before upstream.
//
// Today we reuse the KFD PCS argument layout because the unified KGD/KFD plan specifies
// a similar render-node ioctl. Replace this file when amdgpu_drm.h publishes the final ABI.

#pragma once

#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"

// TODO(kernel): confirm the final DRM command index with the driver team.
// Internal builds may temporarily mirror the KFD PCS ioctl encoding on the render node.
#ifndef DRM_AMDGPU_IOCTL_PC_SAMPLE
#    define DRM_AMDGPU_IOCTL_PC_SAMPLE AMDKFD_IOC_PC_SAMPLE
#endif

// PCS payload matches KFD for now.
using drm_amdgpu_pc_sample_args = kfd_ioctl_pc_sample_args;
