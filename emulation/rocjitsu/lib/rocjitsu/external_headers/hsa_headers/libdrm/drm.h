// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file drm.h
/// @brief Shadows the system `<libdrm/drm.h>` with the vendored kernel DRM
///        UAPI header.
///
/// `linux/uapi/kfd_ioctl.h` is a verbatim copy of the rocr-runtime header, and
/// it includes `<libdrm/drm.h>`. Left alone that resolves under `/usr/include`
/// to whatever libdrm the build host installs -- and since the host header and
/// the vendored `drm_headers/linux/uapi/drm/drm.h` both guard on `_DRM_H_`,
/// whichever is included first wins and the other expands to nothing. Every
/// translation unit here reaches `kfd_ioctl.h` through some rocjitsu header
/// long before it reaches the vendored copy, so the host's libdrm always won
/// and "independent of libdrm" held only by luck: the interposer compiled on a
/// modern host and failed inside the manylinux builder image, whose older
/// libdrm has no `DRM_SYNCOBJ_WAIT_FLAGS_WAIT_DEADLINE`.
///
/// This file sits on `HSA_INCLUDE_DIR` -- the same include directory that makes
/// `kfd_ioctl.h` findable at all -- so it is on the search path of exactly the
/// translation units that need it, and any `-I`/`-isystem` directory is
/// searched ahead of `/usr/include`.
///
/// `<libdrm/amdgpu_drm.h>` is deliberately NOT shadowed: `drm_info_layout_test`
/// includes it to check the interposer's structs against the system layout,
/// which is only a check while it comes from the system.

#include "../../drm_headers/linux/uapi/drm/drm.h"
