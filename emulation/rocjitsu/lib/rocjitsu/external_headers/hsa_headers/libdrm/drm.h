// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file drm.h
/// @brief Shadows the system `<libdrm/drm.h>` with the vendored kernel DRM
///        UAPI header.
///
/// `linux/uapi/kfd_ioctl.h` is a local copy of the rocr-runtime header, and it
/// includes `<libdrm/drm.h>`. That include is itself the local edit: upstream
/// (`projects/rocr-runtime/libhsakmt/include/hsakmt/linux/kfd_ioctl.h`) reads
/// `#include "hsakmt/drm/drm.h"` and reaches rocr's own vendored UAPI, and the
/// copy here has also been reformatted and codespell-corrected in place. So
/// "do not touch it, it is verbatim" is not the reason this shadow exists --
/// the reason is that pointing the include at
/// `../../../drm_headers/linux/uapi/drm/drm.h`, or putting `DRM_INCLUDE_DIR`
/// next to `HSA_INCLUDE_DIR` in `rj_configure_target`/`rj_add_object_library`,
/// are both larger changes to make than this file. If you are already editing
/// the include path, prefer one of those: they are visible from the build.
///
/// Left alone, `<libdrm/drm.h>` resolves under `/usr/include` to whatever
/// libdrm the build host installs -- and since the host header and the
/// vendored `drm_headers/linux/uapi/drm/drm.h` both guard on `_DRM_H_`,
/// whichever is included first wins and the other expands to nothing. Every
/// translation unit here reaches `kfd_ioctl.h` through some rocjitsu header
/// long before it reaches the vendored copy, so the host's libdrm always won
/// and "independent of libdrm" held only by luck: the interposer compiled on a
/// modern host and failed inside the manylinux builder image, whose older
/// libdrm has no `DRM_SYNCOBJ_WAIT_FLAGS_WAIT_DEADLINE`.
///
/// Note the reach this creates: `HSA_INCLUDE_DIR` is no longer self-contained,
/// because this file depends on the sibling `drm_headers/` tree sitting at a
/// fixed relative offset. Moving or installing one without the other breaks
/// every translation unit that includes `kfd_ioctl.h`.
///
/// Note also what comes with it: the vendored `drm.h` ends with
/// `#include "drm_mode.h"`, so the vendored `drm_mode.h` (same `_DRM_MODE_H`
/// guard as the host's) is pulled in too. `<libdrm/drm_mode.h>` and
/// `<libdrm/drm_fourcc.h>` are therefore effectively shadowed as well.
///
/// This file sits on `HSA_INCLUDE_DIR` -- the same include directory that makes
/// `kfd_ioctl.h` findable at all -- so it is on the search path of exactly the
/// translation units that need it, and any `-I`/`-isystem` directory is
/// searched ahead of `/usr/include`. That also means the shadow is invisible
/// from the build files: anything that drops, reorders or narrows
/// `HSA_INCLUDE_DIR` for a target silently hands it the host's libdrm again.
///
/// `<libdrm/amdgpu_drm.h>` is deliberately NOT shadowed: `drm_info_layout_test`
/// includes it to check the interposer's structs against the system layout,
/// which is only a check while it comes from the system. That test guards its
/// own end of the bargain with an `#ifdef _DRM_H_` sentinel.

#pragma once

#include "../../drm_headers/linux/uapi/drm/drm.h"

// The guard collision is narrowed by this file, not closed: a translation unit
// that pulls in a system libdrm header before it reaches `kfd_ioctl.h` defines
// `_DRM_H_` first and makes the include above expand to nothing. Say so here,
// where the cause is, rather than letting it surface as an undeclared
// identifier hundreds of lines into the interposer.
//
// Read this for exactly what it is -- a better error message, NOT an invariant
// check. It can only fire where the host's libdrm predates the flag, which is
// the one place the build already failed loudly. On a modern host a system
// `drm.h` that wins the race defines the flag itself and this stays silent,
// leaving that translation unit built against host structs while its
// neighbours use the vendored ones. A guard on `_DRM_H_` alone would catch
// that on every host, but it cannot be used here: `interposer_dup_test.cpp`
// legitimately includes the vendored `linux/uapi/drm/drm.h` (via
// DRM_INCLUDE_DIR) before `kfd_ioctl.h`, and nothing distinguishes a
// pre-defined `_DRM_H_` that came from the vendored copy from one that came
// from the host. Until something does, the rule is grep-enforced: no rocjitsu
// translation unit includes a system `<libdrm/...>` header except
// `drm_info_layout_test.cpp`, which guards itself with `#ifdef _DRM_H_`.
#ifndef DRM_SYNCOBJ_WAIT_FLAGS_WAIT_DEADLINE
#error "<libdrm/drm.h> did not resolve to the vendored UAPI: a system libdrm \
header was included first and its _DRM_H_ guard made this shadow inert. \
Include the vendored drm.h (DRM_INCLUDE_DIR) before any <libdrm/...> header."
#endif
