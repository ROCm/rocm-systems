/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 */

#ifndef HSAKMT_DRM_VERSION_H
#define HSAKMT_DRM_VERSION_H

#include <stdbool.h>
#include <stdint.h>

static inline bool hsakmt_drm_supports_vm_timeline(uint32_t major,
                                                   uint32_t minor)
{
	return major > 3 || (major == 3 && minor >= 64);
}

#endif /* HSAKMT_DRM_VERSION_H */
