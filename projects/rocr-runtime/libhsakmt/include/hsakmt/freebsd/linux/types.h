/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * FreeBSD stand-in for <linux/types.h>, needed by include/hsakmt/linux/udmabuf.h.
 * The vendored drm.h already defines the Linux __u8 ... __u64 and __s8 ... __s64
 * types on the BSDs, so use it rather than defining them a second time.
 */
#ifndef HSAKMT_FREEBSD_LINUX_TYPES_H
#define HSAKMT_FREEBSD_LINUX_TYPES_H

#include "hsakmt/drm/drm.h"

#endif /* HSAKMT_FREEBSD_LINUX_TYPES_H */
