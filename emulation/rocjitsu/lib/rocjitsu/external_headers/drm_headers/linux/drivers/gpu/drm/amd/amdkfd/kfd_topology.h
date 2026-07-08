/* SPDX-License-Identifier: GPL-2.0 OR MIT */
/*
 * Copyright 2014-2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/*
 * Vendored excerpt of drivers/gpu/drm/amd/amdkfd/kfd_topology.h from the Linux
 * kernel (v7.1, commit e771677c937d). Only the debug_prop address-watch-mask
 * definitions consumed by the synthetic KFD topology generator are copied here;
 * the driver-internal structs, prototypes and includes are intentionally
 * omitted. The macro *values* are verbatim from upstream; whitespace follows
 * this repository's clang-format convention (as does the sibling UAPI header
 * linux/uapi/kfd_sysfs.h), so the multi-line upstream macros are joined onto a
 * single line.
 *
 * These per-GFXIP values are NOT part of the kernel UAPI -- they live in a
 * private amdkfd header -- but they are exactly what the driver writes into the
 * per-node "debug_prop" topology property that libhsakmt and rocdbgapi read
 * back (see kfd_topology_set_capabilities() in kfd_topology.c). They build on
 * the HSA_DBG_WATCH_ADDR_MASK_*_SHIFT macros from the UAPI header
 * linux/uapi/kfd_sysfs.h, which is included below so this excerpt is
 * self-contained.
 */

#ifndef ROCJITSU_VENDOR_AMDKFD_KFD_TOPOLOGY_H_
#define ROCJITSU_VENDOR_AMDKFD_KFD_TOPOLOGY_H_

#include "linux/uapi/kfd_sysfs.h"

#define HSA_DBG_WATCH_ADDR_MASK_LO_BIT_GFX9 6
#define HSA_DBG_WATCH_ADDR_MASK_LO_BIT_GFX9_4_3 7
#define HSA_DBG_WATCH_ADDR_MASK_LO_BIT_GFX10 7
#define HSA_DBG_WATCH_ADDR_MASK_HI_BIT (29 << HSA_DBG_WATCH_ADDR_MASK_HI_BIT_SHIFT)
#define HSA_DBG_WATCH_ADDR_MASK_HI_BIT_GFX9_4_3 (30 << HSA_DBG_WATCH_ADDR_MASK_HI_BIT_SHIFT)

#endif /* ROCJITSU_VENDOR_AMDKFD_KFD_TOPOLOGY_H_ */
