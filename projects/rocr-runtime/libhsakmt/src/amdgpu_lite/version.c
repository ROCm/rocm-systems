/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "libhsakmt.h"
#include <string.h>

/* Emulate KFD version 1.14 (latest KFD interface version) */
#define AMDGPU_LITE_KFD_MAJOR  1
#define AMDGPU_LITE_KFD_MINOR  14

HsaVersionInfo hsakmt_kfd_version_info;

HSAKMT_STATUS HSAKMTAPI hsaKmtGetVersion(HsaVersionInfo *VersionInfo)
{
	CHECK_KFD_OPEN();

	if (!VersionInfo)
		return HSAKMT_STATUS_INVALID_PARAMETER;

	*VersionInfo = hsakmt_kfd_version_info;
	return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS hsakmt_init_kfd_version(void)
{
	/*
	 * amdgpu_lite does not use the KFD ioctl interface, so we
	 * synthesize the version info to report KFD 1.14 compatibility.
	 */
	hsakmt_kfd_version_info.KernelInterfaceMajorVersion =
		AMDGPU_LITE_KFD_MAJOR;
	hsakmt_kfd_version_info.KernelInterfaceMinorVersion =
		AMDGPU_LITE_KFD_MINOR;

	return HSAKMT_STATUS_SUCCESS;
}
