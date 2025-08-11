/*
 * Copyright © 2023 Advanced Micro Devices, Inc.
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
#include "hsakmt/linux/kfd_ioctl.h"
#include <stdlib.h>
#include <stdio.h>
#include "fmm.h"


HSAKMT_STATUS HSAKMTAPI hsaKmtAisReadWriteFile(void *MemoryAddress,
					      HSAuint64 MemorySizeInBytes,
					      HSAuint32 fd,
					      HSAuint64 file_offset,
					      HsaAisFlags AisFlags)
{
	CHECK_KFD_OPEN();

	struct kfd_ioctl_ais_args args = {0};
	uint64_t handle, size_offset = MemorySizeInBytes;

	/* Support is only for dGPUs */


	if (!hsakmt_fmm_get_handle(MemoryAddress, &handle, &size_offset))
		return HSAKMT_STATUS_INVALID_PARAMETER;

	args.handle = handle;
	args.fd = fd;
	args.file_offset = file_offset;
	args.size = MemorySizeInBytes;
	if (AisFlags == HSA_AIS_WRITE)
		args.op = KFD_IOC_AIS_WRITE;
	else if (AisFlags == HSA_AIS_READ)
		args.op = KFD_IOC_AIS_READ;
	else
	{
		pr_err("[%s] invalid AisFlags: %d\n", __func__, AisFlags);
		return HSAKMT_STATUS_INVALID_PARAMETER;
	}

	args.handle_offset = size_offset;

	if (hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_AIS_OP, &args) != 0)
		return HSAKMT_STATUS_ERROR;

	return HSAKMT_STATUS_SUCCESS;
}
