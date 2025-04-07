/*
 * Copyright © 2020 Advanced Micro Devices, Inc.
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


HSAKMT_STATUS HSAKMTAPI hsaKmtSPMAcquire(HSAuint32 PreferredNode)
{
	int ret;
	struct kfd_ioctl_spm_args args = {0};
	uint32_t gpu_id;

	ret = hsakmt_validate_nodeid(PreferredNode, &gpu_id);
	if (ret != HSAKMT_STATUS_SUCCESS) {
		pr_err("[%s] invalid node ID: %d\n", __func__, PreferredNode);
		return ret;
	}

	ret = HSAKMT_STATUS_SUCCESS;
	args.op = KFD_IOCTL_SPM_OP_ACQUIRE;
	args.gpu_id = gpu_id;

	ret = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_RLC_SPM, &args);

	return ret;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSPMSetDestBuffer(HSAuint32 PreferredNode,
						HSAuint32 SizeInBytes,
						HSAuint32 * timeout,
						HSAuint32 * SizeCopied,
						void *DestMemoryAddress,
						bool *isSPMDataLoss)
{
	int ret;
	struct kfd_ioctl_spm_args args = {0};
	uint32_t gpu_id = 0;

	ret = hsakmt_validate_nodeid(PreferredNode, &gpu_id);
	if (ret != HSAKMT_STATUS_SUCCESS) {
		return ret;
	}

	args.timeout    = *timeout;
	args.dest_buf    = (uint64_t)DestMemoryAddress;
	args.buf_size   = SizeInBytes;
	args.op         = KFD_IOCTL_SPM_OP_SET_DEST_BUF;
	args.gpu_id     = gpu_id;

	ret = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_RLC_SPM, &args);

	*SizeCopied = args.bytes_copied;
	*isSPMDataLoss = args.has_data_loss;
	*timeout = args.timeout;

	return ret;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSPMRelease(HSAuint32 PreferredNode)
{
	int ret = HSAKMT_STATUS_SUCCESS;
	struct kfd_ioctl_spm_args args = {0};
	uint32_t gpu_id;

	ret = hsakmt_validate_nodeid(PreferredNode, &gpu_id);
	if (ret != HSAKMT_STATUS_SUCCESS) {
		pr_err("[%s] invalid node ID: %d\n", __func__, PreferredNode);
		return ret;
	}

	args.op = KFD_IOCTL_SPM_OP_RELEASE;
	args.gpu_id = gpu_id;

	ret = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_RLC_SPM, &args);

	return ret;
}


