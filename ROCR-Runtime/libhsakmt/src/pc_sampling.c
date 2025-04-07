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
#include <assert.h>
#include <errno.h>

#define INVALID_TRACE_ID 0x0

HSAKMT_STATUS HSAKMTAPI hsaKmtPcSamplingSupport(void)
{
    CHECK_KFD_OPEN();
    CHECK_KFD_MINOR_VERSION(16);

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPcSamplingQueryCapabilities(HSAuint32 NodeId, void *sample_info,
                            HSAuint32 sample_info_sz, HSAuint32 *size)
{
    struct kfd_ioctl_pc_sample_args args = {0};
    uint32_t gpu_id;

    if (size == NULL)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    CHECK_KFD_OPEN();
    CHECK_KFD_MINOR_VERSION(16);

    HSAKMT_STATUS ret = hsakmt_validate_nodeid(NodeId, &gpu_id);
    if (ret != HSAKMT_STATUS_SUCCESS) {
        pr_err("[%s] invalid node ID: %d\n", __func__, NodeId);
        return ret;
    }
    assert(sizeof(HsaPcSamplingInfo) == sizeof(struct kfd_pc_sample_info));

    args.op = KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES;
    args.gpu_id = gpu_id;
    args.sample_info_ptr = (uint64_t)sample_info;
    args.num_sample_info = sample_info_sz;
    args.flags = 0;

    int err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_PC_SAMPLE, &args);

    *size = args.num_sample_info;

    if (err) {
        switch (errno) {
        case ENOSPC:
                return HSAKMT_STATUS_BUFFER_TOO_SMALL;
        case EINVAL:
                return HSAKMT_STATUS_INVALID_PARAMETER;
        case EOPNOTSUPP:
                return HSAKMT_STATUS_NOT_SUPPORTED;
        case EBUSY:
                return HSAKMT_STATUS_UNAVAILABLE;
        default:
                return HSAKMT_STATUS_ERROR;
        }
    }

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPcSamplingCreate(HSAuint32 NodeId, HsaPcSamplingInfo *sample_info,
						HsaPcSamplingTraceId *traceId)
{
    struct kfd_ioctl_pc_sample_args args = {0};
    uint32_t gpu_id;

    if (sample_info == NULL || traceId == NULL)
        return HSAKMT_STATUS_INVALID_PARAMETER;

    CHECK_KFD_OPEN();

    *traceId = INVALID_TRACE_ID;
    HSAKMT_STATUS ret = hsakmt_validate_nodeid(NodeId, &gpu_id);
    if (ret != HSAKMT_STATUS_SUCCESS) {
        pr_err("[%s] invalid node ID: %d\n", __func__, NodeId);
        return ret;
    }

    args.op = KFD_IOCTL_PCS_OP_CREATE;
    args.gpu_id = gpu_id;
    args.sample_info_ptr = (uint64_t)sample_info;
    args.num_sample_info = 1;
    args.trace_id = INVALID_TRACE_ID;

    int err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_PC_SAMPLE, &args);
    if (err) {
        switch (errno) {
        case EINVAL:
            return HSAKMT_STATUS_INVALID_PARAMETER;
        case ENOMEM:
            return HSAKMT_STATUS_NO_MEMORY;
        case EBUSY:
            return HSAKMT_STATUS_UNAVAILABLE;
        default:
            return HSAKMT_STATUS_ERROR;
        }
    }

    *traceId = args.trace_id;
    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPcSamplingDestroy(HSAuint32 NodeId, HsaPcSamplingTraceId traceId)
{
    struct kfd_ioctl_pc_sample_args args = {0};
    uint32_t gpu_id;

    if (traceId == INVALID_TRACE_ID)
        return HSAKMT_STATUS_INVALID_HANDLE;

    CHECK_KFD_OPEN();

    HSAKMT_STATUS ret = hsakmt_validate_nodeid(NodeId, &gpu_id);
    if (ret != HSAKMT_STATUS_SUCCESS) {
        pr_err("[%s] invalid node ID: %d\n", __func__, NodeId);
        return ret;
    }

    hsaKmtPcSamplingStop(NodeId, traceId);

    args.op = KFD_IOCTL_PCS_OP_DESTROY;
    args.gpu_id = gpu_id;
    args.trace_id = traceId;

    int err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_PC_SAMPLE, &args);
    if (err) {
        if (errno == EINVAL)
            return HSAKMT_STATUS_INVALID_PARAMETER;
        return HSAKMT_STATUS_ERROR;
    }

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPcSamplingStart(HSAuint32 NodeId, HsaPcSamplingTraceId traceId)
{
    struct kfd_ioctl_pc_sample_args args = {0};
    uint32_t gpu_id;

    if (traceId == INVALID_TRACE_ID)
        return HSAKMT_STATUS_INVALID_HANDLE;

    CHECK_KFD_OPEN();

    HSAKMT_STATUS ret = hsakmt_validate_nodeid(NodeId, &gpu_id);
    if (ret != HSAKMT_STATUS_SUCCESS) {
        pr_err("[%s] invalid node ID: %d\n", __func__, NodeId);
        return ret;
    }

    args.op = KFD_IOCTL_PCS_OP_START;
    args.gpu_id = gpu_id;
    args.trace_id = traceId;

    int err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_PC_SAMPLE, &args);
    if (err) {
        switch (errno) {
        case EINVAL:
            return HSAKMT_STATUS_INVALID_PARAMETER;
        case ENOMEM:
            return HSAKMT_STATUS_OUT_OF_RESOURCES;
        case EBUSY:
            return HSAKMT_STATUS_UNAVAILABLE;
        case EALREADY:
            return HSAKMT_STATUS_KERNEL_ALREADY_OPENED;
        default:
            return HSAKMT_STATUS_ERROR;
        }
    }

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtPcSamplingStop(HSAuint32 NodeId, HsaPcSamplingTraceId traceId)
{
    struct kfd_ioctl_pc_sample_args args = {0};
    uint32_t gpu_id;

    if (traceId == INVALID_TRACE_ID)
        return HSAKMT_STATUS_INVALID_HANDLE;

    CHECK_KFD_OPEN();

    HSAKMT_STATUS ret = hsakmt_validate_nodeid(NodeId, &gpu_id);
    if (ret != HSAKMT_STATUS_SUCCESS) {
        pr_err("[%s] invalid node ID: %d\n", __func__, NodeId);
        return ret;
    }

    args.op = KFD_IOCTL_PCS_OP_STOP;
    args.gpu_id = gpu_id;
    args.trace_id = traceId;

    int err = hsakmt_ioctl(hsakmt_kfd_fd, AMDKFD_IOC_PC_SAMPLE, &args);
    if (err) {
        switch (errno) {
        case EINVAL:
            return HSAKMT_STATUS_INVALID_PARAMETER;
        case EALREADY:
            return HSAKMT_STATUS_KERNEL_ALREADY_OPENED;
        default:
            return HSAKMT_STATUS_ERROR;
        }
    }
    return HSAKMT_STATUS_SUCCESS;
}
