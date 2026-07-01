/*
 * Copyright © 2025 Advanced Micro Devices, Inc.
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
#include <amdgpu_drm.h>

int hsakmt_open_drm_render_device(HsaKFDContext *ctx, int minor);
static HSAKMT_STATUS get_clock_counters_kfd(HsaKFDContext *ctx,
                                            HSAuint32 NodeId,
                                            HsaClockCounters *Counters) {
    struct kfd_ioctl_get_clock_counters_args args = {0};
    int err;
    uint32_t gpu_id;
    HSAKMT_STATUS result;

    CHECK_KFD_OPEN();

    result = hsakmt_validate_nodeid(ctx, NodeId, &gpu_id);

    if (result != HSAKMT_STATUS_SUCCESS) {
        return result;
    }

    args.gpu_id = gpu_id;

    err = hsakmt_ioctl(ctx->fd, AMDKFD_IOC_GET_CLOCK_COUNTERS, &args);
    if (err < 0) {
        return HSAKMT_STATUS_ERROR;
    } else {
        Counters->GPUClockCounter = args.gpu_clock_counter;
        Counters->CPUClockCounter = args.cpu_clock_counter;
        Counters->SystemClockCounter = args.system_clock_counter;
        Counters->SystemClockFrequencyHz = args.system_clock_freq;
        return HSAKMT_STATUS_SUCCESS;
    }
}

static HSAKMT_STATUS get_clock_counters_drm(HsaKFDContext *ctx,
                                HSAuint32 NodeId, HsaClockCounters *Counters) {
    struct drm_amdgpu_info args = {0};
    struct drm_amdgpu_info_clock_counters clock_counters;
    int fd;
    int ret;
    HsaNodeProperties props;

    ret = hsakmt_topology_get_node_props(ctx, NodeId, &props);
    if (ret != HSAKMT_STATUS_SUCCESS) {
        return ret;
    }

    /* Skip non-GPU nodes */
    if (!props.KFDGpuID) {
        return HSAKMT_STATUS_INVALID_NODE_UNIT;
    }

    fd = hsakmt_open_drm_render_device(ctx, props.DrmRenderMinor);
    if (fd <= 0) {
        return HSAKMT_STATUS_ERROR;
    }

    args.query = AMDGPU_INFO_CLOCK_COUNTERS;
    args.return_pointer = (uintptr_t)&clock_counters;
    args.return_size = sizeof(clock_counters);

    ret = hsakmt_ioctl(fd, DRM_IOCTL_AMDGPU_INFO, &args);
    if (ret) {
        return HSAKMT_STATUS_ERROR;
    }

    Counters->GPUClockCounter = clock_counters.gpu_clock_counter;
    Counters->CPUClockCounter = clock_counters.cpu_clock_counter;
    Counters->SystemClockCounter = clock_counters.system_clock_counter;
    Counters->SystemClockFrequencyHz = clock_counters.system_clock_freq;

    return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetClockCountersCtx(HsaKFDContext *ctx,
					                            HSAuint32 NodeId,
					                            HsaClockCounters *Counters) {
    HSAKMT_STATUS result;

    if (hsakmt_enable_drm) {
        result = get_clock_counters_drm(ctx, NodeId, Counters);

        if (result == HSAKMT_STATUS_SUCCESS) {
            return result;
        }
    }
    result = get_clock_counters_kfd(ctx, NodeId, Counters);

    return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetClockCounters(HSAuint32 NodeId,
					       HsaClockCounters *Counters)
{
	return hsaKmtGetClockCountersCtx(&hsakmt_primary_kfd_ctx, NodeId, Counters);
}
