/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * amdgpu_pmu_uapi.h - Userspace API for AMDGPU PMU trace events
 *
 * This header defines the ioctl interface for rocprofv3 to emit
 * tracepoint events through the perf-dkms kernel module.
 */

#ifndef _AMDGPU_PMU_UAPI_H
#define _AMDGPU_PMU_UAPI_H

/* Support both kernel and userspace compilation */
#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/ioctl.h>
#else
/* Userspace: use linux/types.h for __u64, __u32, __u16 */
#include <linux/types.h>
#include <sys/ioctl.h>
#endif

#define AMDGPU_PMU_TRACE_MAGIC 'G'
#define AMDGPU_PMU_TRACE_VERSION 1

/* Maximum batch size to prevent DoS */
#define AMDGPU_PMU_MAX_BATCH_SIZE 4096

/* Event types */
#define AMDGPU_PMU_EVENT_KERNEL_DISPATCH  0
#define AMDGPU_PMU_EVENT_HSA_API          1
#define AMDGPU_PMU_EVENT_HIP_API          2

/* Kernel dispatch event data */
struct amdgpu_pmu_kernel_dispatch {
	__u64 dispatch_id;
	__u64 correlation_id;
	__u64 kernel_id;
	__u64 start_ts;
	__u64 end_ts;
	__u32 agent_id;
	__u32 queue_id;
	__u32 grid_x;
	__u32 grid_y;
	__u32 grid_z;
	__u32 workgroup_size;  /* Packed: wg_x | (wg_y << 10) | (wg_z << 20) */
};

/* Maximum length for API operation name string */
#define AMDGPU_PMU_API_NAME_MAX 64

/* HSA/HIP API event data */
struct amdgpu_pmu_api_event {
	__u32 kind;
	__u32 operation;
	__u64 correlation_id;
	__u64 start_ts;
	__u64 end_ts;
	__u32 thread_id;
	__u32 _pad;
	char operation_name[AMDGPU_PMU_API_NAME_MAX];
};

/* Batch emit structure for efficiency */
struct amdgpu_pmu_emit_events {
	__u32 event_type;       /* AMDGPU_PMU_EVENT_* */
	__u32 count;            /* Number of events */
	__u64 events_ptr;       /* Pointer to array of events */
};

/* ioctl commands */
#define AMDGPU_PMU_IOCTL_EMIT_KERNEL_DISPATCH \
	_IOW(AMDGPU_PMU_TRACE_MAGIC, 1, struct amdgpu_pmu_kernel_dispatch)

#define AMDGPU_PMU_IOCTL_EMIT_HSA_API \
	_IOW(AMDGPU_PMU_TRACE_MAGIC, 2, struct amdgpu_pmu_api_event)

#define AMDGPU_PMU_IOCTL_EMIT_HIP_API \
	_IOW(AMDGPU_PMU_TRACE_MAGIC, 3, struct amdgpu_pmu_api_event)

#define AMDGPU_PMU_IOCTL_EMIT_BATCH \
	_IOW(AMDGPU_PMU_TRACE_MAGIC, 4, struct amdgpu_pmu_emit_events)

/* Query if tracepoints are enabled (for optimization) */
#define AMDGPU_PMU_IOCTL_QUERY_ENABLED \
	_IOR(AMDGPU_PMU_TRACE_MAGIC, 5, __u32)

/* Get interface version for compatibility checking */
#define AMDGPU_PMU_IOCTL_GET_VERSION \
	_IOR(AMDGPU_PMU_TRACE_MAGIC, 6, __u32)

#endif /* _AMDGPU_PMU_UAPI_H */
