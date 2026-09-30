/*
 * amdgpu_pmu_trace_dev.c - Userspace interface for trace events
 *
 * Provides a miscdevice (/dev/amdgpu_pmu_trace) that allows rocprofv3
 * to emit kernel tracepoints via ioctl.
 */

#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>

#include "amdgpu_pmu_uapi.h"
#include "amdgpu_pmu_trace.h"

/* Forward declaration */
static long amdgpu_pmu_emit_batch(struct amdgpu_pmu_emit_events *batch);

static long amdgpu_pmu_trace_ioctl(struct file *file, unsigned int cmd,
				    unsigned long arg)
{
	void __user *argp = (void __user *)arg;

	/*
	 * Optional trace ingress path:
	 * userspace profiler tools push pre-correlated events into kernel
	 * tracepoints so counter data and timeline events can be consumed in a
	 * single tracing pipeline. This path is independent from PMU counting.
	 */
	switch (cmd) {
	case AMDGPU_PMU_IOCTL_EMIT_KERNEL_DISPATCH: {
		struct amdgpu_pmu_kernel_dispatch ev;
		if (copy_from_user(&ev, argp, sizeof(ev)))
			return -EFAULT;

		trace_kernel_dispatch(ev.dispatch_id, ev.correlation_id,
				      ev.kernel_id, ev.start_ts, ev.end_ts,
				      ev.agent_id, ev.queue_id,
				      ev.grid_x, ev.grid_y, ev.grid_z,
				      ev.workgroup_size);
		return 0;
	}

	case AMDGPU_PMU_IOCTL_EMIT_HSA_API: {
		struct amdgpu_pmu_api_event ev;
		if (copy_from_user(&ev, argp, sizeof(ev)))
			return -EFAULT;

		trace_hsa_api(ev.kind, ev.operation, ev.correlation_id,
			      ev.start_ts, ev.end_ts, ev.thread_id, ev.operation_name);
		return 0;
	}

	case AMDGPU_PMU_IOCTL_EMIT_HIP_API: {
		struct amdgpu_pmu_api_event ev;
		if (copy_from_user(&ev, argp, sizeof(ev)))
			return -EFAULT;

		trace_hip_api(ev.kind, ev.operation, ev.correlation_id,
			      ev.start_ts, ev.end_ts, ev.thread_id, ev.operation_name);
		return 0;
	}

	case AMDGPU_PMU_IOCTL_EMIT_BATCH: {
		struct amdgpu_pmu_emit_events batch;
		if (copy_from_user(&batch, argp, sizeof(batch)))
			return -EFAULT;

		return amdgpu_pmu_emit_batch(&batch);
	}

	case AMDGPU_PMU_IOCTL_QUERY_ENABLED: {
		u32 enabled = 0;
		if (trace_kernel_dispatch_enabled())
			enabled |= (1 << AMDGPU_PMU_EVENT_KERNEL_DISPATCH);
		if (trace_hsa_api_enabled())
			enabled |= (1 << AMDGPU_PMU_EVENT_HSA_API);
		if (trace_hip_api_enabled())
			enabled |= (1 << AMDGPU_PMU_EVENT_HIP_API);

		if (copy_to_user(argp, &enabled, sizeof(enabled)))
			return -EFAULT;
		return 0;
	}

	case AMDGPU_PMU_IOCTL_GET_VERSION: {
		u32 version = AMDGPU_PMU_TRACE_VERSION;
		if (copy_to_user(argp, &version, sizeof(version)))
			return -EFAULT;
		return 0;
	}

	default:
		return -ENOTTY;
	}
}

/* Batch emit for efficiency */
static long amdgpu_pmu_emit_batch(struct amdgpu_pmu_emit_events *batch)
{
	u32 i;

	/* Prevent DoS via unbounded batch size */
	if (batch->count > AMDGPU_PMU_MAX_BATCH_SIZE)
		return -EINVAL;

	switch (batch->event_type) {
	case AMDGPU_PMU_EVENT_KERNEL_DISPATCH: {
		struct amdgpu_pmu_kernel_dispatch __user *user_events =
			(struct amdgpu_pmu_kernel_dispatch __user *)batch->events_ptr;
		struct amdgpu_pmu_kernel_dispatch ev;

		for (i = 0; i < batch->count; i++) {
			if (copy_from_user(&ev, &user_events[i], sizeof(ev)))
				return -EFAULT;
			trace_kernel_dispatch(ev.dispatch_id, ev.correlation_id,
					      ev.kernel_id, ev.start_ts, ev.end_ts,
					      ev.agent_id, ev.queue_id,
					      ev.grid_x, ev.grid_y, ev.grid_z,
					      ev.workgroup_size);
		}
		return 0;
	}

	case AMDGPU_PMU_EVENT_HSA_API: {
		struct amdgpu_pmu_api_event __user *user_events =
			(struct amdgpu_pmu_api_event __user *)batch->events_ptr;
		struct amdgpu_pmu_api_event ev;

		for (i = 0; i < batch->count; i++) {
			if (copy_from_user(&ev, &user_events[i], sizeof(ev)))
				return -EFAULT;
			trace_hsa_api(ev.kind, ev.operation, ev.correlation_id,
				      ev.start_ts, ev.end_ts, ev.thread_id, ev.operation_name);
		}
		return 0;
	}

	case AMDGPU_PMU_EVENT_HIP_API: {
		struct amdgpu_pmu_api_event __user *user_events =
			(struct amdgpu_pmu_api_event __user *)batch->events_ptr;
		struct amdgpu_pmu_api_event ev;

		for (i = 0; i < batch->count; i++) {
			if (copy_from_user(&ev, &user_events[i], sizeof(ev)))
				return -EFAULT;
			trace_hip_api(ev.kind, ev.operation, ev.correlation_id,
				      ev.start_ts, ev.end_ts, ev.thread_id, ev.operation_name);
		}
		return 0;
	}

	default:
		return -EINVAL;
	}
}

static int amdgpu_pmu_trace_open(struct inode *inode, struct file *file)
{
	return 0;
}

static int amdgpu_pmu_trace_release(struct inode *inode, struct file *file)
{
	return 0;
}

static const struct file_operations amdgpu_pmu_trace_fops = {
	.owner          = THIS_MODULE,
	.open           = amdgpu_pmu_trace_open,
	.release        = amdgpu_pmu_trace_release,
	.unlocked_ioctl = amdgpu_pmu_trace_ioctl,
	.compat_ioctl   = amdgpu_pmu_trace_ioctl,
};

static struct miscdevice amdgpu_pmu_trace_dev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name  = "amdgpu_pmu_trace",
	.fops  = &amdgpu_pmu_trace_fops,
	.mode  = 0666,  /* Allow non-root access */
};

int amdgpu_pmu_trace_dev_init(void)
{
	/* Best-effort auxiliary interface: PMU remains functional if this fails. */
	return misc_register(&amdgpu_pmu_trace_dev);
}

void amdgpu_pmu_trace_dev_exit(void)
{
	misc_deregister(&amdgpu_pmu_trace_dev);
}
