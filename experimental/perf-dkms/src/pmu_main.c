/*
 * pmu_main.c - Main PMU Stub Driver Implementation
 *
 * This module implements a PMU driver for the Linux perf subsystem that
 * exposes GFX12 hardware performance counters through AQL integration.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/perf_event.h>
#include <linux/hrtimer.h>
#include <linux/slab.h>
#include <linux/cpu.h>
#include <linux/bitmap.h>
#include <linux/atomic.h>
#include <linux/device.h>

#include "amdgpu_pmu.h"
#include "aql_perf.h"
#include "aql_c/counter_registry.h"
#include "pmu_dimension.h"
#include "amdgpu_pmu_trace.h"
#include "amdgpu_pmu_trace_dev.h"

/* Global PMU instance */
static struct amdgpu_pmu *amdgpu_pmu_instance;

/* Module parameters */
bool debug_enable = true;
module_param(debug_enable, bool, 0644);
MODULE_PARM_DESC(debug_enable, "Enable debug output (default: true)");
EXPORT_SYMBOL(debug_enable);

static int timer_period_ms = 20;
module_param(timer_period_ms, int, 0644);
MODULE_PARM_DESC(timer_period_ms, "Timer period in milliseconds (default: 20)");

/* Debug: skip synchronous read in pmu_del */
static int aql_skip_del_read;
module_param(aql_skip_del_read, int, 0644);
MODULE_PARM_DESC(aql_skip_del_read, "Skip sync read in pmu_del (debug)");

/* Forward declarations for PMU callbacks */
static int amdgpu_pmu_event_init(struct perf_event *event);
static int amdgpu_pmu_add(struct perf_event *event, int flags);
static void amdgpu_pmu_del(struct perf_event *event, int flags);
static void amdgpu_pmu_start(struct perf_event *event, int flags);
static void amdgpu_pmu_stop(struct perf_event *event, int flags);
static void amdgpu_pmu_read(struct perf_event *event);
static void __amdgpu_pmu_read_internal(struct perf_event *event);

/*
 * PMU_FORMAT_ATTR - Define a perf format attribute
 *
 * This macro creates the necessary boilerplate for exposing perf event
 * format attributes via sysfs. Format attributes tell userspace tools
 * like perf how to encode event parameters into the config/config1/config2
 * fields of struct perf_event_attr.
 *
 * The format string uses the syntax "field:start-end" or "field:bit" to
 * describe which bits of the config field are used for this parameter.
 *
 * Example:
 *   PMU_FORMAT_ATTR(se, "config1:8-15")
 *   Creates /sys/bus/event_source/devices/amdgpu_pmu/format/se
 *   with contents "config1:8-15"
 *
 * This allows perf to parse: perf stat -e amdgpu_pmu/event,se=2/
 */
#define PMU_FORMAT_ATTR(_name, _format)                                                       \
	static ssize_t __pmu_format_##_name##_show(struct device *dev,                        \
						   struct device_attribute *attr, char *page) \
	{                                                                                     \
		return sprintf(page, _format "\n");                                           \
	}                                                                                     \
	static struct device_attribute format_attr_##_name =                                  \
		__ATTR(_name, 0444, __pmu_format_##_name##_show, NULL)

/*
 * Format Attributes - Define how perf encodes event parameters
 *
 * These attributes tell the perf tool how to encode named parameters
 * into the config and config1 fields of struct perf_event_attr.
 *
 * config1 bit layout (defined in pmu_dimension.h):
 *   Bits  0-7  : XCC index
 *   Bits  8-15 : SE index
 *   Bits 16-23 : SA index
 *   Bits 24-31 : WGP index
 *   Bits 32-39 : CU index
 *   Bit  40    : Aggregate flag
 */
PMU_FORMAT_ATTR(config, "config:0-63"); /* Counter ID */
PMU_FORMAT_ATTR(config1, "config1:0-63"); /* Raw dimension encoding */
PMU_FORMAT_ATTR(xcc, "config1:0-7"); /* XCC index */
PMU_FORMAT_ATTR(se, "config1:8-15"); /* SE index */
PMU_FORMAT_ATTR(sa, "config1:16-23"); /* SA index */
PMU_FORMAT_ATTR(wgp, "config1:24-31"); /* WGP index */
PMU_FORMAT_ATTR(cu, "config1:32-39"); /* CU index */
PMU_FORMAT_ATTR(aggregate, "config1:40"); /* Aggregate across dimensions */

static struct attribute *amdgpu_pmu_format_attrs[] = {
	&format_attr_config.attr, &format_attr_config1.attr,   &format_attr_xcc.attr,
	&format_attr_se.attr,	  &format_attr_sa.attr,	       &format_attr_wgp.attr,
	&format_attr_cu.attr,	  &format_attr_aggregate.attr, NULL,
};

static struct attribute_group amdgpu_pmu_format_group = {
	.name = "format",
	.attrs = amdgpu_pmu_format_attrs,
};

/* Event attributes - dynamically generated from counter_registry */
struct pmu_event_attr {
	struct device_attribute attr;
	u64 id;
};

static ssize_t amdgpu_pmu_event_show(struct device *dev, struct device_attribute *attr, char *buf)
{
	struct pmu_event_attr *pmu_attr = container_of(attr, struct pmu_event_attr, attr);
	return sprintf(buf, "config=0x%llx\n", pmu_attr->id);
}

/* Dynamic event attributes - allocated during init */
static struct pmu_event_attr *amdgpu_pmu_event_attrs_dynamic = NULL;
static struct attribute **amdgpu_pmu_event_attrs = NULL;
static size_t amdgpu_pmu_event_count = 0;

static struct attribute_group amdgpu_pmu_events_group = {
	.name = "events",
	.attrs = NULL, /* Set during init */
};

/* CPU mask attribute - shows which CPUs (GPUs) are available */
static ssize_t cpumask_show(struct device *dev, struct device_attribute *attr, char *buf)
{
	int num_gpus = aql_pmu_get_gpu_count();
	cpumask_var_t mask;
	ssize_t ret;
	int i;

	if (!zalloc_cpumask_var(&mask, GFP_KERNEL))
		return -ENOMEM;

	/* Set bits 0 to num_gpus-1 */
	for (i = 0; i < num_gpus && i < nr_cpu_ids; i++)
		cpumask_set_cpu(i, mask);

	ret = cpumap_print_to_pagebuf(true, buf, mask);
	free_cpumask_var(mask);
	return ret;
}

static DEVICE_ATTR_RO(cpumask);

static struct attribute *amdgpu_pmu_cpumask_attrs[] = {
	&dev_attr_cpumask.attr,
	NULL,
};

static struct attribute_group amdgpu_pmu_cpumask_group = {
	.attrs = amdgpu_pmu_cpumask_attrs,
};

static const struct attribute_group *amdgpu_pmu_attr_groups[] = {
	&amdgpu_pmu_format_group,
	&amdgpu_pmu_events_group,
	&amdgpu_pmu_cpumask_group,
	NULL,
};

/**
 * amdgpu_pmu_generate_samples - Update counter values for sampling events
 * @session: AQL performance session (already looked up by timer handler)
 *
 * Called from timer handler - must be atomic-safe.
 * For sampling events (perf record), the timer ensures counter values are
 * kept up-to-date by reading from the cache. The perf subsystem will sample
 * these values at the configured sample frequency via the read() callback.
 *
 * Phase 2: Also emits tracepoint events for explicit sample capture with
 * perf record -e amdgpu_pmu:counter_sample
 *
 * Note: Direct perf_event_overflow() is not available to kernel modules as
 * it's not exported. Instead, we rely on perf's built-in software-based
 * sampling mechanism which reads counter values via our read() callback.
 * The timer's job is to keep the cached values fresh.
 */
static void amdgpu_pmu_generate_samples(struct aql_perf_session *session)
{
	struct aql_measurement *meas;
	unsigned long flags;

	if (!session)
		return;

	spin_lock_irqsave(&session->measurement_lock, flags);

	list_for_each_entry(meas, &session->active_measurements, list)
	{
		struct perf_event *event = meas->event;
		const counter_def_t *counter;
		u64 prev_value, curr_value, delta;
		u32 counter_id;

		/* Skip if not a valid event */
		if (!event)
			continue;

		/* Skip if not a sampling event (sample_period = 0 means counting mode) */
		if (!is_sampling_event(event))
			continue;

		/* Update counter value using quiet internal function (no logging).
		 * This ensures the cached value is fresh when perf subsystem reads it
		 * for sampling. The actual sampling is handled by perf core. */
		__amdgpu_pmu_read_internal(event);

		/* Phase 2: Emit tracepoint with counter information and delta */
		counter_id = (u32)event->attr.config;
		counter = lookup_counter_by_id((counter_id_t)counter_id);
		if (!counter) {
			/* Counter not found - skip tracepoint but continue processing */
			continue;
		}

		/* Calculate delta since last sample */
		curr_value = local64_read(&event->count);
		prev_value = atomic64_read(&meas->prev_counter_value);
		delta = curr_value - prev_value;
		atomic64_set(&meas->prev_counter_value, curr_value);

		/* Emit tracepoint: visible via perf record -e amdgpu_pmu:counter_sample */
		trace_counter_sample(counter_id, counter->name, curr_value, delta, ktime_get_ns());
	}

	spin_unlock_irqrestore(&session->measurement_lock, flags);
}

/**
 * amdgpu_pmu_poll_measurements - Poll active measurements and schedule background reads
 * @session: AQL performance session
 * @wq: Workqueue to schedule work on
 *
 * Returns: Number of measurements successfully scheduled for polling
 */
static int amdgpu_pmu_poll_measurements(struct aql_perf_session *session,
					struct workqueue_struct *wq)
{
	struct aql_measurement *measurement, *tmp;
	unsigned long flags;
	int polled_count = 0;

	if (!session || !wq)
		return 0;

	/* Iterate through active measurements and trigger background reads */
	spin_lock_irqsave(&session->measurement_lock, flags);

	list_for_each_entry_safe(measurement, tmp, &session->active_measurements, list)
	{
		if (measurement->state == MEASUREMENT_ACTIVE) {
			/* Trigger background read to refresh cache on global workqueue */
			struct aql_work_item *work_item =
				aql_create_work_item(measurement, AQL_WORK_READ);
			if (!IS_ERR(work_item)) {
				if (queue_work(wq, &work_item->work)) {
					polled_count++;
					pmu_debug("Timer: Scheduled read for GPU %u\n",
						  measurement->gpu_id);
					pmu_info("Timer: ===== SCHEDULED READ for GPU %u =====\n",
						 measurement->gpu_id);
				} else {
					/* Work already queued - release our reference and free work_item */
					aql_measurement_put(measurement);
					kfree(work_item);
				}
			}
		}
	}

	spin_unlock_irqrestore(&session->measurement_lock, flags);

	pmu_debug("Timer: Polled %d active measurements\n", polled_count);
	return polled_count;
}

/* Timer handler - Periodic polling for background counter refresh */
enum hrtimer_restart amdgpu_pmu_timer_handler(struct hrtimer *timer)
{
	struct amdgpu_pmu *pmu = container_of(timer, struct amdgpu_pmu, timer);
	struct aql_perf_session *session;
	struct workqueue_struct *wq;
	int count;

	/*
	 * This timer is the central "software clock" for perf-dkms:
	 * - counting events: periodically refresh cached counter values
	 * - sampling events: synthesize perf samples from the latest reads
	 *
	 * The callback runs in hrtimer context, so it must not sleep or touch
	 * paths that can block. It only schedules work on the global workqueue.
	 */
	pmu_debug("Timer handler fired - polling active measurements\n");

	/* Get global workqueue */
	wq = aql_get_global_workqueue();
	if (!wq) {
		pmu_debug("Timer: Global workqueue not available\n");
		goto reschedule;
	}

	/* Get AQL session to poll measurements (lookup once, use for all operations) */
	session = aql_pmu_get_session();
	if (!session) {
		pmu_debug("Timer: No AQL session available\n");
		goto reschedule;
	}

	/* Poll all active measurements (for perf stat counting mode) */
	count = amdgpu_pmu_poll_measurements(session, wq);
	pmu_debug("Timer: Updated %d active measurements\n", count);

	/* Generate samples for sampling events (for perf record sampling mode - Phase 1) */
	amdgpu_pmu_generate_samples(session);

	aql_pmu_put_session(session);

reschedule:
	/* Reschedule timer for next period */
	hrtimer_forward_now(timer, pmu->timer_period);
	return HRTIMER_RESTART;
}

/* Start background polling timer - called when first measurement becomes active */
void amdgpu_pmu_start_timer(void)
{
	struct amdgpu_pmu *pmu = amdgpu_pmu_instance;

	/*
	 * Timer lifetime model:
	 * - start is requested when a measurement transitions to ACTIVE
	 * - start is idempotent (hrtimer_active check)
	 * - actual polling work happens asynchronously in timer handler
	 */
	pmu_info("amdgpu_pmu_start_timer() called\n");

	if (!pmu) {
		pmu_err("Cannot start timer: PMU not initialized\n");
		return;
	}

	/* Start timer if not already running */
	if (!hrtimer_active(&pmu->timer)) {
		hrtimer_start(&pmu->timer, pmu->timer_period, HRTIMER_MODE_REL);
		pmu_info("Started background polling timer (period=%d ms)\n", timer_period_ms);
	} else {
		pmu_info("Timer already active, not starting again\n");
	}
}
EXPORT_SYMBOL(amdgpu_pmu_start_timer);

/* Stop background polling timer - called when measurement stops */
void amdgpu_pmu_stop_timer_if_idle(void)
{
	struct amdgpu_pmu *pmu = amdgpu_pmu_instance;

	/*
	 * Current policy is intentionally conservative: stop unconditionally.
	 * Any subsequent measurement start will re-arm the timer.
	 *
	 * This avoids subtle races where "idle" checks can become stale between
	 * active list inspection and timer cancel.
	 */
	pmu_info("amdgpu_pmu_stop_timer_if_idle() called\n");

	if (!pmu) {
		pmu_err("Cannot stop timer: PMU not initialized\n");
		return;
	}

	/* Just cancel the timer unconditionally.
     * This is safe to call from atomic context (hrtimer_cancel handles it).
     * Timer will be restarted when next measurement becomes active. */
	if (hrtimer_active(&pmu->timer)) {
		hrtimer_cancel(&pmu->timer);
		pmu_info("Stopped background polling timer\n");
	}
}
EXPORT_SYMBOL(amdgpu_pmu_stop_timer_if_idle);

/* Find free event slot */
int amdgpu_pmu_get_event_idx(struct amdgpu_pmu *pmu)
{
	int idx;

	idx = find_first_zero_bit(pmu->used_mask, AMDGPU_PMU_MAX_EVENTS);
	if (idx == AMDGPU_PMU_MAX_EVENTS)
		return -EAGAIN;

	set_bit(idx, pmu->used_mask);
	return idx;
}

/* Free event slot */
void amdgpu_pmu_free_event_idx(struct amdgpu_pmu *pmu, int idx)
{
	if (idx >= 0 && idx < AMDGPU_PMU_MAX_EVENTS) {
		clear_bit(idx, pmu->used_mask);
		pmu->events[idx].event = NULL;
	}
}

/**
 * get_gpu_dimension_limits - Get dimension limits for a specific GPU
 * @gpu_id: GPU identifier
 * @limits: Output structure for dimension limits
 *
 * Returns: 0 on success, negative error code on failure
 */
static int get_gpu_dimension_limits(uint32_t gpu_id, struct pmu_dimension_limits *limits)
{
	struct aql_perf_session *session;
	arch_t *arch;
	uint32_t gpu_index;
	int ret = 0;

	if (!limits)
		return -EINVAL;

	session = aql_pmu_get_session();
	if (!session) {
		pmu_err("Cannot get dimension limits: no AQL session\n");
		return -ENODEV;
	}

	/* Find GPU index from GPU ID */
	for (gpu_index = 0; gpu_index < session->num_gpus; gpu_index++) {
		if (session->gpu_ids[gpu_index] == gpu_id)
			break;
	}

	if (gpu_index >= session->num_gpus) {
		pmu_err("GPU %u not found in session\n", gpu_id);
		ret = -ENODEV;
		goto out;
	}

	/* Get architecture for this specific GPU */
	if (!session->archs || !session->archs[gpu_index]) {
		pmu_err("No architecture information for GPU %u\n", gpu_id);
		ret = -ENODEV;
		goto out;
	}

	arch = session->archs[gpu_index];

	/* Populate dimension limits from this GPU's architecture */
	limits->max_xcc = (arch->num_xcc > 0) ? arch->num_xcc - 1 : 0;
	limits->max_se = (arch->num_se > 0) ? arch->num_se - 1 : 0;
	limits->max_sa = (arch->num_sa > 0) ? arch->num_sa - 1 : 0;
	limits->max_wgp = (arch->num_wgp_per_sa > 0) ? arch->num_wgp_per_sa - 1 : 0;
	limits->max_cu = (arch->num_cu > 0) ? arch->num_cu - 1 : 0;

	pmu_debug("GPU %u dimension limits: XCC=0-%u, SE=0-%u, SA=0-%u, WGP=0-%u, CU=0-%u\n",
		  gpu_id, limits->max_xcc, limits->max_se, limits->max_sa, limits->max_wgp,
		  limits->max_cu);

out:
	aql_pmu_put_session(session);
	return ret;
}

/* PMU callback: Initialize event */
static int amdgpu_pmu_event_init(struct perf_event *event)
{
	struct amdgpu_pmu *pmu = amdgpu_pmu_instance;
	struct pmu_dimension_coords dims = { 0 };
	const counter_def_t *counter;
	u64 config = event->attr.config;
	u64 config1 = event->attr.config1;
	int ret;

	/*
	 * perf core calls event_init to validate user attributes and bind driver
	 * specific state. This function defines the PMU contract:
	 * - event->attr.config  : logical counter id from registry
	 * - event->attr.config1 : optional dimension coordinates/flags
	 * - event->cpu          : GPU selection key (mapped to real gpu_id)
	 *
	 * Successful init installs an AQL measurement pointer in hw.config_base.
	 */
	pmu_debug("event_init: config=0x%llx config1=0x%llx\n", config, config1);

	/* Check if event is for our PMU */
	if (event->attr.type != event->pmu->type)
		return -ENOENT;

	/* Check if event configuration is supported */
	if (!amdgpu_pmu_is_valid_event(config)) {
		pmu_err("Unsupported event config: 0x%llx\n", config);
		return -EINVAL;
	}

	/* Allow both counting (sample_period=0) and sampling (sample_period>0) events.
	 * Sampling is supported via hrtimer-based periodic sampling (Task 1.2). */
	if (is_sampling_event(event)) {
		u64 sample_period = event->attr.sample_period;

		pmu_info("Sampling event requested with period %llu\n", sample_period);

		/* Validate sampling period is reasonable */
		if (sample_period == 0) {
			pmu_err("Sampling event must have non-zero sample_period\n");
			return -EINVAL;
		}

		/* Warn if sample_period is less than our timer period (20ms).
		 * Users may expect hardware-based sampling with precise period,
		 * but we use software timer at fixed interval. */
		if (sample_period < timer_period_ms * 1000000ULL) {
			pmu_warn(
				"Sample period %llu ns < timer period %d ms - samples will be at timer rate\n",
				sample_period, timer_period_ms);
		}

		/* Mark this event as a sampling event for the timer handler.
		 * We use hwc->sample_period (set by perf core) to distinguish
		 * sampling events from counting events at runtime. */
		pmu_debug("Event marked for sampling mode (sample_period=%llu)\n", sample_period);
	}

	/* GPU PMU doesn't support inherit (GPU counters aren't per-process) */
	if (event->attr.inherit) {
		pmu_debug("Disabling inherit flag - GPU counters can't inherit to children\n");
		event->attr.inherit = 0;
	}

	/* Require explicit CPU specification to map to GPU.
     * Users must use perf -C <cpu> where CPU number maps to GPU ID.
     * Example: perf stat -C 0 -e amdgpu_pmu/sq_waves/ (monitors GPU 0)
     *          perf stat -C 1 -e amdgpu_pmu/sq_waves/ (monitors GPU 1) */
	if (event->cpu < 0) {
		pmu_err("GPU PMU requires explicit CPU with -C flag (CPU maps to GPU ID)\n");
		pmu_err("Example: perf stat -C 0 -e amdgpu_pmu/event/ (for GPU 0)\n");
		return -EINVAL;
	}

	/* Get actual GPU device ID for this event's CPU affinity */
	uint32_t gpu_id = aql_pmu_get_gpu_id_for_event(event);
	if (gpu_id == U32_MAX) {
		pmu_err("Failed to map CPU %d to GPU\n", event->cpu);
		return -ENODEV;
	}
	pmu_debug("Mapped CPU %d to GPU %u\n", event->cpu, gpu_id);

	/* Store GPU ID for later use (note: this is actual GPU device ID, not index) */
	event->hw.idx = gpu_id;

	/* Extract and validate dimensions from config1 if specified */
	if (config1 != 0) {
		struct pmu_dimension_limits gpu_limits;

		pmu_extract_dimensions(config1, &dims);
		pmu_debug("Extracted dimensions: xcc=%u se=%u sa=%u wgp=%u cu=%u agg=%d valid=%d\n",
			  dims.xcc, dims.se, dims.sa, dims.wgp, dims.cu, dims.aggregate,
			  dims.valid);

		/* Get dimension limits for the specific GPU this event targets */
		ret = get_gpu_dimension_limits(gpu_id, &gpu_limits);
		if (ret != 0) {
			pmu_err("Failed to get dimension limits for GPU %u: %d\n", gpu_id, ret);
			return ret;
		}

		/* Validate dimensions against this GPU's hardware limits */
		if (!pmu_validate_dimensions(&dims, &gpu_limits)) {
			pmu_err("Dimension out of range for GPU %u: xcc=%u se=%u sa=%u wgp=%u cu=%u (max: %u/%u/%u/%u/%u)\n",
				gpu_id, dims.xcc, dims.se, dims.sa, dims.wgp, dims.cu,
				gpu_limits.max_xcc, gpu_limits.max_se, gpu_limits.max_sa,
				gpu_limits.max_wgp, gpu_limits.max_cu);
			return -EINVAL;
		}
		pmu_debug("Dimension validation passed\n");

		/* Get counter definition and validate it supports requested dimensions */
		pmu_debug("Looking up counter definition for config=0x%llx\n", config);
		counter = lookup_counter_by_id((counter_id_t)config);
		pmu_debug("lookup_counter_by_id returned %p\n", counter);
		if (counter) {
			ret = pmu_validate_counter_dimensions(counter, &dims);
			if (ret != 0) {
				pmu_err("Counter '%s' does not support requested dimensions "
					"(supported: 0x%x, requested: xcc=%u se=%u sa=%u wgp=%u cu=%u)\n",
					counter->name, counter->supported_dimensions, dims.xcc,
					dims.se, dims.sa, dims.wgp, dims.cu);
				return ret;
			}
			pmu_debug("Counter dimension validation passed\n");
		}
	}

	/* Initialize AQL hardware counter */
	pmu_debug("About to call aql_pmu_event_init\n");
	ret = aql_pmu_event_init(event, dims.valid ? &dims : NULL);
	pmu_debug("aql_pmu_event_init returned %d\n", ret);
	if (ret != 0) {
		pmu_err("AQL hardware counter initialization failed for config=0x%llx: %d\n",
			config, ret);
		return ret;
	}

	/* Successfully initialized hardware counter */
	if (dims.valid) {
		pmu_debug("Using AQL hardware counter for event config=0x%llx with dimensions\n",
			  config);
	} else {
		pmu_debug("Using AQL hardware counter for event config=0x%llx (no dimensions)\n",
			  config);
	}
	atomic64_inc(&pmu->hardware_events);
	atomic64_inc(&pmu->total_events);

	/* Set destroy callback to properly cleanup when event is freed */
	event->destroy = aql_pmu_event_destroy;

	return 0;
}

/* PMU callback: Add event to PMU */
static int amdgpu_pmu_add(struct perf_event *event, int flags)
{
	struct hw_perf_event *hwc = &event->hw;

	/*
	 * add() transitions a validated event into schedulable PMU state.
	 * If PERF_EF_START is set, START is queued immediately; otherwise the
	 * event stays stopped until perf core later calls ->start().
	 */
	pmu_debug("add: config=0x%llx, flags=0x%x\n", event->attr.config, flags);

	/* Check if this is an AQL hardware event */
	if (hwc->config_base != 0) {
		/* AQL hardware event - start measurement if requested */
		if (flags & PERF_EF_START) {
			int ret = aql_pmu_event_start(event);
			if (ret) {
				pmu_err("add: Failed to start AQL hardware event: %d\n", ret);
				return ret;
			}
			hwc->state = 0;
			/* Timer will be started by aql_perf_measurement_start() after measurement becomes active */
		} else {
			hwc->state = PERF_HES_STOPPED;
		}

		/* Set initial counter value */
		local64_set(&event->count, 0);

		pmu_debug("add: Added AQL hardware event successfully\n");
		return 0;
	}

	/* Event not initialized (config_base=0) - perf may be cloning/reusing event structure.
     * Return success but mark as stopped to prevent retry loop. */
	pmu_debug("add: Event not initialized, marking as stopped\n");
	hwc->state = PERF_HES_STOPPED | PERF_HES_UPTODATE;
	local64_set(&event->count, 0);
	return 0;
}

/* PMU callback: Remove event from PMU */
static void amdgpu_pmu_del(struct perf_event *event, int flags)
{
	struct hw_perf_event *hwc = &event->hw;

	/*
	 * del() removes the event from active scheduling, but does not free the
	 * measurement object. Destruction is deferred to event->destroy callback.
	 *
	 * This split is important because perf may del/add the same event object
	 * across enable/disable cycles.
	 */
	pmu_info("del: ENTRY - config=0x%llx, flags=0x%x\n", event->attr.config, flags);

	/* Check if this is an AQL hardware event */
	if (hwc->config_base != 0) {
		/* CRITICAL: del() can be called from atomic context (e.g., perf stat exit).
         * We cannot do synchronous operations that sleep from atomic context.
         * Check if we're in atomic context and handle accordingly. */
		if (in_atomic() || irqs_disabled()) {
			pmu_warn("del: Called from atomic context, reading cached value\n");

			/* Read cached counter value - this is safe in atomic context because
             * aql_pmu_event_read() just returns the cached value without sleeping */
			uint64_t counter_value = aql_pmu_event_read(event);
			local64_set(&event->count, counter_value);
			pmu_info("del: Final cached counter value=%llu\n", counter_value);

			/* Just mark as stopped - cleanup will happen in event_destroy */
			hwc->state = PERF_HES_STOPPED | PERF_HES_UPTODATE;
			pmu_debug("del: Removed AQL hardware event (atomic mode)\n");
			return;
		}

		/* Read final count before stopping.
		 * IMPORTANT: Do NOT call aql_pmu_event_read_sync from here because
		 * pmu_del runs in user process context (perf process), but GPU
		 * submissions use kthread_use_mm which is only safe from kernel
		 * threads. Instead, use the cached value from background reads.
		 * The timer has been firing periodic reads, so cached value is fresh. */
		{
			uint64_t counter_value = aql_pmu_event_read(event);
			local64_set(&event->count, counter_value);
			pmu_info("del: Final cached counter value=%llu\n", counter_value);
		}

		/* Stop the event but DON'T destroy it - the event may be re-added later.
         * Only event_destroy should free the measurement structure. */
		aql_pmu_event_stop(event);

		/* Mark event as stopped */
		hwc->state = PERF_HES_STOPPED | PERF_HES_UPTODATE;

		pmu_debug("del: Removed AQL hardware event successfully\n");
		return;
	}

	pmu_debug("del: Event not initialized (config_base=0)\n");
}

/* PMU callback: Start event */
static void amdgpu_pmu_start(struct perf_event *event, int flags)
{
	struct hw_perf_event *hwc = &event->hw;

	/*
	 * start() is pure state transition from perf perspective:
	 * - optional reload resets software-visible count
	 * - hardware START is delegated to AQL path
	 * - timer activation is handled after measurement reaches ACTIVE
	 */
	pmu_info("start: ENTRY - config=0x%llx, flags=0x%x, hwc->config_base=0x%lx\n",
		 event->attr.config, flags, hwc->config_base);

	/* Check if this is an AQL hardware event */
	if (hwc->config_base != 0) {
		pmu_debug("start: Starting AQL hardware event\n");
		/* Reset counter if requested */
		if (flags & PERF_EF_RELOAD) {
			pmu_debug("start: Resetting counter (PERF_EF_RELOAD flag set)\n");
			local64_set(&event->count, 0);
		}

		if (aql_pmu_event_start(event) == 0) {
			hwc->state = 0;
			pmu_debug("start: Started AQL hardware event config=0x%llx successfully\n",
				  event->attr.config);
			/* Timer will be started by aql_perf_measurement_start() after measurement becomes active */
		} else {
			pmu_err("start: Failed to start AQL hardware event config=0x%llx\n",
				event->attr.config);
			hwc->state = PERF_HES_STOPPED;
		}
		return;
	}

	/* Only AQL hardware events are supported */
	pmu_err("start: Non-AQL event detected, config=0x%llx\n", event->attr.config);
}

/* PMU callback: Stop event */
static void amdgpu_pmu_stop(struct perf_event *event, int flags)
{
	struct hw_perf_event *hwc = &event->hw;

	/*
	 * stop() mirrors ->start():
	 * - optional PERF_EF_UPDATE requests a final read into event->count
	 * - STOP transitions measurement to idle/offline in AQL layer
	 * - resource release is deferred to destroy path for correctness with
	 *   perf object lifetime.
	 */
	pmu_info("stop: ENTRY - config=0x%llx, flags=0x%x, hwc->config_base=0x%lx\n",
		 event->attr.config, flags, hwc->config_base);

	/* Check if this is an AQL hardware event */
	if (hwc->config_base != 0) {
		pmu_debug("stop: Stopping AQL hardware event\n");
		/* Update count if requested */
		if (flags & PERF_EF_UPDATE) {
			pmu_debug("stop: Reading final count (PERF_EF_UPDATE flag set)\n");
			amdgpu_pmu_read(event);
		}

		if (aql_pmu_event_stop(event) == 0) {
			hwc->state = PERF_HES_STOPPED;
			pmu_debug("stop: Stopped AQL hardware event config=0x%llx successfully\n",
				  event->attr.config);
		} else {
			pmu_err("stop: Failed to stop AQL hardware event config=0x%llx\n",
				event->attr.config);
		}
		return;
	}

	/* Only AQL hardware events are supported */
	pmu_err("stop: Non-AQL event detected, config=0x%llx\n", event->attr.config);
}

/* Internal read function - no logging, safe for high-frequency calls */
static void __amdgpu_pmu_read_internal(struct perf_event *event)
{
	struct hw_perf_event *hwc = &event->hw;

	/* Check if this is an AQL hardware event */
	if (hwc->config_base != 0) {
		uint64_t counter_value = aql_pmu_event_read(event);
		local64_set(&event->count, counter_value);
		return;
	}

	/* Only AQL hardware events are supported - silently return for others */
}

/* PMU callback: Read event counter */
static void amdgpu_pmu_read(struct perf_event *event)
{
	struct hw_perf_event *hwc = &event->hw;
	uint64_t old_count, new_count;

	/*
	 * read() returns the latest cached value from AQL measurement state.
	 * For high-frequency paths, actual GPU reads are done asynchronously by
	 * workqueue/timer workers to keep perf callback context lightweight.
	 */
	old_count = local64_read(&event->count);

	pmu_info("read: ENTRY - config=0x%llx, old_count=%llu, hwc->config_base=0x%lx\n",
		 event->attr.config, (unsigned long long)old_count, hwc->config_base);

	/* Call internal read function */
	__amdgpu_pmu_read_internal(event);

	/* Log results */
	new_count = local64_read(&event->count);
	if (hwc->config_base != 0) {
		pmu_info("read: AQL counter read complete - old=%llu, new=%llu, delta=%lld\n",
			 (unsigned long long)old_count, (unsigned long long)new_count,
			 (long long)(new_count - old_count));
		pmu_info("read: ========== FINAL event->count = %llu (0x%llx) ==========\n",
			 new_count, new_count);
	} else {
		pmu_err("read: Non-AQL event detected, config=0x%llx\n", event->attr.config);
	}
}

/* Initialize event attributes from counter_registry */
static int amdgpu_pmu_init_event_attrs(void)
{
	const counter_def_t *counters;
	size_t i;

	amdgpu_pmu_event_count = get_counter_count();
	counters = get_all_counters();

	pmu_info("Initializing %zu event attributes from counter registry\n",
		 amdgpu_pmu_event_count);

	/* Allocate array of pmu_event_attr structures */
	amdgpu_pmu_event_attrs_dynamic =
		kzalloc(amdgpu_pmu_event_count * sizeof(struct pmu_event_attr), GFP_KERNEL);
	if (!amdgpu_pmu_event_attrs_dynamic) {
		pmu_err("Failed to allocate event attributes\n");
		return -ENOMEM;
	}

	/* Allocate array of attribute pointers (+ 1 for NULL terminator) */
	amdgpu_pmu_event_attrs =
		kzalloc((amdgpu_pmu_event_count + 1) * sizeof(struct attribute *), GFP_KERNEL);
	if (!amdgpu_pmu_event_attrs) {
		kfree(amdgpu_pmu_event_attrs_dynamic);
		amdgpu_pmu_event_attrs_dynamic = NULL;
		pmu_err("Failed to allocate event attribute array\n");
		return -ENOMEM;
	}

	/* Initialize each event attribute */
	for (i = 0; i < amdgpu_pmu_event_count; i++) {
		struct pmu_event_attr *pmu_attr = &amdgpu_pmu_event_attrs_dynamic[i];
		const counter_def_t *counter = &counters[i];
		char *name_lower;

		/* Store the counter ID */
		pmu_attr->id = counter->id;

		/* Use counter name directly (already lowercase in registry) */
		name_lower = kstrdup(counter->name, GFP_KERNEL);
		if (!name_lower) {
			/* Cleanup on error */
			while (i > 0) {
				i--;
				kfree(amdgpu_pmu_event_attrs_dynamic[i].attr.attr.name);
			}
			kfree(amdgpu_pmu_event_attrs);
			kfree(amdgpu_pmu_event_attrs_dynamic);
			amdgpu_pmu_event_attrs = NULL;
			amdgpu_pmu_event_attrs_dynamic = NULL;
			return -ENOMEM;
		}

		/* Initialize device_attribute */
		sysfs_attr_init(&pmu_attr->attr.attr);
		pmu_attr->attr.attr.name = name_lower;
		pmu_attr->attr.attr.mode = 0444;
		pmu_attr->attr.show = amdgpu_pmu_event_show;
		pmu_attr->attr.store = NULL;

		/* Add to attribute array */
		amdgpu_pmu_event_attrs[i] = &pmu_attr->attr.attr;

		pmu_debug("  Event %zu: %s = config=0x%llx\n", i, name_lower, (u64)counter->id);
	}

	/* NULL terminate the array */
	amdgpu_pmu_event_attrs[amdgpu_pmu_event_count] = NULL;

	/* Set the events group attrs pointer */
	amdgpu_pmu_events_group.attrs = amdgpu_pmu_event_attrs;

	pmu_info("Successfully initialized %zu events\n", amdgpu_pmu_event_count);
	return 0;
}

/* Cleanup event attributes */
static void amdgpu_pmu_cleanup_event_attrs(void)
{
	size_t i;

	if (amdgpu_pmu_event_attrs_dynamic) {
		for (i = 0; i < amdgpu_pmu_event_count; i++) {
			kfree(amdgpu_pmu_event_attrs_dynamic[i].attr.attr.name);
		}
		kfree(amdgpu_pmu_event_attrs_dynamic);
		amdgpu_pmu_event_attrs_dynamic = NULL;
	}

	if (amdgpu_pmu_event_attrs) {
		kfree(amdgpu_pmu_event_attrs);
		amdgpu_pmu_event_attrs = NULL;
	}

	amdgpu_pmu_event_count = 0;
	amdgpu_pmu_events_group.attrs = NULL;
}

/* Module initialization */
static int __init amdgpu_pmu_init(void)
{
	struct amdgpu_pmu *pmu;
	int ret;

	/*
	 * Initialization phases (strict order):
	 * 1) Build sysfs event table from counter registry
	 * 2) Allocate/register perf PMU callbacks
	 * 3) Bring up AQL backend session/workqueue
	 * 4) Validate GPU architecture metadata required for dimensions
	 * 5) Optionally register trace miscdevice
	 *
	 * Any failure unwinds in reverse order to avoid leaked global state.
	 */
	pmu_info("Initializing PMU Stub module v%s\n", AMDGPU_PMU_VERSION);

	/* Initialize event attributes from counter_registry */
	ret = amdgpu_pmu_init_event_attrs();
	if (ret)
		return ret;

	/* Allocate PMU structure */
	pmu = kzalloc(sizeof(*pmu), GFP_KERNEL);
	if (!pmu) {
		amdgpu_pmu_cleanup_event_attrs();
		return -ENOMEM;
	}

	/* Initialize PMU structure */
	spin_lock_init(&pmu->lock);
	bitmap_zero(pmu->used_mask, AMDGPU_PMU_MAX_EVENTS);
	pmu->num_events = 0;

	/* Initialize counters */
	atomic64_set(&pmu->total_events, 0);
	atomic64_set(&pmu->total_samples, 0);
	atomic64_set(&pmu->hardware_events, 0);

	/* Initialize AQL hardware integration */
	mutex_init(&pmu->aql_mutex);

	/* Initialize timer */
	hrtimer_setup(&pmu->timer, amdgpu_pmu_timer_handler, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
	pmu->timer_period = ms_to_ktime(timer_period_ms);

	/* Set up PMU structure */
	pmu->pmu = (struct pmu){
		.name = PMU_NAME,
		.task_ctx_nr = -1,
		.event_init = amdgpu_pmu_event_init,
		.add = amdgpu_pmu_add,
		.del = amdgpu_pmu_del,
		.start = amdgpu_pmu_start,
		.stop = amdgpu_pmu_stop,
		.read = amdgpu_pmu_read,
		.attr_groups = amdgpu_pmu_attr_groups,
		/* Note: PERF_PMU_CAP_NO_INTERRUPT removed to enable sampling mode.
		 * We use hrtimer-based sampling instead of hardware overflow interrupts.
		 * See amdgpu_pmu_generate_samples() for sample generation. */
		.capabilities = PERF_PMU_CAP_NO_EXCLUDE,
	};

	/* Register PMU with perf subsystem */
	ret = perf_pmu_register(&pmu->pmu, PMU_NAME, -1);
	if (ret) {
		pmu_err("Failed to register PMU: %d\n", ret);
		amdgpu_pmu_cleanup_event_attrs();
		kfree(pmu);
		return ret;
	}

	amdgpu_pmu_instance = pmu;

	/* Initialize AQL PMU integration */
	ret = aql_pmu_init();
	if (ret == 0) {
		pmu_info("AQL hardware acceleration enabled\n");
	} else {
		pmu_err("AQL hardware acceleration required but not available: %d\n", ret);
		perf_pmu_unregister(&pmu->pmu);
		amdgpu_pmu_cleanup_event_attrs();
		kfree(pmu);
		return ret;
	}

	/* Verify GPU architecture is available for dimension validation */
	{
		struct aql_perf_session *session = aql_pmu_get_session();
		if (!session || session->num_gpus == 0 || !session->archs || !session->archs[0]) {
			pmu_err("GPU architecture unavailable - cannot validate dimensions\n");
			if (session)
				aql_pmu_put_session(session);
			aql_pmu_cleanup();
			perf_pmu_unregister(&pmu->pmu);
			amdgpu_pmu_cleanup_event_attrs();
			kfree(pmu);
			return -ENODEV;
		}
		pmu_info("GPU architecture loaded - %u GPU(s) available\n", session->num_gpus);
		aql_pmu_put_session(session);
	}

	/* Register trace device for userspace tracepoint emission */
	ret = amdgpu_pmu_trace_dev_init();
	if (ret) {
		pmu_err("Failed to register trace device: %d\n", ret);
		/* Non-fatal - PMU still works, just no userspace trace emission */
	}

	pmu_info("PMU Stub module loaded successfully\n");
	pmu_info("Events available under: /sys/bus/event_source/devices/%s/\n", PMU_NAME);

	return 0;
}

/* Module cleanup */
static void __exit amdgpu_pmu_exit(void)
{
	struct amdgpu_pmu *pmu = amdgpu_pmu_instance;

	/*
	 * Teardown ordering is critical:
	 * - stop timer first so no new READ work is queued
	 * - unregister PMU so perf destroys events and drops measurement refs
	 * - flush workqueue so async STOP/READ work cannot race cleanup
	 * - release global AQL session/workqueue last
	 */
	pmu_info("Unloading PMU Stub module\n");

	/* Unregister trace device */
	amdgpu_pmu_trace_dev_exit();

	if (pmu) {
		/* Step 1: Stop timer to prevent new work from being queued */
		hrtimer_cancel(&pmu->timer);

		/* Step 2: Unregister PMU FIRST — this forces perf subsystem to
		 * destroy all open events (calling destroy callbacks), which
		 * releases measurement references. Must happen before session
		 * cleanup so destroy callbacks can access the session. */
		perf_pmu_unregister(&pmu->pmu);
		pmu_info("PMU unregistered\n");

		/* Step 3: Flush workqueues — drain any async work items queued
		 * by destroy callbacks (e.g., async stop from atomic context) */
		aql_pmu_flush_all_measurements();

		/* Step 4: Now safe to release session — all events destroyed,
		 * all work items completed */
		aql_pmu_cleanup();
		pmu_info("AQL hardware acceleration disabled\n");

		/* Print statistics */
		pmu_info("Total events created: %lld\n", atomic64_read(&pmu->total_events));
		pmu_info("Hardware events: %lld\n", atomic64_read(&pmu->hardware_events));
		pmu_info("Total samples: %lld\n", atomic64_read(&pmu->total_samples));

		/* Free memory */
		kfree(pmu);
		amdgpu_pmu_instance = NULL;
	}

	/* Cleanup event attributes */
	amdgpu_pmu_cleanup_event_attrs();

	pmu_info("PMU Stub module unloaded\n");
}

module_init(amdgpu_pmu_init);
module_exit(amdgpu_pmu_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Advanced Micro Devices, Inc.");
MODULE_DESCRIPTION("AMD GPU PMU driver for Linux perf subsystem");
MODULE_VERSION(AMDGPU_PMU_VERSION);
