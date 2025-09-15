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

#include "pmu_stub.h"
#include "kfd_test.h"
#include "aql_perf.h"
#include "aql_c/counter_registry.h"

/* Global PMU instance */
static struct pmu_stub *pmu_stub_instance;

/* Module parameters */
bool debug_enable = true;
module_param(debug_enable, bool, 0644);
MODULE_PARM_DESC(debug_enable, "Enable debug output (default: true)");
EXPORT_SYMBOL(debug_enable);

static int timer_period_ms = 100;
module_param(timer_period_ms, int, 0644);
MODULE_PARM_DESC(timer_period_ms, "Timer period in milliseconds (default: 100)");

/* Forward declarations for PMU callbacks */
static int pmu_stub_event_init(struct perf_event *event);
static int pmu_stub_add(struct perf_event *event, int flags);
static void pmu_stub_del(struct perf_event *event, int flags);
static void pmu_stub_start(struct perf_event *event, int flags);
static void pmu_stub_stop(struct perf_event *event, int flags);
static void pmu_stub_read(struct perf_event *event);

/* Sysfs attribute functions */
static ssize_t pmu_stub_format_show(struct device *dev,
                                    struct device_attribute *attr,
                                    char *buf)
{
    return sprintf(buf, "config:0-63\n");
}

static DEVICE_ATTR(format, 0444, pmu_stub_format_show, NULL);

static struct attribute *pmu_stub_format_attrs[] = {
    &dev_attr_format.attr,
    NULL,
};

static struct attribute_group pmu_stub_format_group = {
    .name = "format",
    .attrs = pmu_stub_format_attrs,
};

/* Event attributes - dynamically generated from counter_registry */
struct pmu_event_attr {
    struct device_attribute attr;
    u64 id;
};

static ssize_t pmu_stub_event_show(struct device *dev,
                                   struct device_attribute *attr,
                                   char *buf)
{
    struct pmu_event_attr *pmu_attr = container_of(attr, struct pmu_event_attr, attr);
    return sprintf(buf, "config=0x%llx\n", pmu_attr->id);
}

/* Dynamic event attributes - allocated during init */
static struct pmu_event_attr *pmu_stub_event_attrs_dynamic = NULL;
static struct attribute **pmu_stub_event_attrs = NULL;
static size_t pmu_stub_event_count = 0;

static struct attribute_group pmu_stub_events_group = {
    .name = "events",
    .attrs = NULL,  /* Set during init */
};

static const struct attribute_group *pmu_stub_attr_groups[] = {
    &pmu_stub_format_group,
    &pmu_stub_events_group,
    NULL,
};

/* Timer handler - Legacy stub (not used with hardware-only mode) */
enum hrtimer_restart pmu_stub_timer_handler(struct hrtimer *timer)
{
    (void)timer; /* Unused in hardware-only mode */

    /* Hardware-only mode: timer not used */
    pmu_debug("Timer handler called but not used in hardware-only mode\n");

    return HRTIMER_NORESTART;
}

/* Find free event slot */
int pmu_stub_get_event_idx(struct pmu_stub *pmu)
{
    int idx;

    idx = find_first_zero_bit(pmu->used_mask, PMU_STUB_MAX_EVENTS);
    if (idx == PMU_STUB_MAX_EVENTS)
        return -EAGAIN;

    set_bit(idx, pmu->used_mask);
    return idx;
}

/* Free event slot */
void pmu_stub_free_event_idx(struct pmu_stub *pmu, int idx)
{
    if (idx >= 0 && idx < PMU_STUB_MAX_EVENTS) {
        clear_bit(idx, pmu->used_mask);
        pmu->events[idx].event = NULL;
        pmu->events[idx].active = false;
    }
}

/* PMU callback: Initialize event */
static int pmu_stub_event_init(struct perf_event *event)
{
    struct pmu_stub *pmu = pmu_stub_instance;
    int ret;

    pmu_debug("event_init: config=0x%llx\n", event->attr.config);

    /* Check if event is for our PMU */
    if (event->attr.type != event->pmu->type)
        return -ENOENT;

    /* Check if event configuration is supported */
    if (!pmu_stub_is_valid_event(event->attr.config)) {
        pmu_err("Unsupported event config: 0x%llx\n", event->attr.config);
        return -EINVAL;
    }

    /* We don't support sampling */
    if (is_sampling_event(event)) {
        pmu_err("Sampling events not supported\n");
        return -EOPNOTSUPP;
    }

    /* GPU PMU doesn't support inherit (GPU counters aren't per-process) */
    if (event->attr.inherit) {
        pmu_debug("Rejecting inherit flag - GPU counters can't inherit to children\n");
        return -EOPNOTSUPP;
    }

    // /* We don't support exclude filters */
    // if (event->attr.exclude_user || event->attr.exclude_kernel ||
    //     event->attr.exclude_hv || event->attr.exclude_idle) {
    //     pmu_err("Exclude filters not supported\n");
    //     return -EOPNOTSUPP;
    // }

    /* Initialize AQL hardware counter */
    ret = aql_pmu_event_init(event);
    if (ret != 0) {
        pmu_err("AQL hardware counter initialization failed for config=0x%llx: %d\n",
                event->attr.config, ret);
        return ret;
    }

    /* Successfully initialized hardware counter */
    pmu_debug("Using AQL hardware counter for event config=0x%llx\n", event->attr.config);
    atomic64_inc(&pmu->hardware_events);
    atomic64_inc(&pmu->total_events);

    return 0;
}

/* PMU callback: Add event to PMU */
static int pmu_stub_add(struct perf_event *event, int flags)
{
    struct hw_perf_event *hwc = &event->hw;

    pmu_info("add: ENTRY - config=0x%llx, flags=0x%x, hwc->config_base=0x%lx\n",
             event->attr.config, flags, hwc->config_base);

    /* Check if this is an AQL hardware event */
    if (hwc->config_base != 0) {
        pmu_debug("add: AQL hardware event detected, hwc->config_base=0x%lx\n", hwc->config_base);
        /* AQL hardware event - start measurement if requested */
        if (flags & PERF_EF_START) {
            pmu_debug("add: Starting AQL hardware event immediately (PERF_EF_START flag set)\n");
            int ret = aql_pmu_event_start(event);
            if (ret) {
                pmu_err("add: Failed to start AQL hardware event: %d\n", ret);
                return ret;
            }
            hwc->state = 0;
            pmu_debug("add: AQL hardware event started successfully, state=0\n");
        } else {
            hwc->state = PERF_HES_STOPPED;
            pmu_debug("add: AQL hardware event added but not started, state=PERF_HES_STOPPED\n");
        }

        /* Set initial counter value */
        local64_set(&event->count, 0);

        pmu_debug("add: Added AQL hardware event config=0x%llx successfully\n", event->attr.config);
        return 0;
    }

    /* Only AQL hardware events are supported */
    pmu_err("add: Non-AQL event detected, config=0x%llx\n", event->attr.config);
    return -EINVAL;
}

/* PMU callback: Remove event from PMU */
static void pmu_stub_del(struct perf_event *event, int flags)
{
    struct hw_perf_event *hwc = &event->hw;

    pmu_info("del: ENTRY - config=0x%llx, flags=0x%x, hwc->config_base=0x%lx\n",
             event->attr.config, flags, hwc->config_base);

    /* Check if this is an AQL hardware event */
    if (hwc->config_base != 0) {
        pmu_debug("del: Removing AQL hardware event\n");
        /* AQL hardware event - stop and cleanup */
        if (flags & PERF_EF_UPDATE) {
            pmu_debug("del: Reading final count (PERF_EF_UPDATE flag set)\n");
            pmu_stub_read(event);
        }

        /* Destroy handles stopping internally - don't call stop separately
         * to avoid duplicate work items and use-after-free */
        pmu_debug("del: Destroying AQL hardware event\n");
        aql_pmu_event_destroy(event);

        pmu_debug("del: Removed AQL hardware event config=0x%llx successfully\n", event->attr.config);
        return;
    }

    /* Only AQL hardware events are supported */
    pmu_err("del: Non-AQL event detected, config=0x%llx\n", event->attr.config);
}

/* PMU callback: Start event */
static void pmu_stub_start(struct perf_event *event, int flags)
{
    struct hw_perf_event *hwc = &event->hw;

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
            pmu_debug("start: Started AQL hardware event config=0x%llx successfully\n", event->attr.config);
        } else {
            pmu_err("start: Failed to start AQL hardware event config=0x%llx\n", event->attr.config);
            hwc->state = PERF_HES_STOPPED;
        }
        return;
    }

    /* Only AQL hardware events are supported */
    pmu_err("start: Non-AQL event detected, config=0x%llx\n", event->attr.config);
}

/* PMU callback: Stop event */
static void pmu_stub_stop(struct perf_event *event, int flags)
{
    struct hw_perf_event *hwc = &event->hw;

    pmu_info("stop: ENTRY - config=0x%llx, flags=0x%x, hwc->config_base=0x%lx\n",
             event->attr.config, flags, hwc->config_base);

    /* Check if this is an AQL hardware event */
    if (hwc->config_base != 0) {
        pmu_debug("stop: Stopping AQL hardware event\n");
        /* Update count if requested */
        if (flags & PERF_EF_UPDATE) {
            pmu_debug("stop: Reading final count (PERF_EF_UPDATE flag set)\n");
            pmu_stub_read(event);
        }

        if (aql_pmu_event_stop(event) == 0) {
            hwc->state = PERF_HES_STOPPED;
            pmu_debug("stop: Stopped AQL hardware event config=0x%llx successfully\n", event->attr.config);
        } else {
            pmu_err("stop: Failed to stop AQL hardware event config=0x%llx\n", event->attr.config);
        }
        return;
    }

    /* Only AQL hardware events are supported */
    pmu_err("stop: Non-AQL event detected, config=0x%llx\n", event->attr.config);
}

/* PMU callback: Read event counter */
static void pmu_stub_read(struct perf_event *event)
{
    struct hw_perf_event *hwc = &event->hw;
    uint64_t old_count, new_count;

    old_count = local64_read(&event->count);

    pmu_info("read: ENTRY - config=0x%llx, old_count=%llu, hwc->config_base=0x%lx\n",
             event->attr.config, (unsigned long long)old_count, hwc->config_base);

    /* Check if this is an AQL hardware event */
    if (hwc->config_base != 0) {
        pmu_info("read: Reading AQL hardware counter for config=0x%llx\n", event->attr.config);
        uint64_t counter_value = aql_pmu_event_read(event);
        local64_set(&event->count, counter_value);
        new_count = local64_read(&event->count);
        pmu_info("read: AQL counter read complete - old=%llu, new=%llu, delta=%lld\n",
                 (unsigned long long)old_count, (unsigned long long)new_count,
                 (long long)(new_count - old_count));
        return;
    }

    /* Only AQL hardware events are supported */
    pmu_err("read: Non-AQL event detected, config=0x%llx\n", event->attr.config);
}

/* Initialize event attributes from counter_registry */
static int pmu_stub_init_event_attrs(void)
{
    const counter_def_t *counters;
    size_t i;

    pmu_stub_event_count = get_counter_count();
    counters = get_all_counters();

    pmu_info("Initializing %zu event attributes from counter registry\n", pmu_stub_event_count);

    /* Allocate array of pmu_event_attr structures */
    pmu_stub_event_attrs_dynamic = kzalloc(pmu_stub_event_count * sizeof(struct pmu_event_attr),
                                           GFP_KERNEL);
    if (!pmu_stub_event_attrs_dynamic) {
        pmu_err("Failed to allocate event attributes\n");
        return -ENOMEM;
    }

    /* Allocate array of attribute pointers (+ 1 for NULL terminator) */
    pmu_stub_event_attrs = kzalloc((pmu_stub_event_count + 1) * sizeof(struct attribute *),
                                   GFP_KERNEL);
    if (!pmu_stub_event_attrs) {
        kfree(pmu_stub_event_attrs_dynamic);
        pmu_stub_event_attrs_dynamic = NULL;
        pmu_err("Failed to allocate event attribute array\n");
        return -ENOMEM;
    }

    /* Initialize each event attribute */
    for (i = 0; i < pmu_stub_event_count; i++) {
        struct pmu_event_attr *pmu_attr = &pmu_stub_event_attrs_dynamic[i];
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
                kfree(pmu_stub_event_attrs_dynamic[i].attr.attr.name);
            }
            kfree(pmu_stub_event_attrs);
            kfree(pmu_stub_event_attrs_dynamic);
            pmu_stub_event_attrs = NULL;
            pmu_stub_event_attrs_dynamic = NULL;
            return -ENOMEM;
        }

        /* Initialize device_attribute */
        sysfs_attr_init(&pmu_attr->attr.attr);
        pmu_attr->attr.attr.name = name_lower;
        pmu_attr->attr.attr.mode = 0444;
        pmu_attr->attr.show = pmu_stub_event_show;
        pmu_attr->attr.store = NULL;

        /* Add to attribute array */
        pmu_stub_event_attrs[i] = &pmu_attr->attr.attr;

        pmu_debug("  Event %zu: %s = config=0x%llx\n", i, name_lower, (u64)counter->id);
    }

    /* NULL terminate the array */
    pmu_stub_event_attrs[pmu_stub_event_count] = NULL;

    /* Set the events group attrs pointer */
    pmu_stub_events_group.attrs = pmu_stub_event_attrs;

    pmu_info("Successfully initialized %zu events\n", pmu_stub_event_count);
    return 0;
}

/* Cleanup event attributes */
static void pmu_stub_cleanup_event_attrs(void)
{
    size_t i;

    if (pmu_stub_event_attrs_dynamic) {
        for (i = 0; i < pmu_stub_event_count; i++) {
            kfree(pmu_stub_event_attrs_dynamic[i].attr.attr.name);
        }
        kfree(pmu_stub_event_attrs_dynamic);
        pmu_stub_event_attrs_dynamic = NULL;
    }

    if (pmu_stub_event_attrs) {
        kfree(pmu_stub_event_attrs);
        pmu_stub_event_attrs = NULL;
    }

    pmu_stub_event_count = 0;
    pmu_stub_events_group.attrs = NULL;
}

/* Module initialization */
static int __init pmu_stub_init(void)
{
    struct pmu_stub *pmu;
    int ret;

    pmu_info("Initializing PMU Stub module v%s\n", PMU_STUB_VERSION);

    /* Initialize event attributes from counter_registry */
    ret = pmu_stub_init_event_attrs();
    if (ret)
        return ret;

    /* Allocate PMU structure */
    pmu = kzalloc(sizeof(*pmu), GFP_KERNEL);
    if (!pmu) {
        pmu_stub_cleanup_event_attrs();
        return -ENOMEM;
    }

    /* Initialize PMU structure */
    spin_lock_init(&pmu->lock);
    bitmap_zero(pmu->used_mask, PMU_STUB_MAX_EVENTS);
    pmu->num_events = 0;

    /* Initialize counters */
    atomic64_set(&pmu->counter_sq_waves, 0);
    atomic64_set(&pmu->counter_sq_instructions, 0);
    atomic64_set(&pmu->counter_ta_busy, 0);
    atomic64_set(&pmu->total_events, 0);
    atomic64_set(&pmu->total_samples, 0);
    atomic64_set(&pmu->hardware_events, 0);
    atomic64_set(&pmu->simulation_events, 0);

    /* Initialize AQL hardware integration */
    mutex_init(&pmu->aql_mutex);

    /* Initialize timer */
    hrtimer_setup(&pmu->timer, pmu_stub_timer_handler, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
    pmu->timer_period = ms_to_ktime(timer_period_ms);

    /* Set up PMU structure */
    pmu->pmu = (struct pmu) {
        .name           = PMU_NAME,
        .task_ctx_nr    = -1,
        .event_init     = pmu_stub_event_init,
        .add            = pmu_stub_add,
        .del            = pmu_stub_del,
        .start          = pmu_stub_start,
        .stop           = pmu_stub_stop,
        .read           = pmu_stub_read,
        .attr_groups    = pmu_stub_attr_groups,
        .capabilities   = PERF_PMU_CAP_NO_INTERRUPT,
    };

    /* Register PMU with perf subsystem */
    ret = perf_pmu_register(&pmu->pmu, PMU_NAME, -1);
    if (ret) {
        pmu_err("Failed to register PMU: %d\n", ret);
        pmu_stub_cleanup_event_attrs();
        kfree(pmu);
        return ret;
    }

    pmu_stub_instance = pmu;

    /* Initialize AQL PMU integration */
    ret = aql_pmu_init();
    if (ret == 0) {
        pmu_info("AQL hardware acceleration enabled\n");
    } else {
        pmu_err("AQL hardware acceleration required but not available: %d\n", ret);
        perf_pmu_unregister(&pmu->pmu);
        pmu_stub_cleanup_event_attrs();
        kfree(pmu);
        return ret;
    }

    pmu_info("PMU Stub module loaded successfully\n");
    pmu_info("Events available under: /sys/bus/event_source/devices/%s/\n", PMU_NAME);

    return 0;
}

/* Module cleanup */
static void __exit pmu_stub_exit(void)
{
    struct pmu_stub *pmu = pmu_stub_instance;

    pmu_info("Unloading PMU Stub module\n");

    if (pmu) {
        /* Cleanup AQL integration first */
        aql_pmu_cleanup();
        pmu_info("AQL hardware acceleration disabled\n");

        /* Cancel timer */
        hrtimer_cancel(&pmu->timer);

        /* Unregister PMU */
        perf_pmu_unregister(&pmu->pmu);

        /* Print statistics */
        pmu_info("Total events created: %lld\n",
                 atomic64_read(&pmu->total_events));
        pmu_info("Hardware events: %lld\n",
                 atomic64_read(&pmu->hardware_events));
        pmu_info("Simulation events: %lld\n",
                 atomic64_read(&pmu->simulation_events));
        pmu_info("Total samples: %lld\n",
                 atomic64_read(&pmu->total_samples));

        /* Free memory */
        kfree(pmu);
        pmu_stub_instance = NULL;
    }

    /* Cleanup event attributes */
    pmu_stub_cleanup_event_attrs();

    pmu_info("PMU Stub module unloaded\n");
}

module_init(pmu_stub_init);
module_exit(pmu_stub_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Skeleton PMU driver for Linux perf subsystem");
MODULE_VERSION(PMU_STUB_VERSION);