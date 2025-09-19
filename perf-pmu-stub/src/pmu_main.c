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

/* Event attributes */
static ssize_t pmu_stub_event_show_sq_waves(struct device *dev,
                                            struct device_attribute *attr,
                                            char *buf)
{
    /* Show event configuration for sq_waves */
    return sprintf(buf, "config=0x%llx\n", (u64)PMU_STUB_EVENT_SQ_WAVES);
}

static ssize_t pmu_stub_event_show_sq_instructions(struct device *dev,
                                                   struct device_attribute *attr,
                                                   char *buf)
{
    /* Show event configuration for sq_instructions */
    return sprintf(buf, "config=0x%llx\n", (u64)PMU_STUB_EVENT_SQ_INSTRUCTIONS);
}

static ssize_t pmu_stub_event_show_ta_busy(struct device *dev,
                                           struct device_attribute *attr,
                                           char *buf)
{
    /* Show event configuration for ta_busy */
    return sprintf(buf, "config=0x%llx\n", (u64)PMU_STUB_EVENT_TA_BUSY);
}

/* Define individual event attributes */
static struct device_attribute pmu_stub_event_attr_sq_waves =
    __ATTR(sq_waves, 0444, pmu_stub_event_show_sq_waves, NULL);
static struct device_attribute pmu_stub_event_attr_sq_instructions =
    __ATTR(sq_instructions, 0444, pmu_stub_event_show_sq_instructions, NULL);
static struct device_attribute pmu_stub_event_attr_ta_busy =
    __ATTR(ta_busy, 0444, pmu_stub_event_show_ta_busy, NULL);

static struct attribute *pmu_stub_event_attrs[] = {
    &pmu_stub_event_attr_sq_waves.attr,
    &pmu_stub_event_attr_sq_instructions.attr,
    &pmu_stub_event_attr_ta_busy.attr,
    NULL,
};

static struct attribute_group pmu_stub_events_group = {
    .name = "events",
    .attrs = pmu_stub_event_attrs,
};

static const struct attribute_group *pmu_stub_attr_groups[] = {
    &pmu_stub_format_group,
    &pmu_stub_events_group,
    NULL,
};

/* Timer handler for simulating counter updates */
enum hrtimer_restart pmu_stub_timer_handler(struct hrtimer *timer)
{
    struct pmu_stub *pmu = container_of(timer, struct pmu_stub, timer);
    unsigned long flags;
    int i;

    spin_lock_irqsave(&pmu->lock, flags);

    /* Update simulated counters */
    atomic64_add(1000, &pmu->counter_sq_waves);
    atomic64_add(500, &pmu->counter_sq_instructions);
    atomic64_add(10, &pmu->counter_ta_busy);

    /* Update active events */
    for (i = 0; i < PMU_STUB_MAX_EVENTS; i++) {
        if (test_bit(i, pmu->used_mask) && pmu->events[i].active) {
            struct perf_event *event = pmu->events[i].event;
            if (event) {
                /* Simulate counter increment based on event type */
                u64 delta = 0;
                switch (event->attr.config) {
                case PMU_STUB_EVENT_SQ_WAVES:
                    delta = 1000;
                    break;
                case PMU_STUB_EVENT_SQ_INSTRUCTIONS:
                    delta = 500;
                    break;
                case PMU_STUB_EVENT_TA_BUSY:
                    delta = 10;
                    break;
                }
                local64_add(delta, &event->count);
            }
        }
    }

    spin_unlock_irqrestore(&pmu->lock, flags);

    /* Restart timer */
    hrtimer_forward_now(timer, pmu->timer_period);
    return HRTIMER_RESTART;
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
    if (event->attr.config >= PMU_STUB_EVENT_MAX) {
        pmu_err("Unsupported event config: 0x%llx\n", event->attr.config);
        return -EINVAL;
    }

    /* We don't support sampling */
    if (is_sampling_event(event)) {
        pmu_err("Sampling events not supported\n");
        return -EOPNOTSUPP;
    }

    // /* We don't support exclude filters */
    // if (event->attr.exclude_user || event->attr.exclude_kernel ||
    //     event->attr.exclude_hv || event->attr.exclude_idle) {
    //     pmu_err("Exclude filters not supported\n");
    //     return -EOPNOTSUPP;
    // }

    /* Try AQL hardware counters first if available and preferred */
    if (pmu->aql_available && pmu->prefer_hardware) {
        ret = aql_pmu_event_init(event);
        if (ret == 0) {
            pmu_debug("Using AQL hardware counter for event config=0x%llx\n", event->attr.config);
            atomic64_inc(&pmu->hardware_events);
            atomic64_inc(&pmu->total_events);
            return 0;
        }
        pmu_debug("AQL hardware counter not available, falling back to simulation\n");
    }

    /* Initialize event for simulation */
    event->hw.idx = -1;
    event->hw.config = event->attr.config;

    atomic64_inc(&pmu->simulation_events);
    atomic64_inc(&pmu->total_events);

    return 0;
}

/* PMU callback: Add event to PMU */
static int pmu_stub_add(struct perf_event *event, int flags)
{
    struct pmu_stub *pmu = pmu_stub_instance;
    struct hw_perf_event *hwc = &event->hw;
    unsigned long irq_flags;
    int idx;

    pmu_info("add: ENTRY - config=0x%llx, flags=0x%x, hwc->config_base=0x%llx\n",
             event->attr.config, flags, hwc->config_base);

    /* Check if this is an AQL hardware event */
    if (hwc->config_base != 0) {
        pmu_debug("add: AQL hardware event detected, hwc->config_base=0x%llx\n", hwc->config_base);
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

    pmu_debug("add: Handling simulation event (hwc->config_base=0)\n");

    /* Simulation event handling */
    spin_lock_irqsave(&pmu->lock, irq_flags);

    /* Get free event slot */
    idx = pmu_stub_get_event_idx(pmu);
    if (idx < 0) {
        pmu_err("add: No free event slots available\n");
        spin_unlock_irqrestore(&pmu->lock, irq_flags);
        return -EAGAIN;
    }

    pmu_debug("add: Allocated simulation event slot %d\n", idx);

    /* Assign event to slot */
    hwc->idx = idx;
    pmu->events[idx].event = event;
    pmu->events[idx].prev_count = 0;
    pmu->events[idx].active = false;
    pmu->events[idx].uses_aql_hardware = false;

    /* Set initial counter value */
    local64_set(&event->count, 0);

    /* Start event if requested */
    if (flags & PERF_EF_START) {
        pmu->events[idx].active = true;
        hwc->state = 0;
        pmu_debug("add: Started simulation event immediately (PERF_EF_START flag set)\n");
    } else {
        hwc->state = PERF_HES_STOPPED;
        pmu_debug("add: Simulation event added but not started\n");
    }

    pmu->num_events++;

    /* Start timer if this is the first event */
    if (pmu->num_events == 1) {
        hrtimer_start(&pmu->timer, pmu->timer_period, HRTIMER_MODE_REL);
        pmu_debug("add: Started simulation timer (first event)\n");
    }

    spin_unlock_irqrestore(&pmu->lock, irq_flags);

    pmu_debug("add: Successfully added simulation event config=0x%llx, slot=%d, total_events=%d\n",
              event->attr.config, idx, pmu->num_events);
    return 0;
}

/* PMU callback: Remove event from PMU */
static void pmu_stub_del(struct perf_event *event, int flags)
{
    struct pmu_stub *pmu = pmu_stub_instance;
    struct hw_perf_event *hwc = &event->hw;
    unsigned long irq_flags;

    pmu_info("del: ENTRY - config=0x%llx, flags=0x%x, hwc->config_base=0x%llx\n",
             event->attr.config, flags, hwc->config_base);

    /* Check if this is an AQL hardware event */
    if (hwc->config_base != 0) {
        pmu_debug("del: Removing AQL hardware event\n");
        /* AQL hardware event - stop and cleanup */
        if (flags & PERF_EF_UPDATE) {
            pmu_debug("del: Reading final count (PERF_EF_UPDATE flag set)\n");
            pmu_stub_read(event);
        }

        pmu_debug("del: Stopping AQL hardware event\n");
        aql_pmu_event_stop(event);
        pmu_debug("del: Destroying AQL hardware event\n");
        aql_pmu_event_destroy(event);

        pmu_debug("del: Removed AQL hardware event config=0x%llx successfully\n", event->attr.config);
        return;
    }

    pmu_debug("del: Removing simulation event\n");

    /* Simulation event handling */
    spin_lock_irqsave(&pmu->lock, irq_flags);

    /* Stop event if running */
    if (hwc->idx >= 0) {
        pmu_debug("del: Stopping simulation event slot %d\n", hwc->idx);
        pmu->events[hwc->idx].active = false;

        /* Update final count */
        if (flags & PERF_EF_UPDATE) {
            pmu_debug("del: Reading final count for simulation event (PERF_EF_UPDATE flag set)\n");
            pmu_stub_read(event);
        }

        /* Free event slot */
        pmu_stub_free_event_idx(pmu, hwc->idx);
        hwc->idx = -1;

        pmu->num_events--;
        pmu_debug("del: Freed simulation event slot, remaining events=%d\n", pmu->num_events);

        /* Stop timer if no more events */
        if (pmu->num_events == 0) {
            hrtimer_cancel(&pmu->timer);
            pmu_debug("del: Stopped simulation timer (no more events)\n");
        }
    } else {
        pmu_debug("del: Simulation event has no valid slot (hwc->idx=%d)\n", hwc->idx);
    }

    spin_unlock_irqrestore(&pmu->lock, irq_flags);
    pmu_debug("del: Successfully removed simulation event config=0x%llx\n", event->attr.config);
}

/* PMU callback: Start event */
static void pmu_stub_start(struct perf_event *event, int flags)
{
    struct pmu_stub *pmu = pmu_stub_instance;
    struct hw_perf_event *hwc = &event->hw;
    unsigned long irq_flags;

    pmu_info("start: ENTRY - config=0x%llx, flags=0x%x, hwc->config_base=0x%llx\n",
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

    pmu_debug("start: Starting simulation event\n");

    /* Simulation event handling */
    spin_lock_irqsave(&pmu->lock, irq_flags);

    if (hwc->idx >= 0 && hwc->idx < PMU_STUB_MAX_EVENTS) {
        pmu_debug("start: Activating simulation event slot %d\n", hwc->idx);
        pmu->events[hwc->idx].active = true;
        hwc->state = 0;

        /* Reset counter if requested */
        if (flags & PERF_EF_RELOAD) {
            pmu_debug("start: Resetting simulation counter (PERF_EF_RELOAD flag set)\n");
            local64_set(&event->count, 0);
            pmu->events[hwc->idx].prev_count = 0;
        }
        pmu_debug("start: Successfully started simulation event config=0x%llx, slot=%d\n",
                  event->attr.config, hwc->idx);
    } else {
        pmu_debug("start: Invalid simulation event slot (hwc->idx=%d)\n", hwc->idx);
    }

    spin_unlock_irqrestore(&pmu->lock, irq_flags);
}

/* PMU callback: Stop event */
static void pmu_stub_stop(struct perf_event *event, int flags)
{
    struct pmu_stub *pmu = pmu_stub_instance;
    struct hw_perf_event *hwc = &event->hw;
    unsigned long irq_flags;

    pmu_info("stop: ENTRY - config=0x%llx, flags=0x%x, hwc->config_base=0x%llx\n",
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

    pmu_debug("stop: Stopping simulation event\n");

    /* Simulation event handling */
    spin_lock_irqsave(&pmu->lock, irq_flags);

    if (hwc->idx >= 0 && hwc->idx < PMU_STUB_MAX_EVENTS) {
        pmu_debug("stop: Deactivating simulation event slot %d\n", hwc->idx);
        pmu->events[hwc->idx].active = false;
        hwc->state = PERF_HES_STOPPED;

        /* Update count if requested */
        if (flags & PERF_EF_UPDATE) {
            pmu_debug("stop: Reading final count for simulation event (PERF_EF_UPDATE flag set)\n");
            pmu_stub_read(event);
        }
        pmu_debug("stop: Successfully stopped simulation event config=0x%llx, slot=%d\n",
                  event->attr.config, hwc->idx);
    } else {
        pmu_debug("stop: Invalid simulation event slot (hwc->idx=%d)\n", hwc->idx);
    }

    spin_unlock_irqrestore(&pmu->lock, irq_flags);
}

/* PMU callback: Read event counter */
static void pmu_stub_read(struct perf_event *event)
{
    struct hw_perf_event *hwc = &event->hw;

    pmu_info("read: ENTRY - config=0x%llx, count=%llu, hwc->config_base=0x%llx\n",
             event->attr.config, (unsigned long long)local64_read(&event->count), hwc->config_base);

    /* Check if this is an AQL hardware event */
    if (hwc->config_base != 0) {
        pmu_debug("read: Reading AQL hardware counter\n");
        uint64_t counter_value = aql_pmu_event_read(event);
        local64_set(&event->count, counter_value);
        pmu_debug("read: Read AQL hardware counter value: %llu\n", counter_value);
        return;
    }

    pmu_debug("read: Reading simulation event counter (already updated by timer)\n");
    /* Simulation events - counter value is already updated in timer handler */
    /* This could be enhanced to read from hardware registers in a real driver */
}

/* Module initialization */
static int __init pmu_stub_init(void)
{
    struct pmu_stub *pmu;
    int ret;

    pmu_info("Initializing PMU Stub module v%s\n", PMU_STUB_VERSION);

    /* Allocate PMU structure */
    pmu = kzalloc(sizeof(*pmu), GFP_KERNEL);
    if (!pmu)
        return -ENOMEM;

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
    pmu->aql_available = false;
    pmu->prefer_hardware = true;  /* Prefer hardware over simulation by default */

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
        kfree(pmu);
        return ret;
    }

    pmu_stub_instance = pmu;

    /* Initialize AQL PMU integration */
    ret = aql_pmu_init();
    if (ret == 0) {
        pmu->aql_available = true;
        pmu_info("AQL hardware acceleration enabled\n");
    } else {
        pmu_info("AQL hardware acceleration not available: %d (using simulation only)\n", ret);
        pmu->aql_available = false;
        pmu->prefer_hardware = false;
    }

    pmu_info("PMU Stub module loaded successfully\n");
    pmu_info("Events available under: /sys/bus/event_source/devices/%s/\n", PMU_NAME);

    /* Test KFD ioctl functionality */
    pmu_info("Testing KFD ioctl functionality...\n");
    ret = kfd_test_get_version();
    if (ret == 0) {
        kfd_test_print_result();
        pmu_info("KFD integration test completed successfully\n");
    } else {
        kfd_test_print_result();
        pmu_info("KFD integration test failed, but module will continue to load\n");
    }

    return 0;
}

/* Module cleanup */
static void __exit pmu_stub_exit(void)
{
    struct pmu_stub *pmu = pmu_stub_instance;

    pmu_info("Unloading PMU Stub module\n");

    if (pmu) {
        /* Cleanup AQL integration first */
        if (pmu->aql_available) {
            aql_pmu_cleanup();
            pmu_info("AQL hardware acceleration disabled\n");
        }

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

    pmu_info("PMU Stub module unloaded\n");
}

module_init(pmu_stub_init);
module_exit(pmu_stub_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Skeleton PMU driver for Linux perf subsystem");
MODULE_VERSION(PMU_STUB_VERSION);