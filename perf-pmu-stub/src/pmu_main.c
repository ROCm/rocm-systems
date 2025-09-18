/*
 * pmu_main.c - Main PMU Stub Driver Implementation
 *
 * This module implements a skeleton PMU driver for the Linux perf subsystem.
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

/* Global PMU instance */
static struct pmu_stub *pmu_stub_instance;

/* Module parameters */
static bool debug_enable = false;
module_param(debug_enable, bool, 0644);
MODULE_PARM_DESC(debug_enable, "Enable debug output (default: false)");

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
static ssize_t pmu_stub_event_show(struct device *dev,
                                   struct device_attribute *attr,
                                   char *buf)
{
    /* Show event configuration */
    return sprintf(buf, "config=0x%llx\n", (u64)0);
}

/* Define individual event attributes */
static struct device_attribute pmu_stub_event_attr_cycles =
    __ATTR(cycles, 0444, pmu_stub_event_show, NULL);
static struct device_attribute pmu_stub_event_attr_instructions =
    __ATTR(instructions, 0444, pmu_stub_event_show, NULL);
static struct device_attribute pmu_stub_event_attr_cache_misses =
    __ATTR(cache_misses, 0444, pmu_stub_event_show, NULL);
static struct device_attribute pmu_stub_event_attr_bandwidth =
    __ATTR(bandwidth, 0444, pmu_stub_event_show, NULL);

static struct attribute *pmu_stub_event_attrs[] = {
    &pmu_stub_event_attr_cycles.attr,
    &pmu_stub_event_attr_instructions.attr,
    &pmu_stub_event_attr_cache_misses.attr,
    &pmu_stub_event_attr_bandwidth.attr,
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
    atomic64_add(1000, &pmu->counter_cycles);
    atomic64_add(500, &pmu->counter_instructions);
    atomic64_add(10, &pmu->counter_cache_misses);
    atomic64_add(100, &pmu->counter_bandwidth);

    /* Update active events */
    for (i = 0; i < PMU_STUB_MAX_EVENTS; i++) {
        if (test_bit(i, pmu->used_mask) && pmu->events[i].active) {
            struct perf_event *event = pmu->events[i].event;
            if (event) {
                /* Simulate counter increment based on event type */
                u64 delta = 0;
                switch (event->attr.config) {
                case PMU_STUB_EVENT_CYCLES:
                    delta = 1000;
                    break;
                case PMU_STUB_EVENT_INSTRUCTIONS:
                    delta = 500;
                    break;
                case PMU_STUB_EVENT_CACHE_MISSES:
                    delta = 10;
                    break;
                case PMU_STUB_EVENT_BANDWIDTH:
                    delta = 100;
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

    /* We don't support exclude filters */
    if (event->attr.exclude_user || event->attr.exclude_kernel ||
        event->attr.exclude_hv || event->attr.exclude_idle) {
        pmu_err("Exclude filters not supported\n");
        return -EOPNOTSUPP;
    }

    /* Initialize event */
    event->hw.idx = -1;
    event->hw.config = event->attr.config;

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

    pmu_debug("add: config=0x%llx, flags=0x%x\n", event->attr.config, flags);

    spin_lock_irqsave(&pmu->lock, irq_flags);

    /* Get free event slot */
    idx = pmu_stub_get_event_idx(pmu);
    if (idx < 0) {
        spin_unlock_irqrestore(&pmu->lock, irq_flags);
        return -EAGAIN;
    }

    /* Assign event to slot */
    hwc->idx = idx;
    pmu->events[idx].event = event;
    pmu->events[idx].prev_count = 0;
    pmu->events[idx].active = false;

    /* Set initial counter value */
    local64_set(&event->count, 0);

    /* Start event if requested */
    if (flags & PERF_EF_START) {
        pmu->events[idx].active = true;
        hwc->state = 0;
    } else {
        hwc->state = PERF_HES_STOPPED;
    }

    pmu->num_events++;

    /* Start timer if this is the first event */
    if (pmu->num_events == 1) {
        hrtimer_start(&pmu->timer, pmu->timer_period, HRTIMER_MODE_REL);
    }

    spin_unlock_irqrestore(&pmu->lock, irq_flags);

    return 0;
}

/* PMU callback: Remove event from PMU */
static void pmu_stub_del(struct perf_event *event, int flags)
{
    struct pmu_stub *pmu = pmu_stub_instance;
    struct hw_perf_event *hwc = &event->hw;
    unsigned long irq_flags;

    pmu_debug("del: config=0x%llx, flags=0x%x\n", event->attr.config, flags);

    spin_lock_irqsave(&pmu->lock, irq_flags);

    /* Stop event if running */
    if (hwc->idx >= 0) {
        pmu->events[hwc->idx].active = false;

        /* Update final count */
        if (flags & PERF_EF_UPDATE) {
            pmu_stub_read(event);
        }

        /* Free event slot */
        pmu_stub_free_event_idx(pmu, hwc->idx);
        hwc->idx = -1;

        pmu->num_events--;

        /* Stop timer if no more events */
        if (pmu->num_events == 0) {
            hrtimer_cancel(&pmu->timer);
        }
    }

    spin_unlock_irqrestore(&pmu->lock, irq_flags);
}

/* PMU callback: Start event */
static void pmu_stub_start(struct perf_event *event, int flags)
{
    struct pmu_stub *pmu = pmu_stub_instance;
    struct hw_perf_event *hwc = &event->hw;
    unsigned long irq_flags;

    pmu_debug("start: config=0x%llx, flags=0x%x\n", event->attr.config, flags);

    spin_lock_irqsave(&pmu->lock, irq_flags);

    if (hwc->idx >= 0 && hwc->idx < PMU_STUB_MAX_EVENTS) {
        pmu->events[hwc->idx].active = true;
        hwc->state = 0;

        /* Reset counter if requested */
        if (flags & PERF_EF_RELOAD) {
            local64_set(&event->count, 0);
            pmu->events[hwc->idx].prev_count = 0;
        }
    }

    spin_unlock_irqrestore(&pmu->lock, irq_flags);
}

/* PMU callback: Stop event */
static void pmu_stub_stop(struct perf_event *event, int flags)
{
    struct pmu_stub *pmu = pmu_stub_instance;
    struct hw_perf_event *hwc = &event->hw;
    unsigned long irq_flags;

    pmu_debug("stop: config=0x%llx, flags=0x%x\n", event->attr.config, flags);

    spin_lock_irqsave(&pmu->lock, irq_flags);

    if (hwc->idx >= 0 && hwc->idx < PMU_STUB_MAX_EVENTS) {
        pmu->events[hwc->idx].active = false;
        hwc->state = PERF_HES_STOPPED;

        /* Update count if requested */
        if (flags & PERF_EF_UPDATE) {
            pmu_stub_read(event);
        }
    }

    spin_unlock_irqrestore(&pmu->lock, irq_flags);
}

/* PMU callback: Read event counter */
static void pmu_stub_read(struct perf_event *event)
{
    pmu_debug("read: config=0x%llx, count=%llu\n",
              event->attr.config, (unsigned long long)local64_read(&event->count));

    /* Counter value is already updated in timer handler */
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
    atomic64_set(&pmu->counter_cycles, 0);
    atomic64_set(&pmu->counter_instructions, 0);
    atomic64_set(&pmu->counter_cache_misses, 0);
    atomic64_set(&pmu->counter_bandwidth, 0);
    atomic64_set(&pmu->total_events, 0);
    atomic64_set(&pmu->total_samples, 0);

    /* Initialize timer */
    hrtimer_setup(&pmu->timer, pmu_stub_timer_handler, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
    pmu->timer_period = ms_to_ktime(timer_period_ms);

    /* Set up PMU structure */
    pmu->pmu = (struct pmu) {
        .name           = PMU_NAME,
        .task_ctx_nr    = perf_invalid_context,
        .event_init     = pmu_stub_event_init,
        .add            = pmu_stub_add,
        .del            = pmu_stub_del,
        .start          = pmu_stub_start,
        .stop           = pmu_stub_stop,
        .read           = pmu_stub_read,
        .attr_groups    = pmu_stub_attr_groups,
        .capabilities   = PERF_PMU_CAP_NO_INTERRUPT | PERF_PMU_CAP_NO_EXCLUDE,
    };

    /* Register PMU with perf subsystem */
    ret = perf_pmu_register(&pmu->pmu, PMU_NAME, -1);
    if (ret) {
        pmu_err("Failed to register PMU: %d\n", ret);
        kfree(pmu);
        return ret;
    }

    pmu_stub_instance = pmu;

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
        /* Cancel timer */
        hrtimer_cancel(&pmu->timer);

        /* Unregister PMU */
        perf_pmu_unregister(&pmu->pmu);

        /* Print statistics */
        pmu_info("Total events created: %lld\n",
                 atomic64_read(&pmu->total_events));
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