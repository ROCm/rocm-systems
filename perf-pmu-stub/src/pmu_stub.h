/*
 * pmu_stub.h - Header for PMU Stub Kernel Module
 *
 * This module implements a skeleton PMU driver for the Linux perf subsystem.
 * It provides a minimal but complete implementation that can be extended
 * for real hardware performance monitoring.
 */

#ifndef _PMU_STUB_H
#define _PMU_STUB_H

#include <linux/perf_event.h>
#include <linux/hrtimer.h>
#include <linux/spinlock.h>
#include <linux/list.h>
#include <linux/device.h>

/* Module information */
#define MODULE_NAME "pmu_stub"
#define PMU_NAME "pmu_stub"
#define PMU_STUB_VERSION "1.0.0"

/* Maximum number of concurrent events we support */
#define PMU_STUB_MAX_EVENTS 64

/* Event types we support (dummy events for demonstration) */
enum pmu_stub_event_id {
    PMU_STUB_EVENT_CYCLES = 0x00,
    PMU_STUB_EVENT_INSTRUCTIONS = 0x01,
    PMU_STUB_EVENT_CACHE_MISSES = 0x02,
    PMU_STUB_EVENT_BANDWIDTH = 0x03,
    PMU_STUB_EVENT_MAX
};

/* Event configuration structure */
struct pmu_stub_event_config {
    const char *name;
    u64 config;
    u64 config_mask;
    const char *description;
};

/* Per-event private data */
struct pmu_stub_event {
    struct perf_event *event;
    u64 prev_count;
    u64 period;
    bool active;
};

/* Main PMU structure */
struct pmu_stub {
    struct pmu pmu;                          /* Base PMU structure */
    struct device *dev;                      /* Device for sysfs */

    /* Event management */
    spinlock_t lock;                         /* Protects event_list */
    struct pmu_stub_event events[PMU_STUB_MAX_EVENTS];
    DECLARE_BITMAP(used_mask, PMU_STUB_MAX_EVENTS);
    int num_events;

    /* Timer for simulating counter updates */
    struct hrtimer timer;
    ktime_t timer_period;

    /* Statistics */
    atomic64_t total_events;
    atomic64_t total_samples;

    /* Simulated counter values */
    atomic64_t counter_cycles;
    atomic64_t counter_instructions;
    atomic64_t counter_cache_misses;
    atomic64_t counter_bandwidth;
};

/* Function prototypes */

/* Helper functions - shared between modules */
enum hrtimer_restart pmu_stub_timer_handler(struct hrtimer *timer);
void pmu_stub_update_counters(struct pmu_stub *pmu);
int pmu_stub_get_event_idx(struct pmu_stub *pmu);
void pmu_stub_free_event_idx(struct pmu_stub *pmu, int idx);

/* Event utility functions */
const char *pmu_stub_get_event_name(u64 config);
const char *pmu_stub_get_event_description(u64 config);
bool pmu_stub_is_valid_event(u64 config);
u64 pmu_stub_get_counter_value(struct pmu_stub *pmu, u64 config);
void pmu_stub_update_counter(struct pmu_stub *pmu, u64 config, s64 delta);
void pmu_stub_reset_counters(struct pmu_stub *pmu);
void pmu_stub_print_event_stats(struct pmu_stub *pmu);

/* Sysfs functions */
int pmu_stub_init_sysfs(struct pmu_stub *pmu);
void pmu_stub_cleanup_sysfs(struct pmu_stub *pmu);

/* Debug helpers */
#ifdef DEBUG
#define pmu_debug(fmt, ...) \
    pr_debug("[" MODULE_NAME "] " fmt, ##__VA_ARGS__)
#else
#define pmu_debug(fmt, ...) do {} while (0)
#endif

#define pmu_info(fmt, ...) \
    pr_info("[" MODULE_NAME "] " fmt, ##__VA_ARGS__)

#define pmu_err(fmt, ...) \
    pr_err("[" MODULE_NAME "] " fmt, ##__VA_ARGS__)

#endif /* _PMU_STUB_H */