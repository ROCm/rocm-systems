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

/* Forward declarations */
struct aql_perf_stats;

/* Module information */
#define MODULE_NAME "pmu_stub"
#define PMU_NAME "pmu_stub"
#define PMU_STUB_VERSION "1.0.0"

/* Maximum number of concurrent events we support */
#define PMU_STUB_MAX_EVENTS 64

/* Event types we support (maps to GFX12 hardware counters) */
/*
 * Event IDs are now dynamically defined by counter_registry.h
 * The event config value corresponds to counter_id_t from the registry.
 * Available events can be queried using pmu_stub_get_event_count().
 */

/* Per-event private data */
struct pmu_stub_event {
    struct perf_event *event;
    u64 prev_count;
    u64 period;
    bool active;
    bool uses_aql_hardware;                  /* Event uses AQL hardware counters */
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

    /* AQL Hardware Integration */
    struct mutex aql_mutex;                  /* Protects AQL operations */

    /* Statistics */
    atomic64_t total_events;
    atomic64_t total_samples;
    atomic64_t hardware_events;              /* Number of hardware events */
    atomic64_t simulation_events;            /* Number of simulation events */

    /* Simulated counter values */
    atomic64_t counter_sq_waves;
    atomic64_t counter_sq_instructions;
    atomic64_t counter_ta_busy;
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
size_t pmu_stub_get_event_count(void);
u64 pmu_stub_get_counter_value(struct pmu_stub *pmu, u64 config);
void pmu_stub_update_counter(struct pmu_stub *pmu, u64 config, s64 delta);
void pmu_stub_reset_counters(struct pmu_stub *pmu);
void pmu_stub_print_event_stats(struct pmu_stub *pmu);

/* Sysfs functions */
int pmu_stub_init_sysfs(struct pmu_stub *pmu);
void pmu_stub_cleanup_sysfs(struct pmu_stub *pmu);

/* AQL Hardware Integration Functions */
int aql_pmu_init(void);
void aql_pmu_cleanup(void);
bool aql_pmu_is_available(void);
int aql_pmu_event_init(struct perf_event *event);
void aql_pmu_event_destroy(struct perf_event *event);
int aql_pmu_event_start(struct perf_event *event);
int aql_pmu_event_stop(struct perf_event *event);
uint64_t aql_pmu_event_read(struct perf_event *event);
void aql_pmu_get_stats(struct aql_perf_stats *stats);

/* Debug helpers */
#define pmu_debug(fmt, ...) \
    pr_info("[" MODULE_NAME "] " fmt, ##__VA_ARGS__)

#define pmu_info(fmt, ...) \
    pr_info("[" MODULE_NAME "] " fmt, ##__VA_ARGS__)

#define pmu_err(fmt, ...) \
    pr_err("[" MODULE_NAME "] " fmt, ##__VA_ARGS__)

#endif /* _PMU_STUB_H */