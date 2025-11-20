/*
 * amdgpu_pmu.h - Header for AMD GPU PMU Kernel Module
 *
 * This module implements an AMD GPU PMU driver for the Linux perf subsystem.
 * It provides performance monitoring for AMD GPU hardware counters.
 */

#ifndef _AMDGPU_PMU_H
#define _AMDGPU_PMU_H

#include <linux/perf_event.h>
#include <linux/hrtimer.h>
#include <linux/spinlock.h>
#include <linux/list.h>
#include <linux/device.h>

/* Forward declarations */
struct aql_perf_stats;
struct pmu_dimension_coords;

/* Module information */
#define MODULE_NAME "amdgpu_pmu"
#define PMU_NAME "amdgpu_pmu"
#define AMDGPU_PMU_VERSION "1.0.0"

/* Maximum number of concurrent events we support */
#define AMDGPU_PMU_MAX_EVENTS 64

/* Event types we support (maps to GFX12 hardware counters) */
/*
 * Event IDs are now dynamically defined by counter_registry.h
 * The event config value corresponds to counter_id_t from the registry.
 * Available events can be queried using amdgpu_pmu_get_event_count().
 */

/* Per-event private data */
struct amdgpu_pmu_event {
    struct perf_event *event;
    u64 prev_count;
    u64 period;
    bool active;
    bool uses_aql_hardware;                  /* Event uses AQL hardware counters */
};

/* Main PMU structure */
struct amdgpu_pmu {
    struct pmu pmu;                          /* Base PMU structure */
    struct device *dev;                      /* Device for sysfs */

    /* Event management */
    spinlock_t lock;                         /* Protects event_list */
    struct amdgpu_pmu_event events[AMDGPU_PMU_MAX_EVENTS];
    DECLARE_BITMAP(used_mask, AMDGPU_PMU_MAX_EVENTS);
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
enum hrtimer_restart amdgpu_pmu_timer_handler(struct hrtimer *timer);
void amdgpu_pmu_update_counters(struct amdgpu_pmu *pmu);
int amdgpu_pmu_get_event_idx(struct amdgpu_pmu *pmu);
void amdgpu_pmu_free_event_idx(struct amdgpu_pmu *pmu, int idx);
void amdgpu_pmu_start_timer(void);
void amdgpu_pmu_stop_timer_if_idle(void);

/* Event utility functions */
const char *amdgpu_pmu_get_event_name(u64 config);
const char *amdgpu_pmu_get_event_description(u64 config);
bool amdgpu_pmu_is_valid_event(u64 config);
size_t amdgpu_pmu_get_event_count(void);
u64 amdgpu_pmu_get_counter_value(struct amdgpu_pmu *pmu, u64 config);
void amdgpu_pmu_update_counter(struct amdgpu_pmu *pmu, u64 config, s64 delta);
void amdgpu_pmu_reset_counters(struct amdgpu_pmu *pmu);
void amdgpu_pmu_print_event_stats(struct amdgpu_pmu *pmu);

/* Sysfs functions */
int amdgpu_pmu_init_sysfs(struct amdgpu_pmu *pmu);
void amdgpu_pmu_cleanup_sysfs(struct amdgpu_pmu *pmu);

/* AQL Hardware Integration Functions */
int aql_pmu_init(void);
void aql_pmu_cleanup(void);
bool aql_pmu_is_available(void);
int aql_pmu_get_gpu_count(void);
uint32_t aql_pmu_get_gpu_id_for_event(struct perf_event *event);
int aql_pmu_event_init(struct perf_event *event, const struct pmu_dimension_coords *dims);
void aql_pmu_event_destroy(struct perf_event *event);
int aql_pmu_event_start(struct perf_event *event);
int aql_pmu_event_stop(struct perf_event *event);
uint64_t aql_pmu_event_read(struct perf_event *event);
uint64_t aql_pmu_event_read_sync(struct perf_event *event);
void aql_pmu_get_stats(struct aql_perf_stats *stats);

/* Debug helpers */
#define pmu_debug(fmt, ...) \
    pr_info("[" MODULE_NAME "] " fmt, ##__VA_ARGS__)

#define pmu_info(fmt, ...) \
    pr_info("[" MODULE_NAME "] " fmt, ##__VA_ARGS__)

#define pmu_warn(fmt, ...) \
    pr_warn("[" MODULE_NAME "] " fmt, ##__VA_ARGS__)

#define pmu_err(fmt, ...) \
    pr_err("[" MODULE_NAME "] " fmt, ##__VA_ARGS__)

#endif /* _AMDGPU_PMU_H */