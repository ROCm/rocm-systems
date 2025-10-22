/*
 * aql_pmu_integration.c - AQL PMU Integration Implementation
 *
 * This module implements integration between the AQL packet submission system
 * and the existing perf-dkms infrastructure.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/perf_event.h>
#include <linux/atomic.h>
#include <linux/limits.h>

#include "aql_perf.h"
#include "amdgpu_pmu.h"

/* Global AQL session - initialized during module load */
static struct aql_perf_session *global_aql_session = NULL;
static struct mutex aql_pmu_mutex;
static bool aql_feature_available = false;

/**
 * aql_pmu_init - Initialize AQL PMU integration
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_pmu_init(void)
{
    int ret;

    mutex_init(&aql_pmu_mutex);

    aql_info("Initializing AQL PMU integration");

    /* Create global AQL session */
    global_aql_session = aql_perf_session_create();
    if (IS_ERR(global_aql_session)) {
        ret = PTR_ERR(global_aql_session);
        aql_err("Failed to create global AQL session: %d", ret);
        global_aql_session = NULL;
        return ret;
    }

    /* Initialize the session */
    ret = aql_perf_session_initialize(global_aql_session);
    if (ret) {
        aql_err("Failed to initialize global AQL session: %d", ret);
        aql_perf_session_put(global_aql_session);
        global_aql_session = NULL;
        return ret;
    }

    aql_feature_available = true;
    aql_info("AQL PMU integration initialized successfully with %u GPUs",
             global_aql_session->num_gpus);

    return 0;
}

/**
 * aql_pmu_cleanup - Cleanup AQL PMU integration
 */
void aql_pmu_cleanup(void)
{
    aql_info("Cleaning up AQL PMU integration");

    mutex_lock(&aql_pmu_mutex);

    if (global_aql_session) {
        aql_perf_session_put(global_aql_session);
        global_aql_session = NULL;
    }

    aql_feature_available = false;

    mutex_unlock(&aql_pmu_mutex);

    aql_info("AQL PMU integration cleanup complete");
}

/**
 * aql_pmu_is_available - Check if AQL PMU feature is available
 *
 * Returns: true if available, false otherwise
 */
bool aql_pmu_is_available(void)
{
    bool available;

    mutex_lock(&aql_pmu_mutex);
    available = aql_feature_available && global_aql_session &&
                (global_aql_session->state == SESSION_ACTIVE);
    mutex_unlock(&aql_pmu_mutex);

    return available;
}

/**
 * aql_pmu_should_use_hardware - Determine if event should use hardware counters
 * @event: Perf event to check
 *
 * Returns: true if should use hardware, false to use simulation
 */
static bool aql_pmu_should_use_hardware(struct perf_event *event)
{
    if (!event || !event->pmu)
        return false;

    /* Check if AQL is available */
    if (!aql_pmu_is_available())
        return false;

    /* All events are now supported by hardware (validated by counter_registry) */
    return true;
}

/**
 * aql_pmu_select_gpu - Select appropriate GPU for event
 * @event: Perf event
 *
 * Returns: GPU ID to use, or UINT32_MAX on error
 */
static uint32_t aql_pmu_select_gpu(struct perf_event *event)
{
    struct aql_perf_session *session = global_aql_session;
    uint32_t cpu = event->cpu;
    uint32_t gpu_id;

    aql_debug("GPU selection: event->cpu=%d, session=%p", cpu, session);

    if (!session || session->num_gpus == 0) {
        aql_debug("GPU selection failed: session=%p, num_gpus=%u",
                  session, session ? session->num_gpus : 0);
        return U32_MAX;
    }

    aql_debug("GPU selection: available GPUs=%u", session->num_gpus);

    /* For now, use simple CPU-to-GPU mapping */
    if (cpu == -1) {
        aql_debug("GPU selection: CPU-wide event (cpu=-1), selecting first healthy GPU");
        /* Use first healthy GPU for CPU-wide events */
        for (uint32_t i = 0; i < session->num_gpus; i++) {
            gpu_id = session->gpu_ids[i];
            aql_debug("GPU selection: checking GPU[%u] ID=%u, disabled=%s",
                      i, gpu_id, aql_perf_is_gpu_disabled(session, gpu_id) ? "yes" : "no");
            if (!aql_perf_is_gpu_disabled(session, gpu_id)) {
                aql_debug("GPU selection: selected GPU ID=%u for CPU-wide event", gpu_id);
                return gpu_id;
            }
        }
    } else {
        /* Map CPU to GPU using modulo */
        uint32_t healthy_count = aql_perf_get_healthy_gpu_count(session);
        aql_debug("GPU selection: CPU-specific event (cpu=%u), healthy_gpus=%u", cpu, healthy_count);

        if (healthy_count == 0) {
            aql_debug("GPU selection: no healthy GPUs available");
            return U32_MAX;
        }

        uint32_t gpu_index = cpu % healthy_count;
        uint32_t healthy_idx = 0;

        aql_debug("GPU selection: mapping CPU %u to GPU index %u (cpu %% healthy_count)",
                  cpu, gpu_index);

        for (uint32_t i = 0; i < session->num_gpus; i++) {
            gpu_id = session->gpu_ids[i];
            if (!aql_perf_is_gpu_disabled(session, gpu_id)) {
                aql_debug("GPU selection: healthy GPU[%u] index=%u, target_index=%u, GPU_ID=%u",
                          i, healthy_idx, gpu_index, gpu_id);
                if (healthy_idx == gpu_index) {
                    aql_debug("GPU selection: selected GPU ID=%u for CPU %u", gpu_id, cpu);
                    return gpu_id;
                }
                healthy_idx++;
            }
        }
    }

    aql_debug("GPU selection: failed to find suitable GPU");
    return U32_MAX;
}

/**
 * aql_pmu_event_init - Initialize event with AQL support
 * @event: Perf event to initialize
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_pmu_event_init(struct perf_event *event)
{
    struct aql_measurement *measurement;
    uint32_t gpu_id;

    if (!aql_pmu_should_use_hardware(event)) {
        return -EOPNOTSUPP; /* Use simulation instead */
    }

    mutex_lock(&aql_pmu_mutex);

    if (!global_aql_session || global_aql_session->state != SESSION_ACTIVE) {
        aql_debug("AQL session not active, falling back to simulation");
        mutex_unlock(&aql_pmu_mutex);
        return -EOPNOTSUPP;
    }

    /* Select GPU for this event */
    gpu_id = aql_pmu_select_gpu(event);
    if (gpu_id == U32_MAX) {
        aql_debug("No suitable GPU found for event, falling back to simulation");
        mutex_unlock(&aql_pmu_mutex);
        return -EOPNOTSUPP;
    }

    /* Create AQL measurement */
    measurement = aql_perf_measurement_create(global_aql_session, gpu_id, event);
    if (IS_ERR(measurement)) {
        aql_err("Failed to create AQL measurement: %ld", PTR_ERR(measurement));
        mutex_unlock(&aql_pmu_mutex);
        return PTR_ERR(measurement);
    }

    /* Store measurement in event's hardware config */
    event->hw.config_base = (unsigned long)measurement;

    aql_debug("Initialized hardware event for GPU %u (event config=0x%llx)",
              gpu_id, event->attr.config);

    mutex_unlock(&aql_pmu_mutex);
    return 0;
}

/**
 * aql_pmu_event_destroy - Destroy event with AQL support
 * @event: Perf event to destroy
 */
void aql_pmu_event_destroy(struct perf_event *event)
{
    struct aql_measurement *measurement;

    if (!event->hw.config_base)
        return;

    measurement = (struct aql_measurement *)event->hw.config_base;

    mutex_lock(&aql_pmu_mutex);

    aql_debug("Destroying hardware event for GPU %u", measurement->gpu_id);
    aql_perf_measurement_destroy(measurement);

    event->hw.config_base = 0;

    mutex_unlock(&aql_pmu_mutex);
}

/**
 * aql_pmu_event_start - Start AQL event measurement
 * @event: Perf event to start
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_pmu_event_start(struct perf_event *event)
{
    struct aql_measurement *measurement;
    int ret;

    if (!event->hw.config_base)
        return -EINVAL;

    measurement = (struct aql_measurement *)event->hw.config_base;

    /* Use atomic version to avoid sleeping in PMU callback */
    ret = aql_perf_measurement_start_atomic(measurement);
    if (ret) {
        aql_err("Failed to schedule start for AQL measurement on GPU %u: %d",
                measurement->gpu_id, ret);
    } else {
        aql_debug("Scheduled start for AQL measurement on GPU %u", measurement->gpu_id);
    }

    return ret;
}

/**
 * aql_pmu_event_stop - Stop AQL event measurement
 * @event: Perf event to stop
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_pmu_event_stop(struct perf_event *event)
{
    struct aql_measurement *measurement;
    int ret;

    if (!event->hw.config_base)
        return -EINVAL;

    measurement = (struct aql_measurement *)event->hw.config_base;

    /* Use atomic version to avoid sleeping in PMU callback */
    ret = aql_perf_measurement_stop_atomic(measurement);
    if (ret) {
        aql_err("Failed to schedule stop for AQL measurement on GPU %u: %d",
                measurement->gpu_id, ret);
    } else {
        aql_debug("Scheduled stop for AQL measurement on GPU %u", measurement->gpu_id);
    }

    return ret;
}

/**
 * aql_pmu_event_read - Read AQL event counter value
 * @event: Perf event to read
 *
 * Returns: Counter value
 */
uint64_t aql_pmu_event_read(struct perf_event *event)
{
    struct aql_measurement *measurement;
    uint64_t counter_value;

    aql_info("[PMU] READ: Entry - config=0x%llx, config_base=0x%lx",
             event->attr.config, event->hw.config_base);

    if (!event->hw.config_base) {
        aql_warn("[PMU] READ: No measurement attached to event");
        return 0;
    }

    measurement = (struct aql_measurement *)event->hw.config_base;

    aql_info("[PMU] READ: Calling aql_perf_measurement_read_atomic for GPU %u",
             measurement->gpu_id);

    /* Use atomic version to return cached value and schedule refresh */
    counter_value = aql_perf_measurement_read_atomic(measurement);

    aql_info("[PMU] READ: Got counter value %llu from GPU %u (state=%d, cache_valid=%d)",
             counter_value, measurement->gpu_id, measurement->state, measurement->cache_valid);

    return counter_value;
}

/**
 * aql_pmu_get_session - Get reference to global AQL session
 *
 * Retrieves the global AQL session and increments its reference count.
 * Caller must call aql_pmu_put_session() when done.
 *
 * Note: Currently returns the single global session. The reference counting
 * is designed for future flexibility if multiple sessions or session lifecycle
 * management becomes necessary.
 *
 * Returns: Global AQL session with incremented refcount, or NULL
 */
struct aql_perf_session *aql_pmu_get_session(void)
{
    struct aql_perf_session *session;

    mutex_lock(&aql_pmu_mutex);
    session = global_aql_session;
    if (session)
        aql_perf_session_get(session);
    mutex_unlock(&aql_pmu_mutex);

    return session;
}

/**
 * aql_pmu_put_session - Release reference to AQL session
 * @session: Session to release
 *
 * Decrements the session reference count. If the count reaches zero,
 * the session is freed.
 *
 * Note: For the current single global session, this will only trigger
 * cleanup during module unload.
 */
void aql_pmu_put_session(struct aql_perf_session *session)
{
    if (session)
        aql_perf_session_put(session);
}

/**
 * aql_pmu_get_stats - Get AQL PMU statistics
 * @stats: Output buffer for statistics
 */
void aql_pmu_get_stats(struct aql_perf_stats *stats)
{
    if (!stats)
        return;

    aql_perf_get_stats(stats);

    /* Add session-specific stats if available */
    mutex_lock(&aql_pmu_mutex);
    if (global_aql_session) {
        atomic64_add(atomic64_read(&global_aql_session->stats.packets_submitted),
                     &stats->packets_submitted);
        atomic64_add(atomic64_read(&global_aql_session->stats.packets_completed),
                     &stats->packets_completed);
        atomic64_add(atomic64_read(&global_aql_session->stats.errors_total),
                     &stats->errors_total);
        atomic64_add(atomic64_read(&global_aql_session->stats.sessions_created),
                     &stats->sessions_created);
    }
    mutex_unlock(&aql_pmu_mutex);
}

EXPORT_SYMBOL_GPL(aql_pmu_init);
EXPORT_SYMBOL_GPL(aql_pmu_cleanup);
EXPORT_SYMBOL_GPL(aql_pmu_is_available);
EXPORT_SYMBOL_GPL(aql_pmu_event_init);
EXPORT_SYMBOL_GPL(aql_pmu_event_destroy);
EXPORT_SYMBOL_GPL(aql_pmu_event_start);
EXPORT_SYMBOL_GPL(aql_pmu_event_stop);
EXPORT_SYMBOL_GPL(aql_pmu_event_read);
EXPORT_SYMBOL_GPL(aql_pmu_get_session);
EXPORT_SYMBOL_GPL(aql_pmu_put_session);
EXPORT_SYMBOL_GPL(aql_pmu_get_stats);