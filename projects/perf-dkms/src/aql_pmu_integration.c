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
#include <linux/rcupdate.h>
#include <linux/spinlock.h>

#include "aql_perf.h"
#include "amdgpu_pmu.h"
#include "pmu_dimension.h"

/* Global AQL session - RCU-protected for safe access from atomic context */
static struct aql_perf_session __rcu *global_aql_session = NULL;
static struct mutex aql_pmu_mutex; /* For writes and non-atomic operations */
static bool aql_feature_available = false;

/* Global workqueue - shared by all measurements */
static struct workqueue_struct *aql_global_workqueue = NULL;

/**
 * aql_get_global_workqueue - Get the global workqueue for AQL operations
 *
 * Returns: Pointer to global workqueue, or NULL if not initialized
 */
struct workqueue_struct *aql_get_global_workqueue(void)
{
	return aql_global_workqueue;
}

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

	/* Create global workqueue - shared by all measurements */
	aql_global_workqueue =
		alloc_workqueue("aql_pmu", WQ_MEM_RECLAIM | WQ_HIGHPRI | WQ_UNBOUND, 0);
	if (!aql_global_workqueue) {
		aql_err("Failed to create global workqueue");
		return -ENOMEM;
	}

	aql_info("Created global workqueue for all measurements");

	/* Create global AQL session */
	struct aql_perf_session *session = aql_perf_session_create();
	if (IS_ERR(session)) {
		ret = PTR_ERR(session);
		aql_err("Failed to create global AQL session: %d", ret);
		destroy_workqueue(aql_global_workqueue);
		aql_global_workqueue = NULL;
		return ret;
	}

	/* Initialize the session */
	ret = aql_perf_session_initialize(session);
	if (ret) {
		aql_err("Failed to initialize global AQL session: %d", ret);
		aql_perf_session_put(session);
		destroy_workqueue(aql_global_workqueue);
		aql_global_workqueue = NULL;
		return ret;
	}

	/* Publish session via RCU */
	mutex_lock(&aql_pmu_mutex);
	rcu_assign_pointer(global_aql_session, session);
	aql_feature_available = true;
	mutex_unlock(&aql_pmu_mutex);

	aql_info("AQL PMU integration initialized successfully with %u GPUs", session->num_gpus);

	return 0;
}

/**
 * aql_pmu_flush_all_measurements - Flush global workqueue before cleanup
 *
 * This function ensures all pending work items complete before session cleanup.
 * Must be called AFTER timer is cancelled but BEFORE session is released.
 *
 * IMPORTANT: This prevents use-after-free by ensuring no work handlers are running
 * or queued when we free session resources.
 */
void aql_pmu_flush_all_measurements(void)
{
	mutex_lock(&aql_pmu_mutex);

	if (!aql_global_workqueue) {
		mutex_unlock(&aql_pmu_mutex);
		return;
	}

	pmu_info("Flushing global workqueue");
	flush_workqueue(aql_global_workqueue);

	mutex_unlock(&aql_pmu_mutex);

	pmu_info("Global workqueue flushed");
}

/**
 * aql_pmu_cleanup - Cleanup AQL PMU integration
 */
void aql_pmu_cleanup(void)
{
	struct aql_perf_session *session;

	aql_info("Cleaning up AQL PMU integration");

	/* Clear session pointer under mutex */
	mutex_lock(&aql_pmu_mutex);
	session = rcu_dereference_protected(global_aql_session, lockdep_is_held(&aql_pmu_mutex));
	rcu_assign_pointer(global_aql_session, NULL);
	aql_feature_available = false;
	mutex_unlock(&aql_pmu_mutex);

	/* Wait for all RCU readers to finish before releasing session */
	if (session) {
		synchronize_rcu();
		aql_perf_session_put(session);
	}

	/* Destroy global workqueue - safe because session is already released
     * and timer is already cancelled (done in pmu_main.c before calling this) */
	if (aql_global_workqueue) {
		aql_info("Destroying global workqueue");
		destroy_workqueue(aql_global_workqueue);
		aql_global_workqueue = NULL;
	}

	aql_info("AQL PMU integration cleanup complete");
}

/**
 * aql_pmu_is_available - Check if AQL PMU feature is available
 *
 * Returns: true if available, false otherwise
 */
bool aql_pmu_is_available(void)
{
	struct aql_perf_session *session;
	bool available;

	rcu_read_lock();
	session = rcu_dereference(global_aql_session);
	available = aql_feature_available && session && (session->state == SESSION_ACTIVE);
	rcu_read_unlock();

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
/**
 * aql_pmu_get_gpu_id_for_event_locked - Get actual GPU ID for a perf event (mutex must be held)
 * @event: Perf event to map
 *
 * Internal version that assumes aql_pmu_mutex is already held by caller.
 * Uses rcu_dereference_protected since mutex provides necessary protection.
 *
 * Returns: GPU device ID (e.g., 7410), or U32_MAX on error
 */
static uint32_t aql_pmu_get_gpu_id_for_event_locked(struct perf_event *event)
{
	struct aql_perf_session *session;
	uint32_t cpu;
	uint32_t gpu_id;

	if (!event) {
		return U32_MAX;
	}

	session = rcu_dereference_protected(global_aql_session, lockdep_is_held(&aql_pmu_mutex));
	cpu = event->cpu;

	aql_debug("get_gpu_id_for_event: cpu=%d, num_gpus=%u", cpu,
		  session ? session->num_gpus : 0);

	if (!session || session->num_gpus == 0) {
		aql_err("get_gpu_id_for_event: no session or no GPUs");
		return U32_MAX;
	}

	/* For now, use simple CPU-to-GPU mapping */
	if (cpu == -1) {
		/* Use first healthy GPU for CPU-wide events */
		for (uint32_t i = 0; i < session->num_gpus; i++) {
			gpu_id = session->gpu_ids[i];
			if (!aql_perf_is_gpu_disabled(session, gpu_id)) {
				aql_debug("Selected GPU %u for CPU-wide event", gpu_id);
				return gpu_id;
			}
		}
	} else {
		/* Map CPU to GPU using modulo */
		uint32_t healthy_count = aql_perf_get_healthy_gpu_count(session);

		if (healthy_count == 0) {
			return U32_MAX;
		}

		uint32_t gpu_index = cpu % healthy_count;
		uint32_t healthy_idx = 0;

		for (uint32_t i = 0; i < session->num_gpus; i++) {
			gpu_id = session->gpu_ids[i];
			if (!aql_perf_is_gpu_disabled(session, gpu_id)) {
				if (healthy_idx == gpu_index) {
					aql_debug("Selected GPU %u for CPU %u", gpu_id, cpu);
					return gpu_id;
				}
				healthy_idx++;
			}
		}
	}

	return U32_MAX;
}

/**
 * aql_pmu_get_gpu_id_for_event - Get actual GPU ID for a perf event (public API)
 * @event: Perf event to map
 *
 * Maps the event's CPU affinity to an actual GPU device ID. This is the
 * proper way to get a GPU ID from a perf_event, as it accounts for the
 * AQL session's actual GPU IDs (which may not be sequential 0,1,2...).
 *
 * Returns: GPU device ID (e.g., 7410), or U32_MAX on error
 */
uint32_t aql_pmu_get_gpu_id_for_event(struct perf_event *event)
{
	uint32_t gpu_id;

	mutex_lock(&aql_pmu_mutex);
	gpu_id = aql_pmu_get_gpu_id_for_event_locked(event);
	mutex_unlock(&aql_pmu_mutex);

	return gpu_id;
}

/* Internal helper - calls locked version since we already hold mutex in event_init */
static uint32_t aql_pmu_select_gpu(struct perf_event *event)
{
	return aql_pmu_get_gpu_id_for_event_locked(event);
}

/**
 * aql_pmu_event_init - Initialize event with AQL support
 * @event: Perf event to initialize
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_pmu_event_init(struct perf_event *event, const struct pmu_dimension_coords *dims)
{
	struct aql_measurement *measurement;
	uint32_t gpu_id;

	aql_debug("ENTRY: aql_pmu_event_init");

	if (!aql_pmu_should_use_hardware(event)) {
		aql_debug("Not using hardware, returning -EOPNOTSUPP");
		return -EOPNOTSUPP; /* Use simulation instead */
	}

	aql_debug("Before mutex_lock(&aql_pmu_mutex)");
	mutex_lock(&aql_pmu_mutex);

	struct aql_perf_session *session =
		rcu_dereference_protected(global_aql_session, lockdep_is_held(&aql_pmu_mutex));
	aql_debug("After mutex_lock(&aql_pmu_mutex), session=%p", session);

	if (!session) {
		aql_debug("Global AQL session is NULL, falling back to simulation");
		mutex_unlock(&aql_pmu_mutex);
		return -EOPNOTSUPP;
	}

	aql_debug("About to check session state, session=%p", session);
	if (session->state != SESSION_ACTIVE) {
		aql_debug("AQL session state is %d (not active), falling back to simulation",
			  session->state);
		mutex_unlock(&aql_pmu_mutex);
		return -EOPNOTSUPP;
	}
	aql_debug("Session is active, state=%d", session->state);

	/* Select GPU for this event */
	aql_debug("Before aql_pmu_select_gpu");
	gpu_id = aql_pmu_select_gpu(event);
	aql_debug("After aql_pmu_select_gpu, gpu_id=%u", gpu_id);
	if (gpu_id == U32_MAX) {
		aql_debug("No suitable GPU found for event, falling back to simulation");
		mutex_unlock(&aql_pmu_mutex);
		return -EOPNOTSUPP;
	}

	/* Create AQL measurement */
	aql_debug("Before aql_perf_measurement_create for GPU %u", gpu_id);
	measurement = aql_perf_measurement_create(session, gpu_id, event);
	aql_debug("After aql_perf_measurement_create, measurement=%p", measurement);
	if (IS_ERR(measurement)) {
		aql_err("Failed to create AQL measurement: %ld", PTR_ERR(measurement));
		mutex_unlock(&aql_pmu_mutex);
		return PTR_ERR(measurement);
	}

	/* Store dimension information in measurement if specified */
	if (dims && dims->valid) {
		measurement->target_dims = *dims;
		/* If aggregate flag is set, aggregate across all dimensions.
         * Otherwise, monitor only the specific dimension. */
		if (dims->aggregate) {
			measurement->dimension_specific = false; /* Aggregate mode */
			aql_debug("GPU %u: aggregate mode (all dimensions)", gpu_id);
		} else {
			measurement->dimension_specific = true; /* Dimension-specific mode */
			aql_debug("GPU %u: dimension-specific se=%u sa=%u wgp=%u", gpu_id, dims->se,
				  dims->sa, dims->wgp);
		}
	} else {
		measurement->dimension_specific = false;
		aql_debug("GPU %u: initialized (aggregated)", gpu_id);
	}

	/* Store measurement in event's hardware config */
	event->hw.config_base = (unsigned long)measurement;

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

	/* Clear config_base so add/del/start/stop know the event is destroyed */
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
 * __aql_pmu_event_read_internal - Internal read function without logging
 * @event: Perf event to read
 *
 * This is the quiet version used by high-frequency sampling paths (e.g., timer).
 * No logging to avoid flooding kernel log at 50Hz sampling rate.
 *
 * Returns: Counter value
 */
static uint64_t __aql_pmu_event_read_internal(struct perf_event *event)
{
	struct aql_measurement *measurement;

	if (!event->hw.config_base) {
		return 0;
	}

	measurement = (struct aql_measurement *)event->hw.config_base;

	/* Use atomic version to return cached value and schedule refresh */
	return aql_perf_measurement_read_atomic(measurement);
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

	if (!event->hw.config_base) {
		return 0;
	}

	measurement = (struct aql_measurement *)event->hw.config_base;

	/* Call internal read function */
	counter_value = __aql_pmu_event_read_internal(event);

	/* Log results */
	aql_debug("GPU %u: read=%llu", measurement->gpu_id, counter_value);
	aql_info("[PMU] ========== aql_pmu_event_read RETURNING: %llu (0x%llx) ==========",
		 counter_value, counter_value);

	return counter_value;
}

/**
 * aql_pmu_event_read_sync - Synchronously read AQL hardware counter
 * @event: Perf event to read
 *
 * Performs a synchronous (blocking) read of the hardware counter.
 * This waits for the actual hardware read to complete before returning.
 * Use this for final reads (e.g., in del callback) where accuracy is critical.
 *
 * Returns: Current counter value
 */
uint64_t aql_pmu_event_read_sync(struct perf_event *event)
{
	struct aql_measurement *measurement;
	uint64_t counter_value;

	if (!event->hw.config_base) {
		return 0;
	}

	measurement = (struct aql_measurement *)event->hw.config_base;

	/* Use synchronous version to wait for actual hardware read */
	counter_value = aql_perf_measurement_read(measurement);

	aql_debug("GPU %u: read_sync=%llu", measurement->gpu_id, counter_value);

	return counter_value;
}

/**
 * aql_pmu_get_gpu_count - Get number of available GPUs
 *
 * Returns: Number of GPUs in the global AQL session, or 0 if not initialized
 */
int aql_pmu_get_gpu_count(void)
{
	struct aql_perf_session *session;
	int count;

	rcu_read_lock();
	session = rcu_dereference(global_aql_session);
	count = session ? session->num_gpus : 0;
	rcu_read_unlock();

	return count;
}

/**
 * aql_pmu_get_session - Get reference to global AQL session
 *
 * Retrieves the global AQL session and increments its reference count.
 * Caller must call aql_pmu_put_session() when done.
 *
 * IMPORTANT: This function is safe to call from atomic context (e.g., timer
 * callbacks, hard IRQ handlers) because it uses RCU read-side critical
 * sections instead of mutexes.
 *
 * RCU Protocol:
 * - Readers (this function) use rcu_read_lock/unlock for atomic-safe access
 * - Writers (init/cleanup) use rcu_assign_pointer and synchronize_rcu
 * - Reference count is acquired with refcount_inc_not_zero to avoid races
 *   with session destruction
 *
 * Returns: Global AQL session with incremented refcount, or NULL
 */
struct aql_perf_session *aql_pmu_get_session(void)
{
	struct aql_perf_session *session;

	rcu_read_lock();
	session = rcu_dereference(global_aql_session);
	if (session && !aql_perf_session_get(session))
		session = NULL; /* Session is being destroyed, don't use it */
	rcu_read_unlock();

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
	struct aql_perf_session *session;

	if (!stats)
		return;

	aql_perf_get_stats(stats);

	/* Add session-specific stats if available */
	rcu_read_lock();
	session = rcu_dereference(global_aql_session);
	if (session) {
		atomic64_add(atomic64_read(&session->stats.packets_submitted),
			     &stats->packets_submitted);
		atomic64_add(atomic64_read(&session->stats.packets_completed),
			     &stats->packets_completed);
		atomic64_add(atomic64_read(&session->stats.errors_total), &stats->errors_total);
		atomic64_add(atomic64_read(&session->stats.sessions_created),
			     &stats->sessions_created);
	}
	rcu_read_unlock();
}

EXPORT_SYMBOL_GPL(aql_get_global_workqueue);
EXPORT_SYMBOL_GPL(aql_pmu_init);
EXPORT_SYMBOL_GPL(aql_pmu_cleanup);
EXPORT_SYMBOL_GPL(aql_pmu_flush_all_measurements);
EXPORT_SYMBOL_GPL(aql_pmu_is_available);
EXPORT_SYMBOL_GPL(aql_pmu_event_init);
EXPORT_SYMBOL_GPL(aql_pmu_event_destroy);
EXPORT_SYMBOL_GPL(aql_pmu_event_start);
EXPORT_SYMBOL_GPL(aql_pmu_event_stop);
EXPORT_SYMBOL_GPL(aql_pmu_event_read);
EXPORT_SYMBOL_GPL(aql_pmu_get_session);
EXPORT_SYMBOL_GPL(aql_pmu_put_session);
EXPORT_SYMBOL_GPL(aql_pmu_get_stats);