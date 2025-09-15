/*
 * aql_packet_ops.c - AQL Packet Operations Implementation
 *
 * This module implements AQL packet creation, submission, and measurement
 * management for performance counter integration.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/atomic.h>
#include <linux/delay.h>
#include <linux/workqueue.h>
#include <linux/completion.h>

#include "aql_perf.h"
#include "pmu_stub.h"
#include "aql_c/counter_registry.h"
#include "aql_c/packet_generation.h"
#include "aql_c/pm4_packets.h"

/* External KFD functions */
extern int kfd_ioctl_submit_ib_packet(struct file *filep, struct kfd_process *p,
                                     uint32_t gpu_id, const uint32_t* packet,
                                     size_t ib_len);

/**
 * aql_perf_find_gpu_index - Find GPU index in session from GPU ID
 * @session: AQL performance session
 * @gpu_id: GPU ID to find
 *
 * Returns: GPU index, or -1 if not found
 */
static int aql_perf_find_gpu_index(struct aql_perf_session *session, uint32_t gpu_id)
{
    uint32_t i;

    if (!session || !session->gpu_ids)
        return -1;

    for (i = 0; i < session->num_gpus; i++) {
        if (session->gpu_ids[i] == gpu_id)
            return i;
    }

    return -1;
}

/**
 * aql_perf_get_counter_select - Get counter select value for a counter ID
 * @session: AQL performance session (contains arch for event ID lookup)
 * @gpu_id: Target GPU ID
 * @counter_id: Counter ID (from event->attr.config, maps to counter_id_t)
 *
 * Returns: Architecture-specific hardware event ID for counter programming
 */
__attribute__((unused))
static uint64_t aql_perf_get_counter_select(struct aql_perf_session *session, uint32_t gpu_id, uint32_t counter_id)
{
    const counter_def_t *counter;
    uint32_t event_id;
    int gpu_idx;

    if (!session || !session->archs) {
        aql_err("No architectures available for counter lookup");
        return 0;
    }

    /* Find GPU index */
    gpu_idx = aql_perf_find_gpu_index(session, gpu_id);
    if (gpu_idx < 0 || !session->archs[gpu_idx]) {
        aql_err("GPU %u not found or no architecture available", gpu_id);
        return 0;
    }

    /* Look up counter definition by ID */
    counter = lookup_counter_by_id((counter_id_t)counter_id);
    if (!counter) {
        aql_err("Counter ID %u not found in registry", counter_id);
        return 0;
    }

    /* Get architecture-specific event ID */
    event_id = lookup_event_id(counter, (const arch_t *)session->archs[gpu_idx]);
    if (event_id == 0) {
        aql_err("No event mapping for counter %s on GPU %u architecture", counter->name, gpu_id);
        return 0;
    }

    aql_debug("GPU %u: Counter %s (ID=%u) maps to event_id=0x%x", gpu_id, counter->name, counter_id, event_id);
    return event_id;
}

/* Packet Creation Functions */

/**
 * aql_perf_create_start_packet - Create START performance counting packet using PM4 generation
 * @measurement: Measurement to create START packet for (will allocate counter)
 * @out_pm4_buffer: Output PM4 buffer (caller must call pm4_buffer_destroy)
 *
 * Returns: 0 on success, negative error code on failure
 *
 * This function:
 * 1. Atomically allocates a counter from the appropriate hardware block
 * 2. Builds counter_info_t and counter_collection_t structures
 * 3. Calls generate_start_packet() to create PM4 command buffer
 * 4. Stores allocated counter pointer in measurement->allocated_counter
 */
int aql_perf_create_start_packet(struct aql_measurement *measurement,
                                 pm4_buffer_t **out_pm4_buffer)
{
    struct aql_perf_session *session;
    arch_t *arch;
    int gpu_idx;
    const counter_def_t *counter_def;
    block_info_t *block;
    uint32_t event_id;
    counter_reg_info_t *allocated_counter;
    counter_info_t counter_info;
    counter_collection_t collection;
    pm4_buffer_t *pm4_buffer;
    int ret;

    if (!measurement || !measurement->session || !out_pm4_buffer) {
        aql_err("[PMU] aql_perf_create_start_packet: Invalid parameters");
        return -EINVAL;
    }

    session = measurement->session;

    /* Find GPU architecture */
    gpu_idx = aql_perf_find_gpu_index(session, measurement->gpu_id);
    if (gpu_idx < 0 || !session->archs || !session->archs[gpu_idx]) {
        aql_err("[PMU] Session %llu: GPU %u not found or no architecture available",
                session->session_id, measurement->gpu_id);
        return -ENODEV;
    }
    arch = session->archs[gpu_idx];

    aql_debug("[PMU] aql_perf_create_start_packet: GPU %u, counter_id=%u",
              measurement->gpu_id, measurement->counter_id);

    /* Look up counter definition */
    counter_def = lookup_counter_by_id((counter_id_t)measurement->counter_id);
    if (!counter_def) {
        aql_err("[PMU] Counter ID %u not found in registry", measurement->counter_id);
        return -EINVAL;
    }

    /* Get architecture-specific event ID */
    event_id = lookup_event_id(counter_def, arch);
    if (event_id == 0) {
        aql_err("[PMU] No event mapping for counter %s", counter_def->name);
        return -ENOTSUPP;
    }

    /* Get block from architecture */
    block = arch->block_map.blocks[counter_def->hw_block];
    if (!block) {
        aql_err("[PMU] Block %u not found in architecture", counter_def->hw_block);
        return -EINVAL;
    }

    aql_info("[PMU] generate_start_packet: Allocating counter from block=%s, event_id=0x%x",
             block->name, event_id);

    /* Atomically allocate a counter from the block */
    allocated_counter = aql_counter_try_allocate(block, event_id, measurement->event);
    if (!allocated_counter) {
        aql_err("[PMU] Failed to allocate counter from block %s (all busy)", block->name);
        return -EBUSY;
    }

    /* Build counter_info_t structure */
    ret = aql_build_counter_info(measurement->counter_id, arch, allocated_counter,
                                  &counter_info, &block);
    if (ret) {
        aql_err("[PMU] Failed to build counter info: %d", ret);
        aql_counter_release(allocated_counter);
        return ret;
    }

    /* Build counter_collection_t structure */
    memset(&collection, 0, sizeof(collection));
    collection.counters = &counter_info;
    collection.counter_count = 1;
    collection.gpu_memory_addr = (uint64_t)(uintptr_t)allocated_counter->allocation.data_buffer->gpu_addr;
    collection.memory_size = PAGE_SIZE;

    aql_info("[PMU] generate_start_packet: counter_index=%u, event_id=0x%x, gpu_addr=0x%llx",
             counter_info.counter_index, counter_info.event_id, collection.gpu_memory_addr);

    /* Validate counter collection */
    ret = validate_counter_collection(arch, &collection);
    if (ret) {
        aql_err("[PMU] Counter collection validation failed: %d", ret);
        aql_counter_release(allocated_counter);
        return ret;
    }

    /* Create PM4 buffer */
    pm4_buffer = pm4_buffer_create(256, GFP_KERNEL);
    if (!pm4_buffer) {
        aql_err("[PMU] Failed to create PM4 buffer");
        aql_counter_release(allocated_counter);
        return -ENOMEM;
    }

    /* Generate start packet */
    ret = generate_start_packet(pm4_buffer, arch, &collection);
    if (ret) {
        aql_err("[PMU] generate_start_packet failed: %d", ret);
        pm4_buffer_destroy(pm4_buffer);
        aql_counter_release(allocated_counter);
        return ret;
    }

    aql_info("[PMU] generate_start_packet: PM4 buffer created, size=%zu DWORDs", pm4_buffer->size);

    /* Store allocated counter in measurement */
    measurement->allocated_counter = allocated_counter;

    *out_pm4_buffer = pm4_buffer;
    return 0;
}

/**
 * aql_perf_create_read_packet - Create READ counter values packet using PM4 generation
 * @measurement: Measurement to create READ packet for (must have allocated counter)
 * @out_pm4_buffer: Output PM4 buffer (caller must call pm4_buffer_destroy)
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_create_read_packet(struct aql_measurement *measurement,
                                pm4_buffer_t **out_pm4_buffer)
{
    struct aql_perf_session *session;
    arch_t *arch;
    int gpu_idx;
    counter_info_t counter_info;
    counter_collection_t collection;
    block_info_t *block;
    pm4_buffer_t *pm4_buffer;
    int ret;

    aql_info("[PMU] aql_perf_create_read_packet: Entry for GPU %u, counter_id=%u",
             measurement ? measurement->gpu_id : 0,
             measurement ? measurement->counter_id : 0);

    if (!measurement || !measurement->session || !out_pm4_buffer) {
        aql_err("[PMU] aql_perf_create_read_packet: Invalid parameters");
        return -EINVAL;
    }

    if (!measurement->allocated_counter) {
        aql_err("[PMU] aql_perf_create_read_packet: No counter allocated for measurement");
        return -EINVAL;
    }

    aql_info("[PMU] aql_perf_create_read_packet: GPU %u, allocated_counter=%p",
             measurement->gpu_id, measurement->allocated_counter);

    session = measurement->session;

    /* Find GPU architecture */
    gpu_idx = aql_perf_find_gpu_index(session, measurement->gpu_id);
    if (gpu_idx < 0 || !session->archs || !session->archs[gpu_idx]) {
        aql_err("[PMU] Session %llu: GPU %u not found or no architecture available",
                session->session_id, measurement->gpu_id);
        return -ENODEV;
    }
    arch = session->archs[gpu_idx];

    /* Build counter_info_t from allocated counter */
    ret = aql_build_counter_info(measurement->counter_id, arch,
                                  measurement->allocated_counter,
                                  &counter_info, &block);
    if (ret) {
        aql_err("[PMU] Failed to build counter info: %d", ret);
        return ret;
    }

    /* Build counter_collection_t */
    memset(&collection, 0, sizeof(collection));
    collection.counters = &counter_info;
    collection.counter_count = 1;
    collection.gpu_memory_addr = (uint64_t)(uintptr_t)measurement->allocated_counter->allocation.data_buffer->gpu_addr;
    collection.memory_size = PAGE_SIZE;

    aql_info("[PMU] generate_read_packet: counter_index=%u, gpu_addr=0x%llx",
             counter_info.counter_index, collection.gpu_memory_addr);

    /* Create PM4 buffer */
    pm4_buffer = pm4_buffer_create(256, GFP_KERNEL);
    if (!pm4_buffer) {
        aql_err("[PMU] Failed to create PM4 buffer");
        return -ENOMEM;
    }

    /* Generate read packet */
    ret = generate_read_packet(pm4_buffer, arch, &collection);
    if (ret) {
        aql_err("[PMU] generate_read_packet failed: %d", ret);
        pm4_buffer_destroy(pm4_buffer);
        return ret;
    }

    aql_info("[PMU] generate_read_packet: PM4 buffer created, size=%zu DWORDs", pm4_buffer->size);

    *out_pm4_buffer = pm4_buffer;
    return 0;
}

/**
 * aql_perf_create_end_packet - Create END performance counting packet using PM4 generation
 * @measurement: Measurement to create END packet for
 * @out_pm4_buffer: Output PM4 buffer (caller must call pm4_buffer_destroy)
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_create_end_packet(struct aql_measurement *measurement,
                               pm4_buffer_t **out_pm4_buffer)
{
    struct aql_perf_session *session;
    arch_t *arch;
    int gpu_idx;
    pm4_buffer_t *pm4_buffer;
    int ret;

    if (!measurement || !measurement->session || !out_pm4_buffer) {
        aql_err("[PMU] aql_perf_create_end_packet: Invalid parameters");
        return -EINVAL;
    }

    session = measurement->session;

    /* Find GPU architecture */
    gpu_idx = aql_perf_find_gpu_index(session, measurement->gpu_id);
    if (gpu_idx < 0 || !session->archs || !session->archs[gpu_idx]) {
        aql_err("[PMU] Session %llu: GPU %u not found or no architecture available",
                session->session_id, measurement->gpu_id);
        return -ENODEV;
    }
    arch = session->archs[gpu_idx];

    aql_info("[PMU] generate_stop_packet: GPU %u", measurement->gpu_id);

    /* Create PM4 buffer */
    pm4_buffer = pm4_buffer_create(256, GFP_KERNEL);
    if (!pm4_buffer) {
        aql_err("[PMU] Failed to create PM4 buffer");
        return -ENOMEM;
    }

    /* Generate stop packet */
    ret = generate_stop_packet(pm4_buffer, arch);
    if (ret) {
        aql_err("[PMU] generate_stop_packet failed: %d", ret);
        pm4_buffer_destroy(pm4_buffer);
        return ret;
    }

    aql_info("[PMU] generate_stop_packet: PM4 buffer created, size=%zu DWORDs", pm4_buffer->size);

    *out_pm4_buffer = pm4_buffer;
    return 0;
}

/**
 * aql_perf_submit_pm4_packet - Submit PM4 packet buffer to GPU via KFD
 * @session: AQL performance session
 * @gpu_id: Target GPU ID
 * @pm4_buffer: PM4 buffer to submit
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_submit_pm4_packet(struct aql_perf_session *session,
                               uint32_t gpu_id,
                               pm4_buffer_t *pm4_buffer)
{
    int ret;
    size_t ib_len;

    if (!session || !pm4_buffer || !pm4_buffer->data) {
        aql_err("[PMU] aql_perf_submit_pm4_packet: Invalid parameters");
        return -EINVAL;
    }

    /* IB length is already in DWORDs from pm4_buffer->size */
    ib_len = pm4_buffer->size;

    aql_info("[PMU] kfd_ioctl_submit_ib_packet: gpu_id=%u, buffer=%p, size=%zu DWORDs (%zu bytes)",
             gpu_id, pm4_buffer->data, pm4_buffer->size, pm4_buffer->size * 4);

    /* Submit PM4 buffer via KFD (ib_len is in DWORDs) */
    ret = kfd_ioctl_submit_ib_packet(session->kfd_file, session->process,
                                    gpu_id, pm4_buffer->data, ib_len);

    if (ret) {
        struct aql_error_context error = {
            .severity = AQL_ERROR_GPU_FAULT,
            .gpu_id = gpu_id,
            .error_code = ret,
            .timestamp = ktime_get()
        };

        snprintf(error.error_msg, sizeof(error.error_msg),
                "Failed to submit PM4 packet: %d", ret);

        aql_err("[PMU] Session %llu: %s", session->session_id, error.error_msg);
        aql_perf_handle_error(session, &error);
        aql_perf_inc_stat(AQL_STAT_ERRORS_TOTAL);
        return ret;
    }

    aql_info("[PMU] kfd_ioctl_submit_ib_packet: SUCCESS");

    aql_perf_inc_stat(AQL_STAT_PACKETS_SUBMITTED);
    aql_perf_inc_stat(AQL_STAT_PACKETS_COMPLETED);

    return 0;
}

/* Measurement Management */

/**
 * aql_perf_measurement_create - Create new measurement
 * @session: AQL performance session
 * @gpu_id: Target GPU ID
 * @event: Associated perf event
 *
 * Returns: New measurement or ERR_PTR on error
 */
struct aql_measurement *aql_perf_measurement_create(struct aql_perf_session *session,
                                                    uint32_t gpu_id,
                                                    struct perf_event *event)
{
    struct aql_measurement *measurement;
    int gpu_idx;

    if (!session || !event) {
        aql_err("Invalid parameters for measurement creation");
        return ERR_PTR(-EINVAL);
    }

    gpu_idx = aql_perf_find_gpu_index(session, gpu_id);
    if (gpu_idx < 0) {
        aql_err("Session %llu: GPU %u not found in session",
                session->session_id, gpu_id);
        return ERR_PTR(-ENODEV);
    }

    measurement = kzalloc(sizeof(*measurement), GFP_KERNEL);
    if (!measurement) {
        aql_err("Session %llu: Failed to allocate measurement structure",
                session->session_id);
        return ERR_PTR(-ENOMEM);
    }

    INIT_LIST_HEAD(&measurement->list);
    measurement->session = session;
    measurement->gpu_id = gpu_id;
    measurement->event = event;
    measurement->state = MEASUREMENT_IDLE;
    measurement->counter_mask = 0x1; /* Default to first counter */
    measurement->counter_id = (uint32_t)event->attr.config; /* Use event config as counter ID */
    measurement->last_counter_value = 0;
    measurement->allocated_counter = NULL; /* No counter allocated yet */

    /* Initialize work queue support */
    measurement->work_queue = alloc_workqueue("aql_gpu_%u", WQ_MEM_RECLAIM | WQ_HIGHPRI, 1, gpu_id);
    if (!measurement->work_queue) {
        aql_err("Session %llu: Failed to create work queue for GPU %u",
                session->session_id, gpu_id);
        kfree(measurement);
        return ERR_PTR(-ENOMEM);
    }

    spin_lock_init(&measurement->cache_lock);
    measurement->cached_counter_value = 0;
    measurement->cache_valid = false;

    aql_debug("Session %llu: Created measurement for GPU %u",
              session->session_id, gpu_id);

    return measurement;
}

/**
 * aql_perf_measurement_start - Start measurement with PM4 packet submission
 * @measurement: Measurement to start
 *
 * Returns: 0 on success, negative error code on failure
 *
 * This function:
 * 1. Creates START packet (which atomically allocates a counter)
 * 2. Submits PM4 buffer directly via kfd_ioctl_submit_ib_packet
 * 3. Destroys PM4 buffer after submission
 */
int aql_perf_measurement_start(struct aql_measurement *measurement)
{
    struct aql_perf_session *session;
    pm4_buffer_t *pm4_buffer = NULL;
    unsigned long flags;
    int ret;

    if (!measurement || !measurement->session) {
        aql_err("[PMU] Invalid measurement for start");
        return -EINVAL;
    }

    session = measurement->session;

    mutex_lock(&session->session_mutex);

    if (session->state != SESSION_ACTIVE) {
        aql_err("[PMU] Session %llu: Cannot start measurement, session not active",
                session->session_id);
        ret = -EINVAL;
        goto unlock_session;
    }

    /* Add to active measurements list */
    spin_lock_irqsave(&session->measurement_lock, flags);

    if (measurement->state != MEASUREMENT_IDLE) {
        aql_err("[PMU] Session %llu: Measurement already active for GPU %u",
                session->session_id, measurement->gpu_id);
        ret = -EBUSY;
        spin_unlock_irqrestore(&session->measurement_lock, flags);
        goto unlock_session;
    }

    measurement->state = MEASUREMENT_STARTING;
    list_add_tail(&measurement->list, &session->active_measurements);
    atomic_inc(&session->active_gpu_count);
    measurement->start_time = ktime_get();

    spin_unlock_irqrestore(&session->measurement_lock, flags);

    /* Create START packet (allocates counter atomically) */
    ret = aql_perf_create_start_packet(measurement, &pm4_buffer);
    if (ret) {
        aql_err("[PMU] Session %llu: Failed to create START packet for GPU %u: %d",
                session->session_id, measurement->gpu_id, ret);
        goto cleanup_measurement;
    }

    /* Submit PM4 buffer directly */
    ret = aql_perf_submit_pm4_packet(session, measurement->gpu_id, pm4_buffer);
    if (ret) {
        aql_err("[PMU] Session %llu: Failed to submit START packet for GPU %u: %d",
                session->session_id, measurement->gpu_id, ret);
        /* Counter was allocated, need to release it */
        if (measurement->allocated_counter) {
            aql_counter_release(measurement->allocated_counter);
            measurement->allocated_counter = NULL;
        }
        pm4_buffer_destroy(pm4_buffer);
        goto cleanup_measurement;
    }

    /* Cleanup PM4 buffer */
    pm4_buffer_destroy(pm4_buffer);

    /* Update state to active */
    spin_lock_irqsave(&session->measurement_lock, flags);
    measurement->state = MEASUREMENT_ACTIVE;
    spin_unlock_irqrestore(&session->measurement_lock, flags);

    aql_info("[PMU] Session %llu: Started measurement for GPU %u",
             session->session_id, measurement->gpu_id);

    mutex_unlock(&session->session_mutex);
    return 0;

cleanup_measurement:
    spin_lock_irqsave(&session->measurement_lock, flags);
    list_del(&measurement->list);
    measurement->state = MEASUREMENT_ERROR;
    atomic_dec(&session->active_gpu_count);
    spin_unlock_irqrestore(&session->measurement_lock, flags);

unlock_session:
    mutex_unlock(&session->session_mutex);
    return ret;
}

/**
 * aql_perf_measurement_stop - Stop measurement and release counter
 * @measurement: Measurement to stop
 *
 * Returns: 0 on success, negative error code on failure
 *
 * This function:
 * 1. Creates END packet using PM4 generation
 * 2. Submits PM4 buffer directly
 * 3. Releases allocated counter back to free pool
 */
int aql_perf_measurement_stop(struct aql_measurement *measurement)
{
    struct aql_perf_session *session;
    pm4_buffer_t *pm4_buffer = NULL;
    unsigned long flags;
    int ret;

    if (!measurement || !measurement->session) {
        aql_err("[PMU] Invalid measurement for stop");
        return -EINVAL;
    }

    /* If called from atomic context, use the atomic version instead */
    if (in_atomic() || irqs_disabled()) {
        aql_debug("[PMU] Stop called from atomic context, using async version");
        return aql_perf_measurement_stop_atomic(measurement);
    }

    session = measurement->session;

    mutex_lock(&session->session_mutex);

    spin_lock_irqsave(&session->measurement_lock, flags);

    if (measurement->state != MEASUREMENT_ACTIVE) {
        aql_debug("[PMU] Session %llu: Measurement for GPU %u not active",
                  session->session_id, measurement->gpu_id);
        spin_unlock_irqrestore(&session->measurement_lock, flags);
        mutex_unlock(&session->session_mutex);
        return 0; /* Already stopped */
    }

    measurement->state = MEASUREMENT_STOPPING;
    spin_unlock_irqrestore(&session->measurement_lock, flags);

    /* Create END packet */
    ret = aql_perf_create_end_packet(measurement, &pm4_buffer);
    if (ret) {
        aql_err("[PMU] Session %llu: Failed to create END packet for GPU %u: %d",
                session->session_id, measurement->gpu_id, ret);
        goto cleanup;
    }

    /* Submit PM4 buffer */
    ret = aql_perf_submit_pm4_packet(session, measurement->gpu_id, pm4_buffer);
    if (ret) {
        aql_err("[PMU] Session %llu: Failed to submit END packet for GPU %u: %d",
                session->session_id, measurement->gpu_id, ret);
        pm4_buffer_destroy(pm4_buffer);
        goto cleanup;
    }

    /* Cleanup PM4 buffer */
    pm4_buffer_destroy(pm4_buffer);

    /* Release allocated counter */
    if (measurement->allocated_counter) {
        aql_counter_release(measurement->allocated_counter);
        measurement->allocated_counter = NULL;
    }

    /* Remove from active measurements */
    spin_lock_irqsave(&session->measurement_lock, flags);
    list_del(&measurement->list);
    measurement->state = MEASUREMENT_IDLE;
    atomic_dec(&session->active_gpu_count);
    spin_unlock_irqrestore(&session->measurement_lock, flags);

    aql_info("[PMU] Session %llu: Stopped measurement for GPU %u",
             session->session_id, measurement->gpu_id);

    mutex_unlock(&session->session_mutex);
    return 0;

cleanup:
    spin_lock_irqsave(&session->measurement_lock, flags);
    list_del(&measurement->list);
    measurement->state = MEASUREMENT_ERROR;
    atomic_dec(&session->active_gpu_count);
    spin_unlock_irqrestore(&session->measurement_lock, flags);

    mutex_unlock(&session->session_mutex);
    return ret;
}

/**
 * aql_perf_measurement_read - Read counter value from measurement
 * @measurement: Measurement to read
 *
 * Returns: Counter value
 */
uint64_t aql_perf_measurement_read(struct aql_measurement *measurement)
{
    struct aql_perf_session *session;
    uint64_t *result_buffer;
    uint64_t counter_value;
    int gpu_idx;
    int ret;

    aql_info("[PMU] READ_SYNC: Entry for GPU %u",
             measurement ? measurement->gpu_id : 0);

    if (!measurement || !measurement->session) {
        aql_err("[PMU] READ_SYNC: Invalid measurement for read");
        return 0;
    }

    session = measurement->session;

    aql_info("[PMU] READ_SYNC: GPU %u, state=%d, allocated_counter=%p",
             measurement->gpu_id, measurement->state, measurement->allocated_counter);

    if (measurement->state != MEASUREMENT_ACTIVE) {
        aql_info("[PMU] READ_SYNC: GPU %u not active (state=%d), returning cached value %llu",
                 measurement->gpu_id, measurement->state, measurement->last_counter_value);
        return measurement->last_counter_value;
    }

    gpu_idx = aql_perf_find_gpu_index(session, measurement->gpu_id);
    if (gpu_idx < 0) {
        aql_err("[PMU] READ_SYNC: GPU %u not found for read", measurement->gpu_id);
        return measurement->last_counter_value;
    }

    /* Create and submit READ packet */
    pm4_buffer_t *pm4_buffer = NULL;
    aql_info("[PMU] READ_SYNC: GPU %u, creating READ packet", measurement->gpu_id);
    ret = aql_perf_create_read_packet(measurement, &pm4_buffer);
    if (ret) {
        aql_err("[PMU] READ_SYNC: GPU %u, failed to create READ packet: %d",
                measurement->gpu_id, ret);
        return measurement->last_counter_value;
    }

    aql_info("[PMU] READ_SYNC: GPU %u, submitting PM4 READ packet (size=%zu DWORDs)",
             measurement->gpu_id, pm4_buffer ? pm4_buffer->size : (size_t)0);
    ret = aql_perf_submit_pm4_packet(session, measurement->gpu_id, pm4_buffer);
    pm4_buffer_destroy(pm4_buffer);
    if (ret) {
        aql_err("[PMU] READ_SYNC: GPU %u, failed to submit READ packet: %d",
                measurement->gpu_id, ret);
        return measurement->last_counter_value;
    }

    aql_info("[PMU] READ_SYNC: GPU %u, READ packet submitted successfully", measurement->gpu_id);

    /* Read result from GPU memory buffer in allocated counter */
    result_buffer = NULL;
    if (measurement->allocated_counter &&
        measurement->allocated_counter->allocation.data_buffer) {
        result_buffer = (uint64_t*)measurement->allocated_counter->allocation.data_buffer->cpu_addr;
        aql_info("[PMU] READ_SYNC: GPU %u, data_buffer CPU addr=%p, GPU addr=0x%llx",
                 measurement->gpu_id, result_buffer,
                 (unsigned long long)measurement->allocated_counter->allocation.data_buffer->gpu_addr);
    } else {
        aql_warn("[PMU] READ_SYNC: GPU %u, no data buffer available (allocated_counter=%p)",
                 measurement->gpu_id, measurement->allocated_counter);
    }

    counter_value = result_buffer ? *result_buffer : -1;

    aql_info("[PMU] READ_SYNC: GPU %u, read counter_value=%llu from buffer (buffer=%p)",
             measurement->gpu_id, counter_value, result_buffer);

    /* Update cached value */
    measurement->last_counter_value = counter_value;

    aql_info("[PMU] READ_SYNC: GPU %u, updated last_counter_value=%llu, returning",
             measurement->gpu_id, counter_value);

    return counter_value;
}

/**
 * aql_perf_measurement_destroy - Destroy measurement and ensure counter release
 * @measurement: Measurement to destroy
 */
void aql_perf_measurement_destroy(struct aql_measurement *measurement)
{
    bool is_atomic_ctx;

    if (!measurement)
        return;

    /* Check if we're in atomic context (e.g., called from pmu_stub_del) */
    is_atomic_ctx = in_atomic() || irqs_disabled();

    if (is_atomic_ctx) {
        aql_debug("[PMU] Destroy called from atomic context, deferring cleanup");

        /* Mark for destruction after async work completes */
        measurement->pending_destroy = true;

        /* Queue async STOP work if measurement is still active */
        if (measurement->state == MEASUREMENT_ACTIVE) {
            aql_debug("[PMU] Queueing async STOP for pending destroy");
            aql_perf_measurement_stop_atomic(measurement);
        }
        /* If not active, the async work will never run, so we leak the measurement.
         * This is acceptable for now since it only happens in error paths. */

        return;
    }

    /* Non-atomic context: safe to do full synchronous cleanup */

    /* Ensure measurement is stopped */
    if (measurement->state == MEASUREMENT_ACTIVE) {
        aql_perf_measurement_stop(measurement);
    }

    /* Ensure counter is released if still allocated */
    if (measurement->allocated_counter) {
        aql_warn("[PMU] Measurement still has allocated counter during destroy, releasing");
        aql_counter_release(measurement->allocated_counter);
        measurement->allocated_counter = NULL;
    }

    /* Clean up work queue */
    if (measurement->work_queue) {
        flush_workqueue(measurement->work_queue);
        destroy_workqueue(measurement->work_queue);
        measurement->work_queue = NULL;
    }

    aql_debug("[PMU] Session %llu: Destroyed measurement for GPU %u",
              measurement->session ? measurement->session->session_id : 0,
              measurement->gpu_id);

    kfree(measurement);
}

/* Work Queue Implementation for Atomic Context Support */

/**
 * aql_work_handler - Work queue handler for deferred AQL operations
 * @work: Work item containing operation details
 */
void aql_work_handler(struct work_struct *work)
{
    struct aql_work_item *work_item = container_of(work, struct aql_work_item, work);
    struct aql_measurement *measurement = work_item->measurement;
    int result = 0;
    unsigned long flags;

    aql_debug("Starting work handler for GPU %u, op_type=%d",
              measurement->gpu_id, work_item->op_type);

    switch (work_item->op_type) {
    case AQL_WORK_START:
        result = aql_perf_measurement_start(measurement);
        break;

    case AQL_WORK_STOP:
        result = aql_perf_measurement_stop(measurement);
        break;

    case AQL_WORK_READ:
        {
            aql_info("[PMU] WORK_READ: Starting for GPU %u", measurement->gpu_id);
            uint64_t counter_value = aql_perf_measurement_read(measurement);
            aql_info("[PMU] WORK_READ: GPU %u, read returned counter_value=%llu",
                     measurement->gpu_id, counter_value);

            /* Update cached value with fresh read */
            spin_lock_irqsave(&measurement->cache_lock, flags);
            uint64_t old_cached = measurement->cached_counter_value;
            bool was_valid = measurement->cache_valid;
            measurement->cached_counter_value = counter_value;
            measurement->cache_valid = true;
            spin_unlock_irqrestore(&measurement->cache_lock, flags);

            aql_info("[PMU] WORK_READ: GPU %u, updated cache: old=%llu (valid=%d) -> new=%llu (valid=1)",
                     measurement->gpu_id, old_cached, was_valid, counter_value);
            result = 0; /* Read operations always succeed if we get here */
        }
        break;

    default:
        aql_err("Unknown work operation type: %d", work_item->op_type);
        result = -EINVAL;
        break;
    }

    work_item->result = result;

    /* Signal completion if waiting */
    if (work_item->completion) {
        complete(work_item->completion);
    }

    aql_debug("Completed work handler for GPU %u, op_type=%d, result=%d",
              measurement->gpu_id, work_item->op_type, result);

    /* Check if measurement should be destroyed after this async work */
    if (measurement->pending_destroy && work_item->op_type == AQL_WORK_STOP) {
        aql_debug("[PMU] Async STOP complete, destroying measurement as requested");

        /* Release counter if still allocated */
        if (measurement->allocated_counter) {
            aql_counter_release(measurement->allocated_counter);
            measurement->allocated_counter = NULL;
        }

        /* Can't call destroy_workqueue from work handler running on that queue.
         * Mark queue as NULL and leak it. */
        measurement->work_queue = NULL;
        aql_warn("[PMU] Leaking workqueue in async destroy path");

        /* Free the measurement structure */
        kfree(measurement);
        aql_debug("[PMU] Measurement freed in async destroy path");
    }
}

/**
 * aql_create_work_item - Create and schedule work item
 * @measurement: Target measurement
 * @op_type: Operation type to perform
 *
 * Returns: Work item or ERR_PTR on error
 */
struct aql_work_item *aql_create_work_item(struct aql_measurement *measurement,
                                          enum aql_work_op_type op_type)
{
    struct aql_work_item *work_item;

    if (!measurement || !measurement->work_queue) {
        aql_err("Invalid measurement or work queue");
        return ERR_PTR(-EINVAL);
    }

    work_item = kzalloc(sizeof(*work_item), GFP_ATOMIC);
    if (!work_item) {
        aql_err("Failed to allocate work item");
        return ERR_PTR(-ENOMEM);
    }

    INIT_WORK(&work_item->work, aql_work_handler);
    work_item->measurement = measurement;
    work_item->op_type = op_type;
    work_item->completion = NULL;
    work_item->result = 0;

    return work_item;
}

/**
 * aql_perf_measurement_start_atomic - Start measurement from atomic context
 * @measurement: Measurement to start
 *
 * Returns: 0 on success (work scheduled), negative error code on failure
 */
int aql_perf_measurement_start_atomic(struct aql_measurement *measurement)
{
    struct aql_work_item *work_item;

    if (!measurement) {
        return -EINVAL;
    }

    work_item = aql_create_work_item(measurement, AQL_WORK_START);
    if (IS_ERR(work_item)) {
        return PTR_ERR(work_item);
    }

    /* Schedule work without waiting */
    if (!queue_work(measurement->work_queue, &work_item->work)) {
        kfree(work_item);
        return -EBUSY; /* Work already queued */
    }

    aql_debug("Scheduled START work for GPU %u from atomic context",
              measurement->gpu_id);
    return 0;
}

/**
 * aql_perf_measurement_stop_atomic - Stop measurement from atomic context
 * @measurement: Measurement to stop
 *
 * Returns: 0 on success (work scheduled), negative error code on failure
 */
int aql_perf_measurement_stop_atomic(struct aql_measurement *measurement)
{
    struct aql_work_item *work_item;

    if (!measurement) {
        return -EINVAL;
    }

    work_item = aql_create_work_item(measurement, AQL_WORK_STOP);
    if (IS_ERR(work_item)) {
        return PTR_ERR(work_item);
    }

    /* Schedule work without waiting */
    if (!queue_work(measurement->work_queue, &work_item->work)) {
        kfree(work_item);
        return -EBUSY; /* Work already queued */
    }

    aql_debug("Scheduled STOP work for GPU %u from atomic context",
              measurement->gpu_id);
    return 0;
}

/**
 * aql_perf_measurement_read_atomic - Read measurement from atomic context
 * @measurement: Measurement to read
 *
 * Returns: Cached counter value (may schedule background refresh)
 */
uint64_t aql_perf_measurement_read_atomic(struct aql_measurement *measurement)
{
    struct aql_work_item *work_item;
    unsigned long flags;
    uint64_t cached_value = 0;
    bool cache_was_valid = false;

    aql_info("[PMU] READ_ATOMIC: Entry for GPU %u",
             measurement ? measurement->gpu_id : 0);

    if (!measurement) {
        aql_warn("[PMU] READ_ATOMIC: NULL measurement");
        return 0;
    }

    aql_info("[PMU] READ_ATOMIC: GPU %u, state=%d, allocated_counter=%p",
             measurement->gpu_id, measurement->state, measurement->allocated_counter);

    /* Return cached value immediately */
    spin_lock_irqsave(&measurement->cache_lock, flags);
    cache_was_valid = measurement->cache_valid;
    if (measurement->cache_valid) {
        cached_value = measurement->cached_counter_value;
    }
    spin_unlock_irqrestore(&measurement->cache_lock, flags);

    aql_info("[PMU] READ_ATOMIC: GPU %u, cache_valid=%d, cached_value=%llu",
             measurement->gpu_id, cache_was_valid, cached_value);

    /* Schedule background refresh of cached value */
    work_item = aql_create_work_item(measurement, AQL_WORK_READ);
    if (!IS_ERR(work_item)) {
        if (!queue_work(measurement->work_queue, &work_item->work)) {
            aql_info("[PMU] READ_ATOMIC: GPU %u, READ work already queued",
                     measurement->gpu_id);
            kfree(work_item); /* Work already queued */
        } else {
            aql_info("[PMU] READ_ATOMIC: GPU %u, scheduled READ work for background refresh",
                     measurement->gpu_id);
        }
    } else {
        aql_warn("[PMU] READ_ATOMIC: GPU %u, failed to create work item: %ld",
                 measurement->gpu_id, PTR_ERR(work_item));
    }

    aql_info("[PMU] READ_ATOMIC: GPU %u, returning cached_value=%llu",
             measurement->gpu_id, cached_value);
    return cached_value;
}

EXPORT_SYMBOL_GPL(aql_perf_create_start_packet);
EXPORT_SYMBOL_GPL(aql_perf_create_read_packet);
EXPORT_SYMBOL_GPL(aql_perf_create_end_packet);
EXPORT_SYMBOL_GPL(aql_perf_submit_pm4_packet);
EXPORT_SYMBOL_GPL(aql_perf_measurement_create);
EXPORT_SYMBOL_GPL(aql_perf_measurement_start);
EXPORT_SYMBOL_GPL(aql_perf_measurement_stop);
EXPORT_SYMBOL_GPL(aql_perf_measurement_read);
EXPORT_SYMBOL_GPL(aql_perf_measurement_destroy);
EXPORT_SYMBOL_GPL(aql_perf_measurement_start_atomic);
EXPORT_SYMBOL_GPL(aql_perf_measurement_stop_atomic);
EXPORT_SYMBOL_GPL(aql_perf_measurement_read_atomic);
EXPORT_SYMBOL_GPL(aql_work_handler);
EXPORT_SYMBOL_GPL(aql_create_work_item);