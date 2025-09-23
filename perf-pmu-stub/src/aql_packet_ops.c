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

/* External KFD functions */
extern int kfd_ioctl_submit_ib_packet(struct file *filep, struct kfd_process *p,
                                     uint32_t gpu_id, const uint32_t* packet,
                                     size_t ib_len);

/* Helper function to find GPU index in session */
static int aql_perf_find_gpu_index(struct aql_perf_session *session, uint32_t gpu_id)
{
    uint32_t i;

    for (i = 0; i < session->num_gpus; i++) {
        if (session->gpu_ids[i] == gpu_id)
            return i;
    }

    return -1;
}

/**
 * aql_perf_get_counter_select - Get counter select value for a counter ID
 * @counter_id: Counter ID (from event->attr.config)
 *
 * Returns: Counter select value for hardware programming
 */
static uint64_t aql_perf_get_counter_select(uint32_t counter_id)
{
    struct counter_descriptor *desc;

    desc = aql_perf_find_counter_descriptor(counter_id);
    if (desc) {
        return desc->counter_select;
    }

    /* Default to SQ_WAVES if counter not found */
    aql_debug("Counter ID %u not found, defaulting to SQ_WAVES", counter_id);
    return GFX12_PERF_SEL_SQ_WAVES;
}

/* Packet Creation Functions */

/**
 * aql_perf_create_start_packet - Create START performance counting packet
 * @session: AQL performance session
 * @gpu_id: Target GPU ID
 * @counter_id: Counter ID to use (from event config)
 * @packet: Output packet structure
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_create_start_packet(struct aql_perf_session *session,
                                 uint32_t gpu_id,
                                 uint32_t counter_id,
                                 struct aql_perf_packet *packet)
{
    int gpu_idx;
    uint64_t counter_select;

    if (!session || !packet) {
        aql_err("Invalid parameters for start packet creation");
        return -EINVAL;
    }

    gpu_idx = aql_perf_find_gpu_index(session, gpu_id);
    if (gpu_idx < 0) {
        aql_err("Session %llu: GPU %u not found in session",
                session->session_id, gpu_id);
        return -ENODEV;
    }

    memset(packet, 0, sizeof(*packet));

    packet->packet_type = AQL_PERF_PACKET_START;
    packet->gpu_id = gpu_id;

    /* Get counter select value for the requested counter */
    counter_select = aql_perf_get_counter_select(counter_id);

    packet->counter_select = counter_select;
    packet->result_address = (uint64_t)session->counter_data[gpu_idx].gpu_addr;

    /* Build AQL packet payload for GFX12 counter start */
    packet->packet_data[0] = 0x10000000; /* PM4 header - TYPE3 packet */
    packet->packet_data[1] = 0x00000001; /* PM4 opcode for counter start */
    packet->packet_data[2] = (uint32_t)counter_select; /* Counter select register */
    packet->packet_data[3] = 0x00000001; /* Counter mode - continuous */
    packet->packet_data[4] = (uint32_t)(packet->result_address & 0xFFFFFFFF); /* Result address low */
    packet->packet_data[5] = (uint32_t)(packet->result_address >> 32); /* Result address high */
    packet->packet_data[6] = 0x00000000; /* Reserved */
    packet->packet_data[7] = 0x00000000; /* Reserved */

    packet->packet_size = 8 * sizeof(uint32_t);

    aql_debug("Session %llu: Created START packet for GPU %u, counter=0x%llx",
              session->session_id, gpu_id, counter_select);

    return 0;
}

/**
 * aql_perf_create_read_packet - Create READ counter values packet
 * @session: AQL performance session
 * @gpu_id: Target GPU ID
 * @counter_id: Counter ID to use (from event config)
 * @packet: Output packet structure
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_create_read_packet(struct aql_perf_session *session,
                                uint32_t gpu_id,
                                uint32_t counter_id,
                                struct aql_perf_packet *packet)
{
    int gpu_idx;
    uint64_t counter_select;

    if (!session || !packet) {
        aql_err("Invalid parameters for read packet creation");
        return -EINVAL;
    }

    gpu_idx = aql_perf_find_gpu_index(session, gpu_id);
    if (gpu_idx < 0) {
        aql_err("Session %llu: GPU %u not found in session",
                session->session_id, gpu_id);
        return -ENODEV;
    }

    memset(packet, 0, sizeof(*packet));

    packet->packet_type = AQL_PERF_PACKET_READ;
    packet->gpu_id = gpu_id;

    /* Get counter select value for the requested counter */
    counter_select = aql_perf_get_counter_select(counter_id);

    packet->counter_select = counter_select;
    packet->result_address = (uint64_t)session->counter_data[gpu_idx].gpu_addr;

    /* Build AQL packet payload for GFX12 counter read */
    packet->packet_data[0] = 0x10000000; /* PM4 header - TYPE3 packet */
    packet->packet_data[1] = 0x00000002; /* PM4 opcode for counter read */
    packet->packet_data[2] = (uint32_t)counter_select; /* Counter select register */
    packet->packet_data[3] = (uint32_t)(packet->result_address & 0xFFFFFFFF); /* Result address low */
    packet->packet_data[4] = (uint32_t)(packet->result_address >> 32); /* Result address high */
    packet->packet_data[5] = 0x00000000; /* Reserved */
    packet->packet_data[6] = 0x00000000; /* Reserved */
    packet->packet_data[7] = 0x00000000; /* Reserved */

    packet->packet_size = 8 * sizeof(uint32_t);

    aql_debug("Session %llu: Created READ packet for GPU %u, counter=0x%llx",
              session->session_id, gpu_id, counter_select);

    return 0;
}

/**
 * aql_perf_create_end_packet - Create END performance counting packet
 * @session: AQL performance session
 * @gpu_id: Target GPU ID
 * @counter_id: Counter ID to use (from event config)
 * @packet: Output packet structure
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_create_end_packet(struct aql_perf_session *session,
                               uint32_t gpu_id,
                               uint32_t counter_id,
                               struct aql_perf_packet *packet)
{
    int gpu_idx;
    uint64_t counter_select;

    if (!session || !packet) {
        aql_err("Invalid parameters for end packet creation");
        return -EINVAL;
    }

    gpu_idx = aql_perf_find_gpu_index(session, gpu_id);
    if (gpu_idx < 0) {
        aql_err("Session %llu: GPU %u not found in session",
                session->session_id, gpu_id);
        return -ENODEV;
    }

    memset(packet, 0, sizeof(*packet));

    packet->packet_type = AQL_PERF_PACKET_END;
    packet->gpu_id = gpu_id;

    /* Get counter select value for the requested counter */
    counter_select = aql_perf_get_counter_select(counter_id);

    packet->counter_select = counter_select;
    packet->result_address = (uint64_t)session->counter_data[gpu_idx].gpu_addr;

    /* Build AQL packet payload for GFX12 counter end */
    packet->packet_data[0] = 0x10000000; /* PM4 header - TYPE3 packet */
    packet->packet_data[1] = 0x00000003; /* PM4 opcode for counter end */
    packet->packet_data[2] = (uint32_t)counter_select; /* Counter select register */
    packet->packet_data[3] = (uint32_t)(packet->result_address & 0xFFFFFFFF); /* Result address low */
    packet->packet_data[4] = (uint32_t)(packet->result_address >> 32); /* Result address high */
    packet->packet_data[5] = 0x00000000; /* Reserved */
    packet->packet_data[6] = 0x00000000; /* Reserved */
    packet->packet_data[7] = 0x00000000; /* Reserved */

    packet->packet_size = 8 * sizeof(uint32_t);

    aql_debug("Session %llu: Created END packet for GPU %u, counter=0x%llx",
              session->session_id, gpu_id, counter_select);

    return 0;
}

/**
 * aql_perf_submit_packet - Submit AQL packet to GPU
 * @session: AQL performance session
 * @packet: Packet to submit
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_submit_packet(struct aql_perf_session *session,
                           struct aql_perf_packet *packet)
{
    int ret;
    const char *packet_type_str;

    if (!session || !packet) {
        aql_err("Invalid parameters for packet submission");
        return -EINVAL;
    }

    /* Convert packet type to string for debugging */
    switch (packet->packet_type) {
    case AQL_PERF_PACKET_START:
        packet_type_str = "START";
        break;
    case AQL_PERF_PACKET_READ:
        packet_type_str = "READ";
        break;
    case AQL_PERF_PACKET_END:
        packet_type_str = "END";
        break;
    default:
        packet_type_str = "UNKNOWN";
        break;
    }

    aql_debug("Session %llu: Submitting %s packet to GPU %u",
              session->session_id, packet_type_str, packet->gpu_id);

    /* Submit packet via KFD */
    ret = kfd_ioctl_submit_ib_packet(session->kfd_file, session->process,
                                    packet->gpu_id, packet->packet_data,
                                    packet->packet_size);

    if (ret) {
        struct aql_error_context error = {
            .severity = AQL_ERROR_GPU_FAULT,
            .gpu_id = packet->gpu_id,
            .error_code = ret,
            .timestamp = ktime_get()
        };

        snprintf(error.error_msg, sizeof(error.error_msg),
                "Failed to submit %s packet: %d", packet_type_str, ret);

        aql_err("Session %llu: %s", session->session_id, error.error_msg);
        aql_perf_handle_error(session, &error);
        aql_perf_inc_stat(AQL_STAT_ERRORS_TOTAL);
        return ret;
    }

    aql_debug("Session %llu: Successfully submitted %s packet to GPU %u",
              session->session_id, packet_type_str, packet->gpu_id);

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
 * aql_perf_measurement_start - Start measurement
 * @measurement: Measurement to start
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_measurement_start(struct aql_measurement *measurement)
{
    struct aql_perf_session *session;
    struct aql_perf_packet packet;
    unsigned long flags;
    int ret;

    if (!measurement || !measurement->session) {
        aql_err("Invalid measurement for start");
        return -EINVAL;
    }

    session = measurement->session;

    mutex_lock(&session->session_mutex);

    if (session->state != SESSION_ACTIVE) {
        aql_err("Session %llu: Cannot start measurement, session not active",
                session->session_id);
        ret = -EINVAL;
        goto unlock_session;
    }

    /* Add to active measurements list */
    spin_lock_irqsave(&session->measurement_lock, flags);

    if (measurement->state != MEASUREMENT_IDLE) {
        aql_err("Session %llu: Measurement already active for GPU %u",
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

    /* Create and submit START packet */
    ret = aql_perf_create_start_packet(session, measurement->gpu_id, measurement->counter_id, &packet);
    if (ret) {
        aql_err("Session %llu: Failed to create START packet for GPU %u: %d",
                session->session_id, measurement->gpu_id, ret);
        goto cleanup_measurement;
    }

    ret = aql_perf_submit_packet(session, &packet);
    if (ret) {
        aql_err("Session %llu: Failed to submit START packet for GPU %u: %d",
                session->session_id, measurement->gpu_id, ret);
        goto cleanup_measurement;
    }

    /* Update state to active */
    spin_lock_irqsave(&session->measurement_lock, flags);
    measurement->state = MEASUREMENT_ACTIVE;
    spin_unlock_irqrestore(&session->measurement_lock, flags);

    aql_info("Session %llu: Started measurement for GPU %u",
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
 * aql_perf_measurement_stop - Stop measurement
 * @measurement: Measurement to stop
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_measurement_stop(struct aql_measurement *measurement)
{
    struct aql_perf_session *session;
    struct aql_perf_packet packet;
    unsigned long flags;
    int ret;

    if (!measurement || !measurement->session) {
        aql_err("Invalid measurement for stop");
        return -EINVAL;
    }

    session = measurement->session;

    mutex_lock(&session->session_mutex);

    spin_lock_irqsave(&session->measurement_lock, flags);

    if (measurement->state != MEASUREMENT_ACTIVE) {
        aql_debug("Session %llu: Measurement for GPU %u not active",
                  session->session_id, measurement->gpu_id);
        spin_unlock_irqrestore(&session->measurement_lock, flags);
        mutex_unlock(&session->session_mutex);
        return 0; /* Already stopped */
    }

    measurement->state = MEASUREMENT_STOPPING;
    spin_unlock_irqrestore(&session->measurement_lock, flags);

    /* Create and submit END packet */
    ret = aql_perf_create_end_packet(session, measurement->gpu_id, measurement->counter_id, &packet);
    if (ret) {
        aql_err("Session %llu: Failed to create END packet for GPU %u: %d",
                session->session_id, measurement->gpu_id, ret);
        goto cleanup;
    }

    ret = aql_perf_submit_packet(session, &packet);
    if (ret) {
        aql_err("Session %llu: Failed to submit END packet for GPU %u: %d",
                session->session_id, measurement->gpu_id, ret);
        goto cleanup;
    }

    /* Remove from active measurements */
    spin_lock_irqsave(&session->measurement_lock, flags);
    list_del(&measurement->list);
    measurement->state = MEASUREMENT_IDLE;
    atomic_dec(&session->active_gpu_count);
    spin_unlock_irqrestore(&session->measurement_lock, flags);

    aql_info("Session %llu: Stopped measurement for GPU %u",
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
    struct aql_perf_packet packet;
    uint64_t *result_buffer;
    uint64_t counter_value;
    int gpu_idx;
    int ret;

    if (!measurement || !measurement->session) {
        aql_err("Invalid measurement for read");
        return 0;
    }

    session = measurement->session;

    if (measurement->state != MEASUREMENT_ACTIVE) {
        aql_debug("Session %llu: Measurement for GPU %u not active, returning cached value",
                  session->session_id, measurement->gpu_id);
        return measurement->last_counter_value;
    }

    gpu_idx = aql_perf_find_gpu_index(session, measurement->gpu_id);
    if (gpu_idx < 0) {
        aql_err("Session %llu: GPU %u not found for read",
                session->session_id, measurement->gpu_id);
        return measurement->last_counter_value;
    }

    /* Create and submit READ packet */
    ret = aql_perf_create_read_packet(session, measurement->gpu_id, measurement->counter_id, &packet);
    if (ret) {
        aql_err("Session %llu: Failed to create READ packet for GPU %u: %d",
                session->session_id, measurement->gpu_id, ret);
        return measurement->last_counter_value;
    }

    ret = aql_perf_submit_packet(session, &packet);
    if (ret) {
        aql_err("Session %llu: Failed to submit READ packet for GPU %u: %d",
                session->session_id, measurement->gpu_id, ret);
        return measurement->last_counter_value;
    }

    /* Read result from GPU memory */
    result_buffer = (uint64_t *)session->counter_data[gpu_idx].cpu_addr;
    counter_value = *result_buffer;

    /* Update cached value */
    measurement->last_counter_value = counter_value;

    aql_debug("Session %llu: Read counter value %llu from GPU %u",
              session->session_id, counter_value, measurement->gpu_id);

    return counter_value;
}

/**
 * aql_perf_measurement_destroy - Destroy measurement
 * @measurement: Measurement to destroy
 */
void aql_perf_measurement_destroy(struct aql_measurement *measurement)
{
    if (!measurement)
        return;

    /* Ensure measurement is stopped */
    if (measurement->state == MEASUREMENT_ACTIVE) {
        aql_perf_measurement_stop(measurement);
    }

    /* Clean up work queue */
    if (measurement->work_queue) {
        flush_workqueue(measurement->work_queue);
        destroy_workqueue(measurement->work_queue);
        measurement->work_queue = NULL;
    }

    aql_debug("Session %llu: Destroyed measurement for GPU %u",
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
            uint64_t counter_value = aql_perf_measurement_read(measurement);
            /* Update cached value with fresh read */
            spin_lock_irqsave(&measurement->cache_lock, flags);
            measurement->cached_counter_value = counter_value;
            measurement->cache_valid = true;
            spin_unlock_irqrestore(&measurement->cache_lock, flags);
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

    if (!measurement) {
        return 0;
    }

    /* Return cached value immediately */
    spin_lock_irqsave(&measurement->cache_lock, flags);
    if (measurement->cache_valid) {
        cached_value = measurement->cached_counter_value;
    }
    spin_unlock_irqrestore(&measurement->cache_lock, flags);

    /* Schedule background refresh of cached value */
    work_item = aql_create_work_item(measurement, AQL_WORK_READ);
    if (!IS_ERR(work_item)) {
        if (!queue_work(measurement->work_queue, &work_item->work)) {
            kfree(work_item); /* Work already queued */
        } else {
            aql_debug("Scheduled READ work for GPU %u from atomic context",
                      measurement->gpu_id);
        }
    }

    aql_debug("Returned cached value %llu for GPU %u", cached_value, measurement->gpu_id);
    return cached_value;
}

EXPORT_SYMBOL_GPL(aql_perf_create_start_packet);
EXPORT_SYMBOL_GPL(aql_perf_create_read_packet);
EXPORT_SYMBOL_GPL(aql_perf_create_end_packet);
EXPORT_SYMBOL_GPL(aql_perf_submit_packet);
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