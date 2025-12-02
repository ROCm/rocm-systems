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
#include "amdgpu_pmu.h"
#include "aql_c/counter_registry.h"
#include "aql_c/packet_generation.h"
#include "aql_c/pm4_packets.h"
#include "aql_c/arch_creator_common.h"

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

/* Counter Sharing Helper Functions */

/**
 * find_and_install_shared_counter - Find existing shared counter and install it atomically
 * @session: AQL performance session
 * @counter_id: Counter ID to search for
 * @measurement: Measurement to install the shared counter into
 *
 * Returns: Shared counter reference with incremented ref count, or NULL if not found
 *
 * This function atomically finds a shared counter, increments its reference count,
 * and installs it into the measurement while holding the lock. This prevents race
 * conditions where another thread could decrement the reference count to zero and
 * free the counter between finding it and using it.
 *
 * Note: Caller must release the reference when done using release_shared_counter()
 */
static struct shared_counter_ref *find_and_install_shared_counter(
    struct aql_perf_session *session,
    uint32_t counter_id,
    struct aql_measurement *measurement)
{
    struct shared_counter_ref *ref;
    unsigned long flags;

    spin_lock_irqsave(&session->shared_lock, flags);
    list_for_each_entry(ref, &session->shared_counters, list) {
        if (ref->counter_id == counter_id) {
            atomic_inc(&ref->ref_count);
            /* Install while holding lock - prevents race */
            measurement->shared_ref = ref;
            measurement->owns_counter = false;
            measurement->allocated_counter = ref->measurement->allocated_counter;
            spin_unlock_irqrestore(&session->shared_lock, flags);
            aql_info("[PMU] Found shared counter for counter_id=%u, new refcount=%d",
                     counter_id, atomic_read(&ref->ref_count));
            return ref;
        }
    }
    spin_unlock_irqrestore(&session->shared_lock, flags);
    return NULL;
}

/**
 * create_shared_counter - Create new shared counter reference
 * @session: AQL performance session
 * @counter_id: Counter ID this reference tracks
 * @measurement: Measurement that owns this counter allocation
 *
 * Returns: New shared counter reference, or NULL on allocation failure
 */
static struct shared_counter_ref *create_shared_counter(
    struct aql_perf_session *session,
    uint32_t counter_id,
    struct aql_measurement *measurement)
{
    struct shared_counter_ref *ref;
    unsigned long flags;

    ref = kzalloc(sizeof(*ref), GFP_KERNEL);
    if (!ref) {
        aql_err("[PMU] Failed to allocate shared counter reference");
        return NULL;
    }

    ref->counter_id = counter_id;
    ref->measurement = measurement;
    atomic_set(&ref->ref_count, 1);

    spin_lock_irqsave(&session->shared_lock, flags);
    list_add(&ref->list, &session->shared_counters);
    spin_unlock_irqrestore(&session->shared_lock, flags);

    aql_info("[PMU] Created shared counter for counter_id=%u, ref_count=1",
             counter_id);

    return ref;
}

/**
 * release_shared_counter - Release shared counter reference
 * @session: AQL performance session
 * @ref: Shared counter reference to release
 *
 * Decrements reference count. When count reaches zero, removes from list and frees.
 */
void release_shared_counter(
    struct aql_perf_session *session,
    struct shared_counter_ref *ref)
{
    unsigned long flags;
    int old_count;

    if (!ref)
        return;

    old_count = atomic_read(&ref->ref_count);

    if (atomic_dec_and_test(&ref->ref_count)) {
        spin_lock_irqsave(&session->shared_lock, flags);
        list_del(&ref->list);
        spin_unlock_irqrestore(&session->shared_lock, flags);

        aql_info("[PMU] Released shared counter for counter_id=%u (ref_count %d -> 0)",
                 ref->counter_id, old_count);
        kfree(ref);
    } else {
        aql_info("[PMU] Decremented shared counter for counter_id=%u (ref_count %d -> %d)",
                 ref->counter_id, old_count, atomic_read(&ref->ref_count));
    }
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
    ret = lookup_event_id(counter_def, arch, &event_id);
    if (ret < 0) {
        aql_err("[PMU] No event mapping for counter %s (err=%d)", counter_def->name, ret);
        return -ENOTSUPP;
    }

    /* Get block from architecture */
    block = arch->block_map.blocks[counter_def->hw_block];
    if (!block) {
        aql_err("[PMU] Block %u not found in architecture", counter_def->hw_block);
        return -EINVAL;
    }

    /* Check if another event already allocated this counter */
    struct shared_counter_ref *shared_ref = find_and_install_shared_counter(session, measurement->counter_id, measurement);
    if (shared_ref) {
        /* Reuse existing allocation - counter and references installed atomically */
        const counter_def_t *counter_def_for_log = lookup_counter_by_id((counter_id_t)measurement->counter_id);
        aql_info("[PMU] Sharing counter for %s (counter_id=%u, ref_count=%d)",
                 counter_def_for_log ? counter_def_for_log->name : "unknown",
                 measurement->counter_id,
                 atomic_read(&shared_ref->ref_count));

        /* No START packet needed - counter already started */
        *out_pm4_buffer = NULL;
        return 0;
    }

    aql_info("[PMU] generate_start_packet: Allocating counter from block=%s, event_id=0x%x",
             block->name, event_id);

    /* First event for this counter - allocate hardware counter */
    allocated_counter = aql_counter_try_allocate(block, event_id, measurement->event);
    if (!allocated_counter) {
        aql_err("[PMU] Failed to allocate counter from block %s (all busy)", block->name);
        return -EBUSY;
    }

    /* Create shared reference for this new allocation */
    shared_ref = create_shared_counter(session, measurement->counter_id, measurement);
    if (!shared_ref) {
        aql_err("[PMU] Failed to create shared counter reference");
        aql_counter_release(allocated_counter);
        measurement->allocated_counter = NULL;
        return -ENOMEM;
    }
    measurement->shared_ref = shared_ref;
    measurement->owns_counter = true;

    /* Build counter_info_t structure */
    ret = aql_build_counter_info(measurement->counter_id, arch, allocated_counter,
                                  &counter_info, &block);
    if (ret) {
        aql_err("[PMU] Failed to build counter info: %d", ret);
        aql_counter_release(allocated_counter);
        measurement->allocated_counter = NULL;
        measurement->owns_counter = false;
        release_shared_counter(session, measurement->shared_ref);
        measurement->shared_ref = NULL;
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
        measurement->allocated_counter = NULL;
        measurement->owns_counter = false;
        release_shared_counter(session, measurement->shared_ref);
        measurement->shared_ref = NULL;
        return ret;
    }

    /* Create PM4 buffer */
    pm4_buffer = pm4_buffer_create(256, GFP_KERNEL);
    if (!pm4_buffer) {
        aql_err("[PMU] Failed to create PM4 buffer");
        aql_counter_release(allocated_counter);
        measurement->allocated_counter = NULL;
        measurement->owns_counter = false;
        release_shared_counter(session, measurement->shared_ref);
        measurement->shared_ref = NULL;
        return -ENOMEM;
    }

    /* Generate start packet */
    ret = generate_start_packet(pm4_buffer, arch, &collection);
    if (ret) {
        aql_err("[PMU] generate_start_packet failed: %d", ret);
        pm4_buffer_destroy(pm4_buffer);
        aql_counter_release(allocated_counter);
        measurement->allocated_counter = NULL;
        measurement->owns_counter = false;
        release_shared_counter(session, measurement->shared_ref);
        measurement->shared_ref = NULL;
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
    counter_reg_info_t *counter;
    int ret;

    aql_info("[PMU] aql_perf_create_read_packet: Entry for GPU %u, counter_id=%u",
             measurement ? measurement->gpu_id : 0,
             measurement ? measurement->counter_id : 0);

    if (!measurement || !measurement->session || !out_pm4_buffer) {
        aql_err("[PMU] aql_perf_create_read_packet: Invalid parameters");
        return -EINVAL;
    }

    /* Get local copy of allocated_counter to prevent TOCTOU race */
    counter = measurement->allocated_counter;
    if (!counter) {
        aql_err("[PMU] aql_perf_create_read_packet: No counter allocated for measurement");
        return -EINVAL;
    }

    /* Check if data_buffer is still valid (may be NULL during cleanup) */
    if (!counter->allocation.data_buffer) {
        aql_debug("[PMU] aql_perf_create_read_packet: data_buffer freed (measurement being cleaned up)");
        return -ESHUTDOWN;
    }

    aql_info("[PMU] aql_perf_create_read_packet: GPU %u, allocated_counter=%p",
             measurement->gpu_id, counter);

    session = measurement->session;

    /* Find GPU architecture */
    gpu_idx = aql_perf_find_gpu_index(session, measurement->gpu_id);
    if (gpu_idx < 0 || !session->archs || !session->archs[gpu_idx]) {
        aql_err("[PMU] Session %llu: GPU %u not found or no architecture available",
                session->session_id, measurement->gpu_id);
        return -ENODEV;
    }
    arch = session->archs[gpu_idx];

    /* Build counter_info_t from allocated counter (use local copy) */
    ret = aql_build_counter_info(measurement->counter_id, arch,
                                  counter,
                                  &counter_info, &block);
    if (ret) {
        aql_err("[PMU] Failed to build counter info: %d", ret);
        return ret;
    }

    /* Build counter_collection_t (use local copy) */
    memset(&collection, 0, sizeof(collection));
    collection.counters = &counter_info;
    collection.counter_count = 1;
    collection.gpu_memory_addr = (uint64_t)(uintptr_t)counter->allocation.data_buffer->gpu_addr;
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
 * aql_measurement_release - Release callback for measurement kref
 * @kref: The kref being released
 *
 * This is called when the last reference to a measurement is dropped.
 * It performs final cleanup and frees the measurement structure.
 */
static void aql_measurement_release(struct kref *kref)
{
    struct aql_measurement *m = container_of(kref, struct aql_measurement, refcount);
    counter_reg_info_t *counter;
    unsigned long flags;

    aql_debug("[PMU] Releasing measurement for GPU %u (refcount reached 0)",
              m->gpu_id);

    /* Release counter if still allocated (use local copy to prevent TOCTOU) */
    if (m->owns_counter) {
        counter = m->allocated_counter;
        if (counter) {
            aql_debug("[PMU] Releasing owned counter during final release");
            aql_counter_release(counter);
            m->allocated_counter = NULL;
            m->owns_counter = false;
        }
    }

    /* Release shared counter reference if still held */
    if (m->shared_ref) {
        aql_debug("[PMU] Releasing shared counter reference during final release");
        release_shared_counter(m->session, m->shared_ref);
        m->shared_ref = NULL;
    }

    /* Remove from session's active list if still on it */
    if (m->session && !list_empty(&m->list)) {
        spin_lock_irqsave(&m->session->measurement_lock, flags);
        list_del_init(&m->list);
        atomic_dec(&m->session->active_gpu_count);
        spin_unlock_irqrestore(&m->session->measurement_lock, flags);
    }

    kfree(m);
    aql_debug("[PMU] Measurement freed in release callback");
}

/**
 * aql_measurement_get - Increment measurement reference count
 * @m: Measurement to reference
 */
void aql_measurement_get(struct aql_measurement *m)
{
    if (m) {
        kref_get(&m->refcount);
        aql_debug("[PMU] Measurement ref++ for GPU %u", m->gpu_id);
    }
}

/**
 * aql_measurement_put - Decrement measurement reference count
 * @m: Measurement to dereference
 *
 * When the last reference is dropped, aql_measurement_release() is called
 * to perform final cleanup and free the measurement.
 */
void aql_measurement_put(struct aql_measurement *m)
{
    if (m) {
        aql_debug("[PMU] Measurement ref-- for GPU %u", m->gpu_id);
        kref_put(&m->refcount, aql_measurement_release);
    }
}

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
    measurement->start_counter_value = 0;  /* Will be set when measurement starts */
    measurement->last_counter_value = 0;
    measurement->allocated_counter = NULL; /* No counter allocated yet */
    measurement->shared_ref = NULL; /* No shared reference yet */
    measurement->owns_counter = false; /* Not owning any counter yet */

    /* Initialize cache lock for atomic context handling */
    spin_lock_init(&measurement->cache_lock);
    measurement->cached_counter_value = 0;
    measurement->cache_valid = false;

    /* Initialize reference count - starts at 1 for the creator */
    kref_init(&measurement->refcount);

    aql_debug("Session %llu: Created measurement for GPU %u (initial refcount=1)",
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

    /* Create START packet (allocates counter atomically or shares existing) */
    ret = aql_perf_create_start_packet(measurement, &pm4_buffer);
    if (ret) {
        aql_err("[PMU] Session %llu: Failed to create START packet for GPU %u: %d",
                session->session_id, measurement->gpu_id, ret);
        goto cleanup_measurement;
    }

    /* If pm4_buffer is NULL, counter is shared and already started */
    if (pm4_buffer) {
        /* Submit PM4 buffer directly */
        ret = aql_perf_submit_pm4_packet(session, measurement->gpu_id, pm4_buffer);
        if (ret) {
            counter_reg_info_t *counter;
            aql_err("[PMU] Session %llu: Failed to submit START packet for GPU %u: %d",
                    session->session_id, measurement->gpu_id, ret);
            /* Counter was allocated, need to release it (use local copy to prevent TOCTOU) */
            if (measurement->owns_counter) {
                counter = measurement->allocated_counter;
                if (counter) {
                    aql_counter_release(counter);
                    measurement->allocated_counter = NULL;
                    measurement->owns_counter = false;
                }
            }
            /* Release shared reference */
            if (measurement->shared_ref) {
                release_shared_counter(session, measurement->shared_ref);
                measurement->shared_ref = NULL;
            }
            pm4_buffer_destroy(pm4_buffer);
            goto cleanup_measurement;
        }

        /* Cleanup PM4 buffer */
        pm4_buffer_destroy(pm4_buffer);

        /* KFD ioctl is synchronous - START packet has completed by this point.
         * The GPU has configured GRBM, enabled perfmon, set SQ control,
         * and configured counter registers. Safe to read baseline now. */
        aql_info("[PMU] Session %llu: START packet completed", session->session_id);

        aql_info("[PMU] Session %llu: Started measurement for GPU %u (owns_counter=true)",
                 session->session_id, measurement->gpu_id);
    } else {
        aql_info("[PMU] Session %llu: Started measurement for GPU %u (sharing counter, owns_counter=false)",
                 session->session_id, measurement->gpu_id);
    }

    /* Update state to active */
    spin_lock_irqsave(&session->measurement_lock, flags);
    measurement->state = MEASUREMENT_ACTIVE;
    spin_unlock_irqrestore(&session->measurement_lock, flags);

    /* Start background polling timer now that measurement is active.
     * This ensures timer only polls after measurements are ready. */
    amdgpu_pmu_start_timer();

    /* Read initial counter value for delta tracking.
     * GPU counters don't reset on START, so we need to track the baseline. */
    measurement->start_counter_value = aql_perf_measurement_read(measurement);
    aql_info("[PMU] Session %llu: GPU %u baseline counter value=%llu",
             session->session_id, measurement->gpu_id, measurement->start_counter_value);

    mutex_unlock(&session->session_mutex);
    return 0;

cleanup_measurement:
    spin_lock_irqsave(&session->measurement_lock, flags);
    list_del_init(&measurement->list);
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

    /* Only generate STOP packet if we own the counter */
    if (measurement->owns_counter) {
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

        /* Release allocated counter (we own it - use local copy to prevent TOCTOU) */
        counter_reg_info_t *counter = measurement->allocated_counter;
        if (counter) {
            aql_counter_release(counter);
            measurement->allocated_counter = NULL;
            measurement->owns_counter = false;
        }

        aql_info("[PMU] Session %llu: Stopped measurement for GPU %u (owned counter, released)",
                 session->session_id, measurement->gpu_id);
    } else {
        aql_info("[PMU] Session %llu: Stopped measurement for GPU %u (shared counter, not releasing)",
                 session->session_id, measurement->gpu_id);
    }

    /* Release shared reference (decrements ref count) */
    if (measurement->shared_ref) {
        release_shared_counter(session, measurement->shared_ref);
        measurement->shared_ref = NULL;
    }

    /* Remove from active measurements */
    spin_lock_irqsave(&session->measurement_lock, flags);
    list_del_init(&measurement->list);
    measurement->state = MEASUREMENT_IDLE;
    atomic_dec(&session->active_gpu_count);
    spin_unlock_irqrestore(&session->measurement_lock, flags);

    /* Stop timer if no more active measurements */
    amdgpu_pmu_stop_timer_if_idle();

    mutex_unlock(&session->session_mutex);
    return 0;

cleanup:
    spin_lock_irqsave(&session->measurement_lock, flags);
    list_del_init(&measurement->list);
    measurement->state = MEASUREMENT_ERROR;
    atomic_dec(&session->active_gpu_count);
    spin_unlock_irqrestore(&session->measurement_lock, flags);

    mutex_unlock(&session->session_mutex);
    return ret;
}

/**
 * aql_aggregate_counter_instances - Sum counter values across all hardware instances
 * @session: AQL session containing architecture information
 * @measurement: Measurement containing counter metadata
 * @result_buffer: Buffer containing per-instance counter values
 * @gpu_idx: Index of GPU in session
 *
 * Aggregates counter values by summing all hardware instances (SE x SA x WGP).
 * The number of instances is determined from the counter's hardware block
 * dimension information.
 *
 * Returns: Sum of all instance values
 */
static uint64_t aql_aggregate_counter_instances(
    struct aql_perf_session *session,
    struct aql_measurement *measurement,
    uint64_t *result_buffer,
    int gpu_idx)
{
    arch_t *arch = session->archs[gpu_idx];
    const counter_def_t *counter_def = lookup_counter_by_id((counter_id_t)measurement->counter_id);
    block_info_t *block = arch->block_map.blocks[counter_def->hw_block];
    uint64_t sum = 0;
    uint32_t num_instances = 1;

    /* Determine total number of instances based on block dimensions */
    for (size_t dim_idx = 0; dim_idx < block->dimension_count; dim_idx++) {
        num_instances *= block->dimensions[dim_idx].size;
    }

    aql_info("[PMU] Aggregating %u instances (block dimensions: count=%zu)",
             num_instances, block->dimension_count);

    /* Sum all instances and log individual values for debugging */
    for (uint32_t i = 0; i < num_instances; i++) {
        uint64_t value = result_buffer[i];
        sum += value;

        /* Log first 20 and last 5 values to see patterns */
        if (i < 20 || i >= (num_instances - 5)) {
            aql_info("[PMU]   instance[%u] = %llu", i, value);
        } else if (i == 20) {
            aql_info("[PMU]   ... (skipping middle instances) ...");
        }
    }

    aql_info("[PMU] Aggregated %u instances, total=%llu", num_instances, sum);
    aql_info("[PMU] ========== AGGREGATION RESULT: %llu (0x%llx) ==========", sum, sum);

    return sum;
}

/**
 * read_diagnostic_log_buffer - Log buffer contents for diagnostics
 * @measurement: Measurement containing buffer
 * @when: Description of when this is being logged (e.g., "BEFORE READ", "AFTER READ")
 */
static void read_diagnostic_log_buffer(struct aql_measurement *measurement, const char *when)
{
    counter_reg_info_t *counter = measurement->allocated_counter;
    uint64_t *buffer;

    if (!counter || !counter->allocation.data_buffer)
        return;

    buffer = (uint64_t*)counter->allocation.data_buffer->cpu_addr;

    if (strcmp(when, "BEFORE READ") == 0) {
        aql_info("[PMU] READ_SYNC: GPU %u, buffer %s: [0]=0x%llx [1]=0x%llx [2]=0x%llx [3]=0x%llx",
                 measurement->gpu_id, when, buffer[0], buffer[1], buffer[2], buffer[3]);
    } else {
        aql_info("[PMU] READ_SYNC: GPU %u, buffer %s: [0]=0x%llx [1]=0x%llx [2]=0x%llx [3]=0x%llx [4]=0x%llx [5]=0x%llx [6]=0x%llx [7]=0x%llx",
                 measurement->gpu_id, when, buffer[0], buffer[1], buffer[2], buffer[3],
                 buffer[4], buffer[5], buffer[6], buffer[7]);
    }
}

/**
 * read_submit_packet - Create and submit READ packet to GPU
 * @session: AQL performance session
 * @measurement: Measurement to read
 *
 * Returns: 0 on success, negative error code on failure
 */
static int read_submit_packet(struct aql_perf_session *session, struct aql_measurement *measurement)
{
    pm4_buffer_t *pm4_buffer = NULL;
    int ret;

    aql_info("[PMU] READ_SYNC: GPU %u, creating READ packet", measurement->gpu_id);
    ret = aql_perf_create_read_packet(measurement, &pm4_buffer);
    if (ret) {
        aql_err("[PMU] READ_SYNC: GPU %u, failed to create READ packet: %d",
                measurement->gpu_id, ret);
        return ret;
    }

    aql_info("[PMU] READ_SYNC: GPU %u, submitting PM4 READ packet (size=%zu DWORDs)",
             measurement->gpu_id, pm4_buffer ? pm4_buffer->size : (size_t)0);
    ret = aql_perf_submit_pm4_packet(session, measurement->gpu_id, pm4_buffer);
    pm4_buffer_destroy(pm4_buffer);
    if (ret) {
        aql_err("[PMU] READ_SYNC: GPU %u, failed to submit READ packet: %d",
                measurement->gpu_id, ret);
        return ret;
    }

    aql_info("[PMU] READ_SYNC: GPU %u, READ packet submitted successfully", measurement->gpu_id);
    return 0;
}

/**
 * read_get_result_buffer - Get result buffer from measurement
 * @measurement: Measurement containing buffer
 *
 * Returns: Pointer to result buffer, or NULL if unavailable
 */
static uint64_t *read_get_result_buffer(struct aql_measurement *measurement)
{
    counter_reg_info_t *counter = measurement->allocated_counter;
    uint64_t *result_buffer = NULL;

    if (counter && counter->allocation.data_buffer) {
        result_buffer = (uint64_t*)counter->allocation.data_buffer->cpu_addr;
        aql_info("[PMU] READ_SYNC: GPU %u, data_buffer CPU addr=%p, GPU addr=0x%llx",
                 measurement->gpu_id, result_buffer,
                 (unsigned long long)counter->allocation.data_buffer->gpu_addr);
    } else {
        aql_warn("[PMU] READ_SYNC: GPU %u, no data buffer available (allocated_counter=%p)",
                 measurement->gpu_id, counter);
    }

    return result_buffer;
}

/**
 * read_extract_counter_value - Extract counter value from result buffer
 * @session: AQL performance session
 * @measurement: Measurement to read
 * @result_buffer: Buffer containing counter results
 * @gpu_idx: GPU index in session
 *
 * Returns: Counter value
 */
static uint64_t read_extract_counter_value(struct aql_perf_session *session,
                                          struct aql_measurement *measurement,
                                          uint64_t *result_buffer,
                                          int gpu_idx)
{
    uint64_t counter_value;

    if (!result_buffer) {
        return -1;
    }

    if (measurement->dimension_specific) {
        /*
         * Dimension-specific counter: filter to return only the specific instance.
         * The GPU read packet collects all instances in a flat array.
         * Calculate the flat index for the target dimension and return only that value.
         */
        arch_t *arch = session->archs[gpu_idx];
        uint32_t flat_idx = encode_dimension_index(
            measurement->target_dims.se,
            measurement->target_dims.sa,
            measurement->target_dims.wgp,
            arch->num_sa,
            arch->num_wgp_per_sa
        );

        counter_value = result_buffer[flat_idx];

        aql_info("[PMU] READ_SYNC: GPU %u, dimension-specific read: SE=%u SA=%u WGP=%u -> flat_idx=%u, value=%llu",
                 measurement->gpu_id,
                 measurement->target_dims.se,
                 measurement->target_dims.sa,
                 measurement->target_dims.wgp,
                 flat_idx, counter_value);
    } else {
        /* Aggregate across all instances */
        counter_value = aql_aggregate_counter_instances(session, measurement, result_buffer, gpu_idx);
    }

    return counter_value;
}

/**
 * read_compute_delta - Compute delta from start value
 * @measurement: Measurement containing start value
 * @counter_value: Current counter value
 *
 * Returns: Delta value
 */
static uint64_t read_compute_delta(struct aql_measurement *measurement, uint64_t counter_value)
{
    uint64_t delta = 0;

    /* Compute delta from start value.
     * GPU counters are cumulative and don't reset on START, so we track
     * the baseline value and return the delta for this measurement period. */
    if (counter_value >= measurement->start_counter_value) {
        delta = counter_value - measurement->start_counter_value;
    } else {
        /* Counter wrapped around (very unlikely for 64-bit counters) */
        aql_warn("[PMU] READ_SYNC: GPU %u counter wrapped: start=%llu, current=%llu",
                 measurement->gpu_id, measurement->start_counter_value, counter_value);
        delta = counter_value; /* Best effort */
    }

    aql_info("[PMU] READ_SYNC: GPU %u, delta=%llu (current=%llu - start=%llu)",
             measurement->gpu_id, delta, counter_value, measurement->start_counter_value);
    aql_info("[PMU] ========== COMPUTED DELTA: %llu (0x%llx) ==========", delta, delta);

    return delta;
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

    /* DIAGNOSTIC: Log buffer contents before READ */
    read_diagnostic_log_buffer(measurement, "BEFORE READ");

    /* Create and submit READ packet */
    ret = read_submit_packet(session, measurement);
    if (ret) {
        return measurement->last_counter_value;
    }

    /* DIAGNOSTIC: Log buffer contents after READ */
    read_diagnostic_log_buffer(measurement, "AFTER READ");

    /* Get result buffer and extract counter value */
    result_buffer = read_get_result_buffer(measurement);
    counter_value = read_extract_counter_value(session, measurement, result_buffer, gpu_idx);

    aql_info("[PMU] READ_SYNC: GPU %u, absolute counter_value=%llu (dimension_specific=%d)",
             measurement->gpu_id, counter_value, measurement->dimension_specific);
    aql_info("[PMU] ========== RAW COUNTER VALUE: %llu (0x%llx) ==========",
             counter_value, counter_value);

    /* Update cached value with absolute counter value */
    measurement->last_counter_value = counter_value;

    /* Compute and return delta */
    return read_compute_delta(measurement, counter_value);
}

/**
 * aql_perf_measurement_destroy - Destroy measurement and ensure counter release
 * @measurement: Measurement to destroy
 *
 * This releases the creator's reference to the measurement. The actual cleanup
 * and freeing happens in aql_measurement_release() when the last reference is dropped.
 * If there are pending work items, they will keep the measurement alive until they complete.
 */
void aql_perf_measurement_destroy(struct aql_measurement *measurement)
{
    bool is_atomic_ctx;
    unsigned long flags;

    if (!measurement)
        return;

    aql_debug("[PMU] Destroying measurement for GPU %u", measurement->gpu_id);

    /* Check if we're in atomic context (e.g., called from pmu_stub_del) */
    is_atomic_ctx = in_atomic() || irqs_disabled();

    if (is_atomic_ctx) {
        aql_debug("[PMU] Destroy called from atomic context");

        /* Queue async STOP work if measurement is still active */
        if (measurement->state == MEASUREMENT_ACTIVE) {
            aql_debug("[PMU] Queueing async STOP from atomic destroy");
            aql_perf_measurement_stop_atomic(measurement);
        }

        /* Just drop the creator's reference - if there are pending work items,
         * they hold references and will keep the measurement alive until they complete */
        aql_measurement_put(measurement);
        return;
    }

    /* Non-atomic context: safe to do synchronous cleanup before releasing reference */

    /* Ensure measurement is stopped */
    if (measurement->state == MEASUREMENT_ACTIVE) {
        aql_debug("[PMU] Stopping active measurement before destroy");
        aql_perf_measurement_stop(measurement);
    }

    /* Remove from session's active measurements list while we still have the session pointer.
     * The release callback will handle this too, but doing it here ensures it's done
     * before we drop our reference. */
    if (measurement->session && !list_empty(&measurement->list)) {
        spin_lock_irqsave(&measurement->session->measurement_lock, flags);
        if (!list_empty(&measurement->list)) {
            list_del_init(&measurement->list);
            atomic_dec(&measurement->session->active_gpu_count);
            aql_debug("[PMU] Removed measurement from active list");
        }
        spin_unlock_irqrestore(&measurement->session->measurement_lock, flags);
    }

    /* Ensure counter is released if still allocated and owned (use local copy to prevent TOCTOU) */
    if (measurement->owns_counter) {
        counter_reg_info_t *counter = measurement->allocated_counter;
        if (counter) {
            aql_debug("[PMU] Releasing owned counter during destroy");
            aql_counter_release(counter);
            measurement->allocated_counter = NULL;
            measurement->owns_counter = false;
        }
    }

    /* Release shared counter reference if still held */
    if (measurement->shared_ref) {
        aql_debug("[PMU] Releasing shared counter reference during destroy");
        release_shared_counter(measurement->session, measurement->shared_ref);
        measurement->shared_ref = NULL;
    }

    aql_debug("[PMU] Session %llu: Cleanup complete for GPU %u, releasing creator's reference",
              measurement->session ? measurement->session->session_id : 0,
              measurement->gpu_id);

    /* Release the creator's reference - this may trigger aql_measurement_release()
     * if there are no pending work items. If work items exist, they hold references
     * and will trigger the final release when they complete. */
    aql_measurement_put(measurement);
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
            aql_info("[PMU] ========== CACHED VALUE UPDATED: %llu (0x%llx) ==========",
                     counter_value, counter_value);
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

    /* Release the measurement reference we took when creating the work item.
     * This may trigger aql_measurement_release() if it was the last reference. */
    aql_measurement_put(measurement);

    /* Free the work item structure */
    kfree(work_item);
    aql_debug("Work handler cleanup complete: released measurement reference and freed work_item");
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

    if (!measurement) {
        aql_err("Invalid measurement");
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

    /* Take a reference to the measurement to prevent use-after-free */
    aql_measurement_get(measurement);
    aql_debug("Work item created for GPU %u, op_type=%d, took measurement reference",
              measurement->gpu_id, op_type);

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
    struct workqueue_struct *wq;

    if (!measurement) {
        return -EINVAL;
    }

    wq = aql_get_global_workqueue();
    if (!wq) {
        aql_err("Global workqueue not available");
        return -EINVAL;
    }

    work_item = aql_create_work_item(measurement, AQL_WORK_START);
    if (IS_ERR(work_item)) {
        return PTR_ERR(work_item);
    }

    /* Schedule work without waiting on global workqueue */
    if (!queue_work(wq, &work_item->work)) {
        /* Work already queued - release our reference and free work_item */
        aql_measurement_put(measurement);
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
    struct workqueue_struct *wq;

    if (!measurement) {
        return -EINVAL;
    }

    wq = aql_get_global_workqueue();
    if (!wq) {
        aql_err("Global workqueue not available");
        return -EINVAL;
    }

    work_item = aql_create_work_item(measurement, AQL_WORK_STOP);
    if (IS_ERR(work_item)) {
        return PTR_ERR(work_item);
    }

    /* Schedule work without waiting on global workqueue */
    if (!queue_work(wq, &work_item->work)) {
        /* Work already queued - release our reference and free work_item */
        aql_measurement_put(measurement);
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
    struct workqueue_struct *wq;
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

    /* Schedule background refresh of cached value on global workqueue */
    wq = aql_get_global_workqueue();
    if (wq) {
        work_item = aql_create_work_item(measurement, AQL_WORK_READ);
        if (!IS_ERR(work_item)) {
            if (!queue_work(wq, &work_item->work)) {
                /* Work already queued - release our reference and free work_item */
                aql_info("[PMU] READ_ATOMIC: GPU %u, READ work already queued",
                         measurement->gpu_id);
                aql_measurement_put(measurement);
                kfree(work_item);
            } else {
                aql_info("[PMU] READ_ATOMIC: GPU %u, scheduled READ work for background refresh",
                         measurement->gpu_id);
            }
        } else {
            aql_warn("[PMU] READ_ATOMIC: GPU %u, failed to create work item: %ld",
                     measurement->gpu_id, PTR_ERR(work_item));
        }
    } else {
        aql_warn("[PMU] READ_ATOMIC: Global workqueue not available");
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
EXPORT_SYMBOL_GPL(aql_measurement_get);
EXPORT_SYMBOL_GPL(aql_measurement_put);
EXPORT_SYMBOL_GPL(aql_perf_measurement_start_atomic);
EXPORT_SYMBOL_GPL(aql_perf_measurement_stop_atomic);
EXPORT_SYMBOL_GPL(aql_perf_measurement_read_atomic);
EXPORT_SYMBOL_GPL(aql_work_handler);
EXPORT_SYMBOL_GPL(aql_create_work_item);