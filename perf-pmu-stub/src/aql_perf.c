/*
 * aql_perf.c - AQL Performance Counter Integration Implementation
 *
 * This module implements AQL packet submission for performance counter
 * integration in the perf-pmu-stub kernel module.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/workqueue.h>
#include <linux/delay.h>
#include <linux/atomic.h>
#include <linux/percpu.h>
#include <uapi/linux/kfd_ioctl.h>

#include "aql_perf.h"
#include "pmu_stub.h"

/* External KFD functions */
extern int kfd_alloc_device(struct file *filep, struct kfd_process *p,
                           uint32_t gpu_id, size_t data_size,
                           struct kfd_data_alloc *data_alloc);
extern int kfd_destroy_device_mem(struct file *filep, struct kfd_process *p,
                                 uint32_t gpu_id,
                                 struct kfd_data_alloc *data_alloc);
extern int kfd_ioctl_submit_ib_packet(struct file *filep, struct kfd_process *p,
                                     uint32_t gpu_id, const uint32_t* packet,
                                     size_t ib_len);
extern int kfd_process_get_all_gpuids(struct kfd_process *p, uint32_t *gpu_ids,
                                     uint32_t *num_gpus);

/* Global session ID counter */
static atomic64_t session_id_counter = ATOMIC64_INIT(0);

/* Per-CPU statistics */
DEFINE_PER_CPU(struct aql_perf_stats, aql_stats);

/* GFX12 Counter Descriptors */
struct counter_descriptor gfx12_counters[] = {
    {
        .counter_id = 0,
        .counter_select = GFX12_PERF_SEL_SQ_WAVES,
        .counter_mode = 0x1,
        .name = "sq_waves",
        .description = "Number of waves in shader queues",
        .supported_gpus = BIT(GFX12_GPU_TYPE),
    },
    {
        .counter_id = 1,
        .counter_select = GFX12_PERF_SEL_SQ_INSTS,
        .counter_mode = 0x1,
        .name = "sq_instructions",
        .description = "Number of shader instructions executed",
        .supported_gpus = BIT(GFX12_GPU_TYPE),
    },
    {
        .counter_id = 2,
        .counter_select = GFX12_PERF_SEL_TA_BUSY,
        .counter_mode = 0x1,
        .name = "ta_busy",
        .description = "Texture unit busy cycles",
        .supported_gpus = BIT(GFX12_GPU_TYPE),
    }
};

size_t gfx12_counters_count = ARRAY_SIZE(gfx12_counters);

/* Helper Functions */

/**
 * aql_perf_inc_stat - Increment per-CPU statistics
 * @type: Statistics type to increment
 */
void aql_perf_inc_stat(enum aql_stat_type type)
{
    struct aql_perf_stats *stats = this_cpu_ptr(&aql_stats);

    switch (type) {
    case AQL_STAT_PACKETS_SUBMITTED:
        atomic64_inc(&stats->packets_submitted);
        break;
    case AQL_STAT_PACKETS_COMPLETED:
        atomic64_inc(&stats->packets_completed);
        break;
    case AQL_STAT_ERRORS_TOTAL:
        atomic64_inc(&stats->errors_total);
        break;
    case AQL_STAT_SESSIONS_CREATED:
        atomic64_inc(&stats->sessions_created);
        break;
    }
}

/**
 * aql_perf_get_stats - Get aggregated statistics
 * @stats: Output buffer for statistics
 */
void aql_perf_get_stats(struct aql_perf_stats *stats)
{
    int cpu;

    memset(stats, 0, sizeof(*stats));

    for_each_possible_cpu(cpu) {
        struct aql_perf_stats *cpu_stats = per_cpu_ptr(&aql_stats, cpu);
        atomic64_add(atomic64_read(&cpu_stats->packets_submitted), &stats->packets_submitted);
        atomic64_add(atomic64_read(&cpu_stats->packets_completed), &stats->packets_completed);
        atomic64_add(atomic64_read(&cpu_stats->errors_total), &stats->errors_total);
        atomic64_add(atomic64_read(&cpu_stats->sessions_created), &stats->sessions_created);
    }
}

/**
 * aql_perf_find_counter_descriptor - Find counter descriptor by ID
 * @counter_id: Counter ID to find
 *
 * Returns: Counter descriptor or NULL if not found
 */
struct counter_descriptor *aql_perf_find_counter_descriptor(uint32_t counter_id)
{
    size_t i;

    for (i = 0; i < gfx12_counters_count; i++) {
        if (gfx12_counters[i].counter_id == counter_id)
            return &gfx12_counters[i];
    }

    return NULL;
}

/* Session Management */

/**
 * aql_perf_session_release - Release function for reference counting
 * @session: Session to release
 */
static void aql_perf_session_release(struct aql_perf_session *session)
{

    aql_debug("Releasing session %llu", session->session_id);

    /* Ensure all measurements are stopped */
    // This will be implemented in measurement management

    /* Free GPU memory allocations */
    aql_perf_free_gpu_memory(session);

    /* Close KFD file handle */
    if (session->kfd_file) {
        filp_close(session->kfd_file, NULL);
        session->kfd_file = NULL;
    }

    /* Cancel recovery work */
    cancel_delayed_work_sync(&session->recovery.recovery_work);

    /* Free dynamic allocations */
    kfree(session->gpu_ids);
    kfree(session->counter_data);
    kfree(session->packet_data);
    kfree(session->counters.descriptors);
    kfree(session->counters.counter_masks);

    aql_debug("Session %llu fully released", session->session_id);
    kfree(session);
}

/**
 * aql_perf_session_create - Create new AQL performance session
 *
 * Returns: New session or ERR_PTR on error
 */
struct aql_perf_session *aql_perf_session_create(void)
{
    struct aql_perf_session *session;

    session = kzalloc(sizeof(*session), GFP_KERNEL);
    if (!session) {
        aql_err("Failed to allocate session structure");
        return ERR_PTR(-ENOMEM);
    }

    /* Generate unique session ID */
    session->session_id = atomic64_inc_return(&session_id_counter);

    /* Initialize synchronization primitives */
    mutex_init(&session->session_mutex);
    spin_lock_init(&session->measurement_lock);
    INIT_LIST_HEAD(&session->active_measurements);
    refcount_set(&session->ref_count, 1);
    atomic_set(&session->active_gpu_count, 0);

    /* Initialize state */
    session->state = SESSION_UNINITIALIZED;
    session->max_gpus = AQL_PERF_MAX_GPUS;

    /* Setup error recovery */
    INIT_DELAYED_WORK(&session->recovery.recovery_work, aql_perf_recovery_work);

    /* Initialize statistics */
    memset(&session->stats, 0, sizeof(session->stats));

    aql_info("Created AQL performance session %llu", session->session_id);
    aql_perf_inc_stat(AQL_STAT_SESSIONS_CREATED);

    return session;
}

/**
 * aql_perf_session_get - Increment session reference count
 * @session: Session to reference
 */
void aql_perf_session_get(struct aql_perf_session *session)
{
    if (session)
        refcount_inc(&session->ref_count);
}

/**
 * aql_perf_session_put - Decrement session reference count
 * @session: Session to dereference
 */
void aql_perf_session_put(struct aql_perf_session *session)
{
    if (session && refcount_dec_and_test(&session->ref_count))
        aql_perf_session_release(session);
}

/**
 * aql_perf_session_initialize - Initialize AQL session
 * @session: Session to initialize
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_session_initialize(struct aql_perf_session *session)
{
    int ret;

    if (!session) {
        aql_err("Invalid session pointer");
        return -EINVAL;
    }

    mutex_lock(&session->session_mutex);

    if (session->state != SESSION_UNINITIALIZED) {
        aql_err("Session %llu already initialized", session->session_id);
        ret = -EINVAL;
        goto unlock;
    }

    session->state = SESSION_INITIALIZING;
    aql_debug("Initializing session %llu", session->session_id);

    /* Open KFD device */
    session->kfd_file = filp_open("/dev/kfd", O_RDWR, 0);
    if (IS_ERR(session->kfd_file)) {
        ret = PTR_ERR(session->kfd_file);
        aql_err("Session %llu: Failed to open /dev/kfd: %d",
                session->session_id, ret);
        session->kfd_file = NULL;
        goto error;
    }

    /* Extract kfd_process from private_data */
    session->process = session->kfd_file->private_data;
    if (!session->process) {
        aql_err("Session %llu: No kfd_process in file private_data",
                session->session_id);
        ret = -EINVAL;
        goto error;
    }

    aql_debug("Session %llu: KFD device opened successfully", session->session_id);

    /* Discover available GPUs */
    ret = aql_perf_discover_gpus(session);
    if (ret) {
        aql_err("Session %llu: GPU discovery failed: %d",
                session->session_id, ret);
        goto error;
    }

    /* Allocate GPU memory */
    ret = aql_perf_allocate_gpu_memory(session);
    if (ret) {
        aql_err("Session %llu: GPU memory allocation failed: %d",
                session->session_id, ret);
        goto error;
    }

    /* Initialize counter configuration with default GFX12 counter */
    session->counters.num_counters = 1;
    session->counters.max_counters = 4;

    session->counters.descriptors = kzalloc(session->counters.max_counters *
                                           sizeof(struct gfx12_counter_desc),
                                           GFP_KERNEL);
    if (!session->counters.descriptors) {
        ret = -ENOMEM;
        goto error;
    }

    /* Configure default counter (SQ_WAVES) */
    session->counters.descriptors[0].counter_select = GFX12_PERF_SEL_SQ_WAVES;
    session->counters.descriptors[0].counter_mode = 0x1;
    session->counters.descriptors[0].result_size = sizeof(uint64_t);

    session->state = SESSION_ACTIVE;
    aql_info("Session %llu initialized successfully with %u GPUs",
             session->session_id, session->num_gpus);

    mutex_unlock(&session->session_mutex);
    return 0;

error:
    session->state = SESSION_ERROR;
    if (session->kfd_file) {
        filp_close(session->kfd_file, NULL);
        session->kfd_file = NULL;
    }
unlock:
    mutex_unlock(&session->session_mutex);
    return ret;
}

/**
 * aql_perf_session_destroy - Destroy AQL session
 * @session: Session to destroy
 */
void aql_perf_session_destroy(struct aql_perf_session *session)
{
    if (!session)
        return;

    aql_debug("Destroying session %llu", session->session_id);

    mutex_lock(&session->session_mutex);
    session->state = SESSION_DESTROYING;
    mutex_unlock(&session->session_mutex);

    /* This will trigger cleanup through reference counting */
    aql_perf_session_put(session);
}

/* GPU Discovery and Memory Management */

/**
 * aql_perf_discover_gpus - Discover available GPUs
 * @session: AQL performance session
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_discover_gpus(struct aql_perf_session *session)
{
    uint32_t max_gpus = AQL_PERF_MAX_GPUS;
    int ret;

    session->gpu_ids = kzalloc(max_gpus * sizeof(uint32_t), GFP_KERNEL);
    if (!session->gpu_ids) {
        aql_err("Session %llu: Failed to allocate GPU ID array",
                session->session_id);
        return -ENOMEM;
    }

    ret = kfd_process_get_all_gpuids(session->process, session->gpu_ids, &max_gpus);
    if (ret) {
        aql_err("Session %llu: Failed to get GPU IDs: %d",
                session->session_id, ret);
        kfree(session->gpu_ids);
        session->gpu_ids = NULL;
        return ret;
    }

    session->num_gpus = max_gpus;
    aql_debug("Session %llu: Discovered %u GPUs", session->session_id, session->num_gpus);

    /* Log discovered GPU IDs */
    for (uint32_t i = 0; i < session->num_gpus; i++) {
        aql_debug("Session %llu: GPU[%u] ID = %u",
                  session->session_id, i, session->gpu_ids[i]);
    }

    return 0;
}

/**
 * aql_perf_allocate_gpu_memory - Allocate memory for all GPUs
 * @session: AQL performance session
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_perf_allocate_gpu_memory(struct aql_perf_session *session)
{
    uint32_t i;
    int ret;

    if (!session->gpu_ids || session->num_gpus == 0) {
        aql_err("Session %llu: No GPUs discovered", session->session_id);
        return -ENODEV;
    }

    /* Allocate memory allocation arrays */
    session->counter_data = kzalloc(session->num_gpus * sizeof(struct kfd_data_alloc),
                                   GFP_KERNEL);
    session->packet_data = kzalloc(session->num_gpus * sizeof(struct kfd_data_alloc),
                                  GFP_KERNEL);

    if (!session->counter_data || !session->packet_data) {
        aql_err("Session %llu: Failed to allocate data structures",
                session->session_id);
        ret = -ENOMEM;
        goto cleanup;
    }

    /* Allocate memory for each GPU */
    for (i = 0; i < session->num_gpus; i++) {
        /* Allocate counter result buffer */
        ret = kfd_alloc_device(session->kfd_file, session->process,
                              session->gpu_ids[i], AQL_PERF_RESULT_BUFFER_SIZE,
                              &session->counter_data[i]);
        if (ret) {
            aql_err("Session %llu: Failed to allocate counter buffer for GPU %u: %d",
                    session->session_id, session->gpu_ids[i], ret);
            goto cleanup_partial;
        }

        /* Allocate packet buffer */
        ret = kfd_alloc_device(session->kfd_file, session->process,
                              session->gpu_ids[i], AQL_PERF_PACKET_BUFFER_SIZE,
                              &session->packet_data[i]);
        if (ret) {
            aql_err("Session %llu: Failed to allocate packet buffer for GPU %u: %d",
                    session->session_id, session->gpu_ids[i], ret);
            /* Clean up counter buffer for this GPU */
            kfd_destroy_device_mem(session->kfd_file, session->process,
                                  session->gpu_ids[i], &session->counter_data[i]);
            goto cleanup_partial;
        }

        aql_debug("Session %llu: Allocated memory for GPU %u (counter=%p, packet=%p)",
                  session->session_id, session->gpu_ids[i],
                  session->counter_data[i].cpu_addr,
                  session->packet_data[i].cpu_addr);
    }

    aql_info("Session %llu: Successfully allocated memory for %u GPUs",
             session->session_id, session->num_gpus);
    return 0;

cleanup_partial:
    /* Clean up partially allocated memory */
    for (uint32_t j = 0; j < i; j++) {
        kfd_destroy_device_mem(session->kfd_file, session->process,
                              session->gpu_ids[j], &session->counter_data[j]);
        kfd_destroy_device_mem(session->kfd_file, session->process,
                              session->gpu_ids[j], &session->packet_data[j]);
    }

cleanup:
    kfree(session->counter_data);
    kfree(session->packet_data);
    session->counter_data = NULL;
    session->packet_data = NULL;
    return ret;
}

/**
 * aql_perf_free_gpu_memory - Free all GPU memory allocations
 * @session: AQL performance session
 */
void aql_perf_free_gpu_memory(struct aql_perf_session *session)
{
    uint32_t i;

    if (!session->counter_data || !session->packet_data)
        return;

    aql_debug("Session %llu: Freeing GPU memory for %u GPUs",
              session->session_id, session->num_gpus);

    for (i = 0; i < session->num_gpus; i++) {
        if (session->counter_data[i].cpu_addr) {
            kfd_destroy_device_mem(session->kfd_file, session->process,
                                  session->gpu_ids[i], &session->counter_data[i]);
            aql_debug("Session %llu: Freed counter memory for GPU %u",
                      session->session_id, session->gpu_ids[i]);
        }

        if (session->packet_data[i].cpu_addr) {
            kfd_destroy_device_mem(session->kfd_file, session->process,
                                  session->gpu_ids[i], &session->packet_data[i]);
            aql_debug("Session %llu: Freed packet memory for GPU %u",
                      session->session_id, session->gpu_ids[i]);
        }
    }

    kfree(session->counter_data);
    kfree(session->packet_data);
    session->counter_data = NULL;
    session->packet_data = NULL;
}

EXPORT_SYMBOL_GPL(aql_perf_session_create);
EXPORT_SYMBOL_GPL(aql_perf_session_initialize);
EXPORT_SYMBOL_GPL(aql_perf_session_destroy);
EXPORT_SYMBOL_GPL(aql_perf_session_get);
EXPORT_SYMBOL_GPL(aql_perf_session_put);