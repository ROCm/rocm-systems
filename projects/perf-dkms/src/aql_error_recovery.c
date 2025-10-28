/*
 * aql_error_recovery.c - AQL Error Handling and Recovery Implementation
 *
 * This module implements comprehensive error handling and recovery mechanisms
 * for the AQL packet submission system.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/workqueue.h>
#include <linux/delay.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/list.h>

#include "aql_perf.h"
#include "amdgpu_pmu.h"

/**
 * aql_perf_disable_gpu - Disable a faulty GPU
 * @session: AQL performance session
 * @gpu_id: GPU ID to disable
 */
static void aql_perf_disable_gpu(struct aql_perf_session *session, uint32_t gpu_id)
{
    struct aql_measurement *measurement, *tmp;
    unsigned long flags;

    aql_info("Session %llu: Disabling faulty GPU %u", session->session_id, gpu_id);

    /* Stop all measurements on this GPU */
    spin_lock_irqsave(&session->measurement_lock, flags);

    list_for_each_entry_safe(measurement, tmp, &session->active_measurements, list) {
        if (measurement->gpu_id == gpu_id) {
            measurement->state = MEASUREMENT_ERROR;
            list_del(&measurement->list);
            atomic_dec(&session->active_gpu_count);

            aql_debug("Session %llu: Stopped measurement on disabled GPU %u",
                      session->session_id, gpu_id);
        }
    }

    spin_unlock_irqrestore(&session->measurement_lock, flags);

    /* Mark GPU as disabled in error mask */
    set_bit(gpu_id, &session->recovery.error_mask);
}

/**
 * aql_perf_stop_all_measurements - Stop all active measurements
 * @session: AQL performance session
 */
static void aql_perf_stop_all_measurements(struct aql_perf_session *session)
{
    struct aql_measurement *measurement, *tmp;
    unsigned long flags;

    aql_debug("Session %llu: Stopping all active measurements", session->session_id);

    spin_lock_irqsave(&session->measurement_lock, flags);

    list_for_each_entry_safe(measurement, tmp, &session->active_measurements, list) {
        measurement->state = MEASUREMENT_ERROR;
        list_del(&measurement->list);
        atomic_dec(&session->active_gpu_count);
    }

    spin_unlock_irqrestore(&session->measurement_lock, flags);

    aql_debug("Session %llu: All measurements stopped", session->session_id);
}

/**
 * aql_perf_reinitialize_session - Attempt to reinitialize a session
 * @session: AQL performance session to reinitialize
 *
 * Returns: 0 on success, negative error code on failure
 */
static int aql_perf_reinitialize_session(struct aql_perf_session *session)
{
    int ret;

    aql_info("Session %llu: Attempting session reinitialization", session->session_id);

    /* Stop all active measurements first */
    aql_perf_stop_all_measurements(session);

    /* Free existing counter buffers for all GPUs */
    if (session->archs && session->num_gpus > 0) {
        for (uint32_t i = 0; i < session->num_gpus; i++) {
            if (session->archs[i]) {
                aql_perf_free_counter_buffers(session->archs[i], session->kfd_file,
                                              session->process, session->gpu_ids[i]);
            }
        }
    }

    /* Close existing KFD file */
    if (session->kfd_file) {
        filp_close(session->kfd_file, NULL);
        session->kfd_file = NULL;
        session->process = NULL;
    }

    /* Clear error mask */
    session->recovery.error_mask = 0;

    /* Re-open KFD device */
    session->kfd_file = filp_open("/dev/kfd", O_RDWR, 0);
    if (IS_ERR(session->kfd_file)) {
        ret = PTR_ERR(session->kfd_file);
        aql_err("Session %llu: Failed to reopen /dev/kfd during recovery: %d",
                session->session_id, ret);
        session->kfd_file = NULL;
        return ret;
    }

    /* Extract kfd_process */
    session->process = session->kfd_file->private_data;
    if (!session->process) {
        aql_err("Session %llu: No kfd_process found during recovery",
                session->session_id);
        filp_close(session->kfd_file, NULL);
        session->kfd_file = NULL;
        return -EINVAL;
    }

    /* Rediscover GPUs */
    kfree(session->gpu_ids);
    session->gpu_ids = NULL;
    session->num_gpus = 0;

    ret = aql_perf_discover_gpus(session);
    if (ret) {
        aql_err("Session %llu: GPU rediscovery failed during recovery: %d",
                session->session_id, ret);
        return ret;
    }

    /* Reallocate counter buffers for all GPU architectures */
    if (session->archs && session->num_gpus > 0) {
        for (uint32_t i = 0; i < session->num_gpus; i++) {
            if (session->archs[i]) {
                ret = aql_perf_allocate_counter_buffers(session->archs[i], session->kfd_file,
                                                        session->process, session->gpu_ids[i]);
                if (ret) {
                    aql_err("Session %llu: Counter buffer reallocation failed for GPU %u during recovery: %d",
                            session->session_id, session->gpu_ids[i], ret);
                    return ret;
                }
            }
        }
    }

    aql_info("Session %llu: Session reinitialization successful", session->session_id);
    return 0;
}

/**
 * aql_perf_handle_error - Handle AQL performance errors
 * @session: AQL performance session
 * @error: Error context
 */
void aql_perf_handle_error(struct aql_perf_session *session,
                          struct aql_error_context *error)
{
    if (!session || !error) {
        aql_err("Invalid parameters for error handling");
        return;
    }

    aql_err("Session %llu: Handling error - severity=%d, gpu_id=%u, code=%d, msg='%s'",
            session->session_id, error->severity, error->gpu_id,
            error->error_code, error->error_msg);

    /* Update error statistics */
    atomic64_inc(&session->stats.errors_total);

    switch (error->severity) {
    case AQL_ERROR_RECOVERABLE:
        /* Simple retry with exponential backoff */
        if (session->recovery.recovery_attempts < AQL_PERF_MAX_RECOVERY_ATTEMPTS) {
            unsigned long delay = msecs_to_jiffies(100 << session->recovery.recovery_attempts);

            aql_info("Session %llu: Scheduling recoverable error retry (attempt %d)",
                     session->session_id, session->recovery.recovery_attempts + 1);

            schedule_delayed_work(&session->recovery.recovery_work, delay);
            session->recovery.recovery_attempts++;
        } else {
            aql_err("Session %llu: Max recovery attempts exceeded for recoverable error",
                    session->session_id);

            /* Escalate to system fault */
            error->severity = AQL_ERROR_SYSTEM_FAULT;
            aql_perf_handle_error(session, error);
        }
        break;

    case AQL_ERROR_GPU_FAULT:
        /* Isolate the faulty GPU */
        aql_perf_disable_gpu(session, error->gpu_id);
        aql_info("Session %llu: GPU %u disabled due to fault: %s",
                 session->session_id, error->gpu_id, error->error_msg);

        /* If all GPUs are disabled, escalate to system fault */
        if (hweight_long(session->recovery.error_mask) >= session->num_gpus) {
            aql_err("Session %llu: All GPUs disabled, escalating to system fault",
                    session->session_id);
            error->severity = AQL_ERROR_SYSTEM_FAULT;
            aql_perf_handle_error(session, error);
        }
        break;

    case AQL_ERROR_SYSTEM_FAULT:
        /* Reset the entire session */
        mutex_lock(&session->session_mutex);

        if (session->state != SESSION_ERROR) {
            aql_info("Session %llu: System fault detected, stopping all measurements",
                     session->session_id);
            session->state = SESSION_ERROR;
            aql_perf_stop_all_measurements(session);
        }

        /* Schedule recovery work */
        if (session->recovery.recovery_attempts < AQL_PERF_MAX_RECOVERY_ATTEMPTS) {
            schedule_delayed_work(&session->recovery.recovery_work,
                                 msecs_to_jiffies(1000));
            aql_info("Session %llu: Scheduled system recovery (attempt %d)",
                     session->session_id, session->recovery.recovery_attempts + 1);
        } else {
            aql_err("Session %llu: Max system recovery attempts exceeded",
                    session->session_id);
            error->severity = AQL_ERROR_PERMANENT;
            aql_perf_handle_error(session, error);
        }

        mutex_unlock(&session->session_mutex);
        break;

    case AQL_ERROR_PERMANENT:
        /* Disable AQL feature entirely */
        mutex_lock(&session->session_mutex);
        session->state = SESSION_ERROR;
        aql_perf_stop_all_measurements(session);
        mutex_unlock(&session->session_mutex);

        aql_err("Session %llu: Permanent error detected, AQL performance monitoring disabled",
                session->session_id);
        break;

    default:
        aql_err("Session %llu: Unknown error severity %d",
                session->session_id, error->severity);
        break;
    }
}

/**
 * aql_perf_recovery_work - Recovery work function
 * @work: Work structure
 */
void aql_perf_recovery_work(struct work_struct *work)
{
    struct aql_perf_session *session = container_of(work, struct aql_perf_session,
                                                    recovery.recovery_work.work);
    int ret;

    aql_info("Session %llu: Starting recovery work (attempt %d)",
             session->session_id, session->recovery.recovery_attempts);

    mutex_lock(&session->session_mutex);

    if (session->state == SESSION_ERROR) {
        /* Attempt to reinitialize the session */
        ret = aql_perf_reinitialize_session(session);
        if (ret == 0) {
            session->state = SESSION_ACTIVE;
            session->recovery.recovery_attempts = 0;
            aql_info("Session %llu: Recovery successful", session->session_id);
        } else {
            session->recovery.recovery_attempts++;

            if (session->recovery.recovery_attempts < AQL_PERF_MAX_RECOVERY_ATTEMPTS) {
                /* Retry recovery with longer delay */
                unsigned long delay = msecs_to_jiffies(5000 * session->recovery.recovery_attempts);
                schedule_delayed_work(&session->recovery.recovery_work, delay);

                aql_info("Session %llu: Recovery failed (attempt %d), retrying in %ums",
                         session->session_id, session->recovery.recovery_attempts,
                         jiffies_to_msecs(delay));
            } else {
                /* Give up after max attempts */
                struct aql_error_context error = {
                    .severity = AQL_ERROR_PERMANENT,
                    .gpu_id = 0,
                    .error_code = ret,
                    .timestamp = ktime_get()
                };
                snprintf(error.error_msg, sizeof(error.error_msg),
                        "Recovery failed after %d attempts", AQL_PERF_MAX_RECOVERY_ATTEMPTS);

                aql_err("Session %llu: Recovery failed after %d attempts",
                        session->session_id, AQL_PERF_MAX_RECOVERY_ATTEMPTS);

                /* Handle as permanent error */
                mutex_unlock(&session->session_mutex);
                aql_perf_handle_error(session, &error);
                return;
            }
        }
    } else {
        aql_debug("Session %llu: Recovery work called but session state is %d",
                  session->session_id, session->state);
    }

    mutex_unlock(&session->session_mutex);
}

/**
 * aql_perf_create_error_context - Create error context structure
 * @severity: Error severity level
 * @gpu_id: GPU ID associated with error
 * @error_code: Error code
 * @error_msg: Error message format string
 * @...: Format string arguments
 *
 * Returns: Allocated error context or NULL on failure
 */
struct aql_error_context *aql_perf_create_error_context(enum aql_error_severity severity,
                                                        uint32_t gpu_id,
                                                        int error_code,
                                                        const char *error_msg, ...)
{
    struct aql_error_context *error;
    va_list args;

    error = kzalloc(sizeof(*error), GFP_KERNEL);
    if (!error)
        return NULL;

    error->severity = severity;
    error->gpu_id = gpu_id;
    error->error_code = error_code;
    error->timestamp = ktime_get();

    if (error_msg) {
        va_start(args, error_msg);
        vsnprintf(error->error_msg, sizeof(error->error_msg), error_msg, args);
        va_end(args);
    }

    return error;
}

/**
 * aql_perf_destroy_error_context - Destroy error context
 * @error: Error context to destroy
 */
void aql_perf_destroy_error_context(struct aql_error_context *error)
{
    kfree(error);
}

/**
 * aql_perf_is_gpu_disabled - Check if GPU is disabled
 * @session: AQL performance session
 * @gpu_id: GPU ID to check
 *
 * Returns: true if GPU is disabled, false otherwise
 */
bool aql_perf_is_gpu_disabled(struct aql_perf_session *session, uint32_t gpu_id)
{
    uint32_t i;

    if (!session)
        return true;

    /* Find GPU index from GPU ID */
    for (i = 0; i < session->num_gpus; i++) {
        if (session->gpu_ids[i] == gpu_id) {
            /* Use index (0, 1, 2...) not hardware ID (7410, 7411...) */
            return test_bit(i, &session->recovery.error_mask);
        }
    }

    /* GPU ID not found in session - treat as disabled */
    return true;
}

/**
 * aql_perf_get_healthy_gpu_count - Get count of healthy (non-disabled) GPUs
 * @session: AQL performance session
 *
 * Returns: Number of healthy GPUs
 */
uint32_t aql_perf_get_healthy_gpu_count(struct aql_perf_session *session)
{
    uint32_t healthy_count = 0;
    uint32_t i;

    if (!session)
        return 0;

    for (i = 0; i < session->num_gpus; i++) {
        if (!aql_perf_is_gpu_disabled(session, session->gpu_ids[i])) {
            healthy_count++;
        }
    }

    return healthy_count;
}

EXPORT_SYMBOL_GPL(aql_perf_handle_error);
EXPORT_SYMBOL_GPL(aql_perf_recovery_work);
EXPORT_SYMBOL_GPL(aql_perf_create_error_context);
EXPORT_SYMBOL_GPL(aql_perf_destroy_error_context);
EXPORT_SYMBOL_GPL(aql_perf_is_gpu_disabled);
EXPORT_SYMBOL_GPL(aql_perf_get_healthy_gpu_count);