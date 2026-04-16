#define _GNU_SOURCE

#include <inttypes.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "shim_sdk_compat.h"

#include <rocprofiler-sdk/rocprofiler.h>

#include "shim_ipc.h"
#include "shim_protocol.h"

static volatile int g_stop = 0;

static void
on_sigint(int sig)
{
    (void) sig;
    g_stop = 1;
}

typedef struct
{
    char   buffer[512];
    size_t length;
} shim_arg_string_t;

static void
shim_appendf(shim_arg_string_t* out, const char* fmt, ...)
{
    va_list args;
    int     written = 0;

    if(out == NULL || out->length >= sizeof(out->buffer)) return;

    va_start(args, fmt);
    written = vsnprintf(out->buffer + out->length, sizeof(out->buffer) - out->length, fmt, args);
    va_end(args);

    if(written <= 0) return;

    if((size_t) written >= sizeof(out->buffer) - out->length)
        out->length = sizeof(out->buffer) - 1;
    else
        out->length += (size_t) written;
}

static int
shim_record_args_cb(rocprofiler_buffer_tracing_kind_t kind,
                    rocprofiler_tracing_operation_t   operation,
                    uint32_t                          arg_number,
                    const void* const                 arg_value_addr,
                    int32_t                           arg_indirection_count,
                    const char*                       arg_type,
                    const char*                       arg_name,
                    const char*                       arg_value_str,
                    void*                             data)
{
    shim_arg_string_t* out = (shim_arg_string_t*) data;

    (void) kind;
    (void) operation;
    (void) arg_value_addr;
    (void) arg_indirection_count;
    (void) arg_type;

    if(out == NULL) return 0;
    if(arg_number == 0) shim_appendf(out, " args=");
    else shim_appendf(out, ", ");

    shim_appendf(out, "%s=%s", (arg_name != NULL) ? arg_name : "arg",
                 (arg_value_str != NULL) ? arg_value_str : "?");
    return 0;
}

static const char*
shim_kind_name(uint32_t category, uint32_t kind)
{
    const char* name = NULL;

    if(category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
       rocprofiler_query_buffer_tracing_kind_name((rocprofiler_buffer_tracing_kind_t) kind, &name, NULL) ==
           ROCPROFILER_STATUS_SUCCESS &&
       name != NULL)
        return name;

    return "unknown-kind";
}

static const char*
shim_operation_name(rocprofiler_buffer_tracing_kind_t kind, rocprofiler_tracing_operation_t operation)
{
    const char* name = NULL;

    if(rocprofiler_query_buffer_tracing_kind_operation_name(kind, operation, &name, NULL) ==
           ROCPROFILER_STATUS_SUCCESS &&
       name != NULL)
        return name;

    return "unknown-op";
}

static void
shim_print_runtime_registrations(const shim_ctrl_t* ctrl)
{
    if(ctrl == NULL) return;

    printf("=== Runtime registrations: %u ===\n", ctrl->n_runtime_registrations);
    for(uint32_t i = 0; i < ctrl->n_runtime_registrations; ++i)
    {
        const shim_runtime_registration_t* reg = &ctrl->runtime_registrations[i];
        printf("  [%u] %s v%u.%u.%u instance=%" PRIu64 "\n",
               i,
               reg->common_name,
               reg->major_version,
               reg->minor_version,
               reg->patch_version,
               reg->instance);
    }
}

static void
shim_print_trace_record(const shim_record_t* rec)
{
    rocprofiler_record_header_t hdr = {
        .category = rec->category,
        .kind     = rec->kind,
        .payload  = (void*) rec->payload.bytes,
    };
    shim_arg_string_t args = {{0}, 0};

    if(rec->category == ROCPROFILER_BUFFER_CATEGORY_TRACING)
    {
        switch((rocprofiler_buffer_tracing_kind_t) rec->kind)
        {
            case ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION:
            {
                const rocprofiler_buffer_tracing_runtime_initialization_record_t* record =
                    (const rocprofiler_buffer_tracing_runtime_initialization_record_t*) rec->payload.bytes;
                printf("[#%" PRIu64 "] %s :: %s tid=%" PRIu64
                       " ts=%" PRIu64 " version=%" PRIu64 " instance=%" PRIu64 "\n",
                       rec->sequence,
                       shim_kind_name(rec->category, rec->kind),
                       shim_operation_name((rocprofiler_buffer_tracing_kind_t) rec->kind, record->operation),
                       record->thread_id,
                       record->timestamp,
                       record->version,
                       record->instance);
                return;
            }
            case ROCPROFILER_BUFFER_TRACING_HSA_CORE_API:
            {
                const rocprofiler_buffer_tracing_hsa_api_record_t* record =
                    (const rocprofiler_buffer_tracing_hsa_api_record_t*) rec->payload.bytes;
                (void) rocprofiler_iterate_buffer_tracing_record_args(hdr, &shim_record_args_cb, &args);
                printf("[#%" PRIu64 "] %s :: %s tid=%" PRIu64
                       " cid={i=%" PRIu64 " e=%" PRIu64 " a=%" PRIu64 "}"
                       " start=%" PRIu64 " end=%" PRIu64 "%s\n",
                       rec->sequence,
                       shim_kind_name(rec->category, rec->kind),
                       shim_operation_name((rocprofiler_buffer_tracing_kind_t) rec->kind, record->operation),
                       record->thread_id,
                       record->correlation_id.internal,
                       record->correlation_id.external.value,
                       record->correlation_id.ancestor,
                       record->start_timestamp,
                       record->end_timestamp,
                       args.buffer);
                return;
            }
            case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT:
            {
                const rocprofiler_buffer_tracing_hip_api_ext_record_t* record =
                    (const rocprofiler_buffer_tracing_hip_api_ext_record_t*) rec->payload.bytes;
                (void) rocprofiler_iterate_buffer_tracing_record_args(hdr, &shim_record_args_cb, &args);
                printf("[#%" PRIu64 "] %s :: %s tid=%" PRIu64
                       " cid={i=%" PRIu64 " e=%" PRIu64 " a=%" PRIu64 "}"
                       " start=%" PRIu64 " end=%" PRIu64 "%s\n",
                       rec->sequence,
                       shim_kind_name(rec->category, rec->kind),
                       shim_operation_name((rocprofiler_buffer_tracing_kind_t) rec->kind, record->operation),
                       record->thread_id,
                       record->correlation_id.internal,
                       record->correlation_id.external.value,
                       record->correlation_id.ancestor,
                       record->start_timestamp,
                       record->end_timestamp,
                       args.buffer);
                return;
            }
            default: break;
        }
    }

    printf("[#%" PRIu64 "] category=%u kind=%u payload=%u truncated=%u\n",
           rec->sequence,
           rec->category,
           rec->kind,
           rec->payload_bytes,
           rec->payload_truncated);
}

static void
on_record(const shim_record_t* rec, void* user_data)
{
    uint64_t* count = (uint64_t*) user_data;

    if(rec->sequence == 0 || rec->category == 0) return;

    (*count)++;
    if(*count > 40 && (*count % 5000) != 0) return;

    shim_print_trace_record(rec);
}

int
main(int argc, char** argv)
{
    shim_ipc_consumer_t con;
    uint64_t            total_records = 0;
    uint64_t            traced        = 0;
    uint64_t            dropped       = 0;
    uint64_t            sdk_dropped   = 0;
    pid_t               target        = 0;
    int                 duration      = 5;

    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <pid> [duration_sec]\n", argv[0]);
        return 2;
    }

    target = (pid_t) atoi(argv[1]);
    if(argc > 2) duration = atoi(argv[2]);

    signal(SIGINT, on_sigint);

    if(shim_consumer_attach(target, &con) != 0)
    {
        fprintf(stderr, "attach(%d) failed\n", (int) target);
        return 1;
    }

    printf("=== Attached to pid=%u, services=%u ===\n", con.ctrl->pid, con.ctrl->active_services);
    shim_print_runtime_registrations(con.ctrl);

    atomic_store_explicit(&con.ctrl->capture_enabled, 1, memory_order_release);
    printf("=== Transport capture enabled ===\n");

    time_t start = time(NULL);
    while(!g_stop && (time(NULL) - start) < duration)
    {
        (void) shim_consumer_poll(&con, &on_record, &total_records, 500);
    }

    traced      = atomic_load_explicit(&con.ctrl->events_traced, memory_order_relaxed);
    dropped     = atomic_load_explicit(&con.ctrl->events_dropped, memory_order_relaxed);
    sdk_dropped = atomic_load_explicit(&con.ctrl->sdk_drop_count, memory_order_relaxed);

    shim_print_runtime_registrations(con.ctrl);
    printf("=== Stats: traced=%" PRIu64 " transport_dropped=%" PRIu64
           " sdk_drop_count=%" PRIu64 " consumer_read=%" PRIu64 " ===\n",
           traced,
           dropped,
           sdk_dropped,
           total_records);

    shim_consumer_detach(&con);
    printf("=== Detached ===\n");
    return 0;
}
