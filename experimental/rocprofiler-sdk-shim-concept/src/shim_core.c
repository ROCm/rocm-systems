#define _GNU_SOURCE

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "shim_sdk_compat.h"

#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>

#include "shim_ipc.h"
#include "shim_protocol.h"

#define SHIM_BUFFER_SIZE_BYTES      (1U << 20)
#define SHIM_BUFFER_WATERMARK_BYTES (SHIM_BUFFER_SIZE_BYTES / 2U)

typedef struct
{
    rocprofiler_client_finalize_t finalize_func;
    rocprofiler_context_id_t      context;
    rocprofiler_buffer_id_t       buffer;
    int                           context_started;
} shim_tool_data_t;

static shim_ipc_target_t g_ipc;
static int               g_ipc_ok = 0;
static shim_tool_data_t  g_tool_data;
static pthread_mutex_t   g_runtime_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t   g_ipc_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t  g_sequence_counter = 0;

static int
shim_is_enabled(void)
{
    const char* env = getenv("ROCP_SHIM_CONCEPT_ENABLE");
    return (env != NULL && strcmp(env, "1") == 0);
}

static int
shim_ensure_ipc_ready(void)
{
    int rc = 0;

    pthread_mutex_lock(&g_ipc_lock);
    if(!g_ipc_ok)
    {
        rc = shim_ipc_init(&g_ipc);
        if(rc == 0)
            g_ipc_ok = 1;
        else
            fprintf(stderr, "[shim] failed to initialize IPC transport\n");
    }
    pthread_mutex_unlock(&g_ipc_lock);
    return rc;
}

static const char*
shim_status_name(rocprofiler_status_t status)
{
    const char* name = rocprofiler_get_status_name(status);
    return (name != NULL) ? name : "ROCPROFILER_STATUS_UNKNOWN";
}

static int
shim_expect_status(rocprofiler_status_t status, const char* message)
{
    if(status == ROCPROFILER_STATUS_SUCCESS) return 0;

    fprintf(stderr, "[shim] %s failed: %s\n", message, shim_status_name(status));
    return -1;
}

static void
shim_copy_payload(shim_record_t* out, const rocprofiler_record_header_t* header)
{
    size_t payload_bytes = 0;

    if(header->payload != NULL && header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING)
    {
        payload_bytes = (size_t) (*((const uint64_t*) header->payload));
    }

    if(payload_bytes > sizeof(out->payload.bytes))
    {
        out->payload_truncated = 1U;
        payload_bytes          = sizeof(out->payload.bytes);
    }

    out->payload_bytes = (uint32_t) payload_bytes;

    if(payload_bytes > 0 && header->payload != NULL)
    {
        memcpy(out->payload.bytes, header->payload, payload_bytes);
    }
}

static const char*
shim_runtime_name(rocprofiler_runtime_initialization_operation_t op)
{
    switch(op)
    {
        case ROCPROFILER_RUNTIME_INITIALIZATION_HSA: return "hsa";
        case ROCPROFILER_RUNTIME_INITIALIZATION_HIP: return "hip";
        case ROCPROFILER_RUNTIME_INITIALIZATION_MARKER: return "marker";
        case ROCPROFILER_RUNTIME_INITIALIZATION_RCCL: return "rccl";
        case ROCPROFILER_RUNTIME_INITIALIZATION_ROCDECODE: return "rocdecode";
        case ROCPROFILER_RUNTIME_INITIALIZATION_ROCJPEG: return "rocjpeg";
        default: return "unknown";
    }
}

static void
shim_publish_runtime_init(
    const rocprofiler_buffer_tracing_runtime_initialization_record_t* record)
{
    if(record == NULL || !g_ipc_ok || g_ipc.ctrl == NULL) return;

    pthread_mutex_lock(&g_runtime_lock);

    for(uint32_t i = 0; i < g_ipc.ctrl->n_runtime_registrations; ++i)
    {
        shim_runtime_registration_t* reg = &g_ipc.ctrl->runtime_registrations[i];
        if(strcmp(reg->common_name, shim_runtime_name(record->operation)) == 0 &&
           reg->instance == record->instance)
        {
            pthread_mutex_unlock(&g_runtime_lock);
            return;
        }
    }

    if(g_ipc.ctrl->n_runtime_registrations < SHIM_MAX_RUNTIME_REGISTRATIONS)
    {
        shim_runtime_registration_t* reg =
            &g_ipc.ctrl->runtime_registrations[g_ipc.ctrl->n_runtime_registrations++];
        memset(reg, 0, sizeof(*reg));
        snprintf(reg->common_name, sizeof(reg->common_name), "%s",
                 shim_runtime_name(record->operation));
        reg->major_version = (uint32_t) (record->version / 10000);
        reg->minor_version = (uint32_t) ((record->version % 10000) / 100);
        reg->patch_version = (uint32_t) (record->version % 100);
        reg->instance      = record->instance;
    }

    pthread_mutex_unlock(&g_runtime_lock);
}

static void
shim_sdk_buffer_callback(rocprofiler_context_id_t      context,
                         rocprofiler_buffer_id_t       buffer_id,
                         rocprofiler_record_header_t** headers,
                         size_t                        num_headers,
                         void*                         user_data,
                         uint64_t                      drop_count)
{
    (void) context;
    (void) buffer_id;
    (void) user_data;

    if(!g_ipc_ok || g_ipc.ctrl == NULL) return;

    if(drop_count > 0)
    {
        atomic_fetch_add_explicit(&g_ipc.ctrl->sdk_drop_count, drop_count, memory_order_relaxed);
    }

    if(headers == NULL || num_headers == 0) return;

    for(size_t i = 0; i < num_headers; ++i)
    {
        const rocprofiler_record_header_t* header = headers[i];
        shim_record_t                      rec;

        if(header == NULL) continue;

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
           header->kind == ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION &&
           header->payload != NULL)
        {
            shim_publish_runtime_init(
                (const rocprofiler_buffer_tracing_runtime_initialization_record_t*) header->payload);
        }

        if(atomic_load_explicit(&g_ipc.ctrl->capture_enabled, memory_order_acquire) == 0) continue;

        memset(&rec, 0, sizeof(rec));
        rec.sequence = atomic_fetch_add_explicit(&g_sequence_counter, 1, memory_order_relaxed) + 1;
        rec.category = header->category;
        rec.kind     = header->kind;
        shim_copy_payload(&rec, header);
        (void) shim_ring_write(&g_ipc, &rec);
    }
}

static int
shim_configure_services(shim_tool_data_t* data)
{
    if(shim_expect_status(rocprofiler_create_context(&data->context), "rocprofiler_create_context"))
        return -1;

    if(shim_expect_status(rocprofiler_create_buffer(data->context,
                                                    SHIM_BUFFER_SIZE_BYTES,
                                                    SHIM_BUFFER_WATERMARK_BYTES,
                                                    ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                    &shim_sdk_buffer_callback,
                                                    data,
                                                    &data->buffer),
                          "rocprofiler_create_buffer"))
        return -1;

    if(shim_expect_status(
           rocprofiler_configure_buffer_tracing_service(
               data->context, ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION, NULL, 0, data->buffer),
           "configure runtime initialization tracing"))
        return -1;

    if(shim_expect_status(
           rocprofiler_configure_buffer_tracing_service(
               data->context, ROCPROFILER_BUFFER_TRACING_HSA_CORE_API, NULL, 0, data->buffer),
           "configure HSA core API tracing"))
        return -1;

    if(shim_expect_status(
           rocprofiler_configure_buffer_tracing_service(
               data->context, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT, NULL, 0, data->buffer),
           "configure HIP runtime API ext tracing"))
        return -1;

    if(shim_expect_status(rocprofiler_start_context(data->context), "rocprofiler_start_context"))
        return -1;

    data->context_started = 1;

    if(g_ipc_ok && g_ipc.ctrl != NULL)
    {
        g_ipc.ctrl->active_services = 3;
    }

    return 0;
}

static int
shim_tool_init(rocprofiler_client_finalize_t finalize_func, void* tool_data)
{
    shim_tool_data_t* data = (shim_tool_data_t*) tool_data;

    if(data == NULL) return -1;
    if(shim_ensure_ipc_ready() != 0) return -1;

    data->finalize_func = finalize_func;
    return shim_configure_services(data);
}

static void
shim_tool_fini(void* tool_data)
{
    shim_tool_data_t* data = (shim_tool_data_t*) tool_data;

    if(data == NULL) return;

    if(data->buffer.handle != 0) (void) rocprofiler_flush_buffer(data->buffer);
    if(data->context_started) (void) rocprofiler_stop_context(data->context);
    if(data->buffer.handle != 0) (void) rocprofiler_destroy_buffer(data->buffer);

    memset(data, 0, sizeof(*data));
}

static rocprofiler_tool_configure_result_t g_configure_result = {
    sizeof(rocprofiler_tool_configure_result_t),
    &shim_tool_init,
    &shim_tool_fini,
    &g_tool_data,
};

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* client_id)
{
    (void) version;
    (void) runtime_version;

    if(!shim_is_enabled()) return NULL;
    if(priority > 0) return NULL;
    if(client_id == NULL || client_id->size < sizeof(*client_id)) return NULL;

    client_id->name = "rocprofiler-sdk-shim-concept";
    return &g_configure_result;
}

__attribute__((destructor)) static void
shim_register_dtor(void)
{
    if(g_ipc_ok) shim_ipc_destroy(&g_ipc);
}
