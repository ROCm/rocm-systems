#define _GNU_SOURCE

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <rocprofiler-sdk/registration.h>

#include "shim_ipc.h"
#include "shim_protocol.h"
#include "shim_runtime_bridge.h"

#define SHIM_NUM_OPS SHIM_MAX_TOTAL_OPS
#define SHIM_CORR_STACK_MAX 32

typedef struct
{
    uint64_t internal;
    uint64_t external;
} shim_corr_frame_t;

static _Atomic(void*)    g_runtime_original[SHIM_NUM_OPS];
static _Atomic(void*)    g_next_in_chain[SHIM_NUM_OPS];
static _Atomic uint32_t  g_local_op_mode[SHIM_NUM_OPS];
static _Atomic uint32_t  g_local_total_ops = 0;
static pthread_mutex_t   g_shim_install_lock = PTHREAD_MUTEX_INITIALIZER;
static int               g_shim_installed[SHIM_NUM_OPS];
static shim_ipc_target_t g_ipc;
static int               g_ipc_ok = 0;

static _Atomic uint64_t                g_next_internal  = 0;
static _Thread_local int               tls_corr_depth   = 0;
static _Thread_local int               tls_corr_dropped = 0;
static _Thread_local shim_corr_frame_t tls_corr_stack[SHIM_CORR_STACK_MAX];
static _Thread_local int               tls_ext_depth = 0;
static _Thread_local uint64_t          tls_ext_stack[SHIM_CORR_STACK_MAX];

static _Atomic uint32_t*
shim_op_mode_ptr(int op)
{
    if(g_ipc_ok && g_ipc.ctrl) return &g_ipc.ctrl->op_mode[op];
    return &g_local_op_mode[op];
}

static uint64_t
shim_current_external(void)
{
    return (tls_ext_depth > 0) ? tls_ext_stack[tls_ext_depth - 1] : 0;
}

static shim_correlation_id_t
shim_push_correlation(void)
{
    uint64_t id  = atomic_fetch_add_explicit(&g_next_internal, 1, memory_order_relaxed) + 1;
    uint64_t par = (tls_corr_depth > 0) ? tls_corr_stack[tls_corr_depth - 1].internal : 0;
    uint64_t ext = shim_current_external();

    if(tls_corr_depth < SHIM_CORR_STACK_MAX)
    {
        tls_corr_stack[tls_corr_depth].internal = id;
        tls_corr_stack[tls_corr_depth].external = ext;
        ++tls_corr_depth;
    }
    else
    {
        ++tls_corr_dropped;
    }

    return (shim_correlation_id_t){id, ext, par};
}

static void
shim_pop_correlation(void)
{
    if(tls_corr_dropped > 0)
    {
        --tls_corr_dropped;
        return;
    }

    if(tls_corr_depth > 0) --tls_corr_depth;
}

static void
shim_emit_record(uint32_t                     slot_idx,
                 uint32_t                     phase,
                 const shim_correlation_id_t* corr,
                 const void*                  packed_args,
                 uint32_t                     arg_bytes)
{
    if(!g_ipc_ok) return;

    shim_record_t rec;
    memset(&rec, 0, sizeof(rec));
    rec.tsc            = shim_rdtsc();
    rec.kind           = 0;
    rec.op             = slot_idx;
    rec.phase          = phase;
    rec.thread_id      = (uint64_t) gettid();
    rec.correlation_id = *corr;
    rec.slot_idx       = slot_idx;

    if(packed_args && arg_bytes > 0)
    {
        uint32_t copy_bytes = (arg_bytes < SHIM_RECORD_ARG_BYTES) ? arg_bytes : SHIM_RECORD_ARG_BYTES;
        memcpy(rec.args, packed_args, copy_bytes);
        rec.arg_bytes    = copy_bytes;
        rec.arg_overflow = (copy_bytes < arg_bytes) ? 1U : 0U;
    }

    (void) shim_ring_write(&g_ipc, &rec);
}

static int
shim_check_value_filter(uint32_t slot_idx, const void* packed_args)
{
    if(!g_ipc_ok || !g_ipc.ctrl) return 1;

    uint32_t count = g_ipc.ctrl->value_filter_count[slot_idx];
    if(count == 0) return 1;

    for(uint32_t i = 0; i < count && i < SHIM_MAX_VALUE_RULES_PER_OP; ++i)
    {
        const shim_value_rule_t* rule = &g_ipc.ctrl->value_rules[slot_idx][i];
        uint64_t                 value = 0;
        int                      pass  = 0;

        if(packed_args && rule->arg_index == 0) value = *(const uint64_t*) packed_args;

        switch(rule->comparison)
        {
            case SHIM_CMP_EQ: pass = (value == rule->operand); break;
            case SHIM_CMP_NEQ: pass = (value != rule->operand); break;
            case SHIM_CMP_GT: pass = (value > rule->operand); break;
            case SHIM_CMP_LT: pass = (value < rule->operand); break;
            case SHIM_CMP_BITMASK: pass = ((value & rule->operand) != 0); break;
            default: pass = 1; break;
        }

        if(!pass) return 0;
    }

    return 1;
}

__attribute__((visibility("default"))) uint32_t
shim_load_mode_for_slot(uint32_t slot)
{
    if(slot >= SHIM_NUM_OPS) return ROCP_SHIM_MODE_OFF;
    return atomic_load_explicit(shim_op_mode_ptr((int) slot), memory_order_acquire);
}

__attribute__((visibility("default"))) int
shim_should_trace_slot(uint32_t slot, const void* packed_args)
{
    if(slot >= SHIM_NUM_OPS) return 0;
    if(g_ipc_ok && g_ipc.ctrl && !shim_filter_test(g_ipc.ctrl->name_filter, slot)) return 0;
    return shim_check_value_filter(slot, packed_args);
}

__attribute__((visibility("default"))) void
shim_emit_trace_record(uint32_t                     slot_idx,
                       uint32_t                     phase,
                       const shim_correlation_id_t* corr,
                       const void*                  packed_args,
                       uint32_t                     arg_bytes)
{
    shim_emit_record(slot_idx, phase, corr, packed_args, arg_bytes);
}

__attribute__((visibility("default"))) int
shim_install_slot_wrapper(uint32_t slot, void** slot_ptr, void* wrapper)
{
    if(slot_ptr == NULL || wrapper == NULL || slot >= SHIM_NUM_OPS) return -1;

    pthread_mutex_lock(&g_shim_install_lock);

    if(!g_shim_installed[slot])
    {
        void* current = *slot_ptr;
        if(current != wrapper)
        {
            atomic_store_explicit(&g_runtime_original[slot], current, memory_order_release);
            atomic_store_explicit(&g_next_in_chain[slot], current, memory_order_release);
            atomic_store_explicit((_Atomic(void*)*) slot_ptr, wrapper, memory_order_release);
        }

        g_shim_installed[slot] = 1;
    }

    pthread_mutex_unlock(&g_shim_install_lock);
    return 0;
}

__attribute__((visibility("default"))) int
shim_register_table_metadata(const char*        name,
                             uint64_t           lib_version,
                             uint64_t           lib_instance,
                             uint32_t           n_ops,
                             const char* const* op_names,
                             uint32_t*          out_base)
{
    uint32_t base = 0;

    if(name == NULL || out_base == NULL || n_ops == 0) return -1;

    pthread_mutex_lock(&g_shim_install_lock);

    if(g_ipc_ok && g_ipc.ctrl)
    {
        uint32_t idx = g_ipc.ctrl->n_registrations;
        if(idx >= SHIM_MAX_REGISTRATIONS || (uint64_t) g_ipc.ctrl->total_ops + n_ops > SHIM_MAX_TOTAL_OPS)
        {
            pthread_mutex_unlock(&g_shim_install_lock);
            return -1;
        }

        base = g_ipc.ctrl->total_ops;
        *out_base = base;

        shim_table_registration_t* reg = &g_ipc.ctrl->registrations[idx];
        memset(reg, 0, sizeof(*reg));
        snprintf(reg->name, SHIM_TABLE_NAME_MAX, "%s", name);
        reg->lib_instance  = (uint32_t) lib_instance;
        reg->major_version = (uint32_t) (lib_version / 10000);
        reg->minor_version = (uint32_t) ((lib_version / 100) % 100);
        reg->slot_base     = base;
        reg->n_ops         = n_ops;

        for(uint32_t i = 0; i < n_ops; ++i)
        {
            shim_op_info_t* info = &g_ipc.ctrl->op_info[base + i];
            memset(info, 0, sizeof(*info));
            if(op_names && op_names[i]) snprintf(info->name, SHIM_OP_NAME_MAX, "%s", op_names[i]);
            info->n_args          = 0;
            info->arg_total_bytes = 0;
            shim_filter_set(g_ipc.ctrl->name_filter, base + i);
        }

        g_ipc.ctrl->n_registrations = idx + 1;
        g_ipc.ctrl->total_ops       = base + n_ops;
    }
    else
    {
        base      = atomic_fetch_add_explicit(&g_local_total_ops, n_ops, memory_order_relaxed);
        *out_base = base;
    }

    pthread_mutex_unlock(&g_shim_install_lock);
    return 0;
}

__attribute__((visibility("default"))) void*
shim_get_runtime_original(int op)
{
    if(op < 0 || op >= SHIM_NUM_OPS) return NULL;
    return atomic_load_explicit(&g_runtime_original[op], memory_order_acquire);
}

__attribute__((visibility("default"))) void*
shim_get_next_in_chain(int op)
{
    if(op < 0 || op >= SHIM_NUM_OPS) return NULL;
    return atomic_load_explicit(&g_next_in_chain[op], memory_order_acquire);
}

__attribute__((visibility("default"))) int
shim_set_next_in_chain(int op, void* sdk_wrapper)
{
    if(op < 0 || op >= SHIM_NUM_OPS) return -1;
    atomic_store_explicit(&g_next_in_chain[op], sdk_wrapper, memory_order_release);
    return 0;
}

__attribute__((visibility("default"))) shim_correlation_id_t
shim_push_correlation_public(void)
{
    return shim_push_correlation();
}

__attribute__((visibility("default"))) void
shim_pop_correlation_public(void)
{
    shim_pop_correlation();
}

__attribute__((visibility("default"))) int
rocprofiler_shim_push_external_correlation_id(uint64_t id)
{
    if(tls_ext_depth >= SHIM_CORR_STACK_MAX) return -1;
    tls_ext_stack[tls_ext_depth++] = id;
    return 0;
}

__attribute__((visibility("default"))) int
rocprofiler_shim_pop_external_correlation_id(uint64_t* out)
{
    if(tls_ext_depth == 0)
    {
        if(out) *out = 0;
        return -1;
    }

    --tls_ext_depth;
    if(out) *out = tls_ext_stack[tls_ext_depth];
    return 0;
}

__attribute__((visibility("default"))) uint64_t
shim_current_internal_id(void)
{
    return (tls_corr_depth > 0) ? tls_corr_stack[tls_corr_depth - 1].internal : 0;
}

static rocprofiler_tool_configure_result_t g_configure_result = {
    sizeof(rocprofiler_tool_configure_result_t),
    NULL,
    NULL,
    NULL,
};

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t version, const char* runtime_version, uint32_t priority, rocprofiler_client_id_t* client_id)
{
    (void) version;
    (void) runtime_version;
    (void) priority;

    if(client_id != NULL && client_id->size >= sizeof(*client_id)) client_id->name = "rocprofiler-sdk-shim-concept";

    return &g_configure_result;
}

__attribute__((constructor)) static void
shim_register_ctor(void)
{
    if(shim_ipc_init(&g_ipc) == 0) g_ipc_ok = 1;
}

__attribute__((destructor)) static void
shim_register_dtor(void)
{
    if(g_ipc_ok) shim_ipc_destroy(&g_ipc);
}
