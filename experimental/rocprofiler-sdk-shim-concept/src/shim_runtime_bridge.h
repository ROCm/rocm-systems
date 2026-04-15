#ifndef SHIM_RUNTIME_BRIDGE_H
#define SHIM_RUNTIME_BRIDGE_H

#include <stdint.h>

#ifndef SHIM_PROTOCOL_H
#    define ROCP_SHIM_MODE_OFF         0
#    define ROCP_SHIM_MODE_RECORD      1
#    define ROCP_SHIM_MODE_RECORD_ARGS 2
#    define ROCP_SHIM_MODE_RECORD_FULL 3

#    define SHIM_PHASE_ENTER          0
#    define SHIM_PHASE_EXIT           1
#    define SHIM_PHASE_EXIT_UNREACHED 2

typedef struct
{
    uint64_t internal;
    uint64_t external;
    uint64_t ancestor;
} shim_correlation_id_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

int shim_register_table_metadata(const char*        name,
                                 uint64_t           lib_version,
                                 uint64_t           lib_instance,
                                 uint32_t           n_ops,
                                 const char* const* op_names,
                                 uint32_t*          out_base);

int      shim_install_slot_wrapper(uint32_t slot, void** slot_ptr, void* wrapper);
uint32_t shim_load_mode_for_slot(uint32_t slot);
int      shim_should_trace_slot(uint32_t slot, const void* packed_args);
void     shim_emit_trace_record(uint32_t                    slot_idx,
                                uint32_t                    phase,
                                const shim_correlation_id_t* corr,
                                const void*                 packed_args,
                                uint32_t                    arg_bytes);

shim_correlation_id_t shim_push_correlation_public(void);
void                  shim_pop_correlation_public(void);
void*                 shim_get_runtime_original(int op);
void*                 shim_get_next_in_chain(int op);
int                   shim_set_next_in_chain(int op, void* sdk_wrapper);

#ifdef __cplusplus
}
#endif

#endif
