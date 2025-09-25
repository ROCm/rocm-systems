/**
 * @file counter_registry.h
 * @brief Static counter registry definitions for performance monitoring
 */

#ifndef COUNTER_REGISTRY_H
#define COUNTER_REGISTRY_H

#include "aql_structures.h"

/* Counter ID enumeration - stable identifiers across architectures */
typedef enum {
    /* GL2C Block Counters */
    COUNTER_GL2C_EA_RDREQ = 1,
    COUNTER_GL2C_EA_RDREQ_128B,
    COUNTER_GL2C_EA_RDREQ_32B,
    COUNTER_GL2C_EA_RDREQ_64B,
    COUNTER_GL2C_EA_WRREQ,
    COUNTER_GL2C_EA_WRREQ_64B,
    COUNTER_GL2C_EA_WRREQ_STALL,
    COUNTER_GL2C_HIT,
    COUNTER_GL2C_MISS,

    /* SQ Block Counters */
    COUNTER_SQC_LDS_BANK_CONFLICT,
    COUNTER_SQC_LDS_IDX_ACTIVE,
    COUNTER_SQ_ACCUM_PREV,
    COUNTER_SQ_BUSY_CYCLES,
    COUNTER_SQ_INSTS_FLAT,
    COUNTER_SQ_INSTS_LDS,
    COUNTER_SQ_INSTS_SALU,
    COUNTER_SQ_INSTS_SMEM,
    COUNTER_SQ_INSTS_TEX_LOAD,
    COUNTER_SQ_INSTS_TEX_STORE,
    COUNTER_SQ_INSTS_VALU,
    COUNTER_SQ_INSTS_WAVE32,
    COUNTER_SQ_INSTS_WAVE32_LDS,
    COUNTER_SQ_INSTS_WAVE32_VALU,
    COUNTER_SQ_INST_CYCLES_VMEM,
    COUNTER_SQ_INST_LEVEL_LDS,
    COUNTER_SQ_WAIT_ANY,
    COUNTER_SQ_WAIT_INST_ANY,
    COUNTER_SQ_WAVE32_INSTS,
    COUNTER_SQ_WAVE64_INSTS,
    COUNTER_SQ_WAVES,
    COUNTER_SQ_WAVE_CYCLES,

    /* TA Block Counters */
    COUNTER_TA_BUFFER_LOAD_WAVEFRONTS,
    COUNTER_TA_BUFFER_STORE_WAVEFRONTS,
    COUNTER_TA_TA_BUSY,

    /* GRBM Block Counters */
    COUNTER_GRBM_COUNT,
    COUNTER_GRBM_GUI_ACTIVE,

    COUNTER_ID_LAST
} counter_id_t;

/* Base counter definition - architecture agnostic */
typedef struct {
    counter_id_t id;
    const char* name;
    hardware_ip_block_t hw_block;
} counter_def_t;

/* Architecture-specific event mapping */
typedef struct {
    counter_id_t counter_id;
    uint32_t event_id;
} arch_event_map_t;

/* Function prototypes */
const counter_def_t* lookup_counter_by_name(const char* counter_name);
const counter_def_t* lookup_counter_by_id(counter_id_t id);
uint32_t lookup_event_id(counter_id_t id, arch_type_t arch);
const counter_def_t* get_all_counters(void);
size_t get_counter_count(void);

#endif /* COUNTER_REGISTRY_H */