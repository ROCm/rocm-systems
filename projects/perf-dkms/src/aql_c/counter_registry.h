/**
 * @file counter_registry.h
 * @brief Static counter registry definitions for performance monitoring
 */

#ifndef COUNTER_REGISTRY_H
#define COUNTER_REGISTRY_H

#include "aql_structures.h"

/**
 * @brief Counter ID enumeration - stable identifiers across architectures
 *
 * Each counter has a unique numeric ID that remains constant across different
 * GPU architectures. Architecture-specific event IDs are then mapped to these
 * stable counter IDs via the event map.
 *
 * This provides a level of indirection allowing counter names and IDs to remain
 * consistent while the underlying hardware event codes change between GFX9,
 * GFX10, GFX11, GFX12, etc.
 */
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

/**
 * @brief Base counter definition - architecture agnostic
 *
 * Defines the basic properties of a performance counter that are the same
 * across all GPU architectures: its unique ID, human-readable name, and
 * which hardware block it belongs to.
 */
typedef struct {
    counter_id_t id;                 /* Unique numeric identifier */
    const char* name;                /* Counter name (e.g., "SQ_WAVES") */
    hardware_ip_block_t hw_block;    /* Hardware block this counter belongs to */
} counter_def_t;

/**
 * @brief Architecture-specific event mapping
 *
 * Maps an architecture-agnostic counter ID to the specific hardware event
 * ID for a particular GPU architecture (e.g., GFX12). Different architectures
 * may use different event IDs for the same logical counter.
 *
 * This structure is analogous to the event select values in
 * projects/aqlprofile/def/gfx12/ event definition files.
 */
struct arch_event_map {
    counter_id_t counter_id;         /* Architecture-agnostic counter ID */
    uint32_t event_id;               /* Architecture-specific hardware event ID */
};

/* Function prototypes */

/**
 * @brief Look up a counter definition by its string name
 *
 * @param counter_name String name of the counter (e.g., "SQ_WAVES")
 * @return Pointer to counter definition, or NULL if not found
 */
const counter_def_t* lookup_counter_by_name(const char* counter_name);

/**
 * @brief Look up a counter definition by its numeric ID
 *
 * @param id Numeric counter identifier from counter_id_t enum
 * @return Pointer to counter definition, or NULL if not found
 */
const counter_def_t* lookup_counter_by_id(counter_id_t id);

/**
 * @brief Look up the architecture-specific hardware event ID for a counter
 *
 * @param counter Pointer to counter definition
 * @param arch Architecture information containing the event map
 * @return Hardware event ID, or 0 if not found
 */
uint32_t lookup_event_id(const counter_def_t* counter, const arch_t* arch);

/**
 * @brief Get pointer to the full array of counter definitions
 *
 * @return Pointer to array of counter definitions
 */
const counter_def_t* get_all_counters(void);

/**
 * @brief Get the total number of registered counters
 *
 * @return Number of entries in the base counter array
 */
size_t get_counter_count(void);

#endif /* COUNTER_REGISTRY_H */