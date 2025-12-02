/**
 * @file counter_registry.h
 * @brief Static counter registry definitions for performance monitoring
 */

#ifndef COUNTER_REGISTRY_H
#define COUNTER_REGISTRY_H

#include "aql_structures.h"

/* Forward declaration for dimension coordinates structure */
struct pmu_dimension_coords;

/*
 * Event ID Constraints
 * ====================
 *
 * Hardware event IDs are 9-bit values programmed into counter SELECT registers.
 * Event IDs exceeding this limit will be silently truncated, causing incorrect
 * counter behavior or reading the wrong event.
 *
 * This limit is enforced at runtime in lookup_event_id() and should also be
 * validated at compile-time in architecture-specific event definition files.
 */
#define EVENT_ID_MAX 0x1FF  /* 9-bit limit (0-511) */

/*
 * Dimension Capability Flags
 * ===========================
 *
 * These flags indicate which hardware dimensions a performance counter supports.
 * Different counter types have different granularities:
 *
 * - GRBM counters: Global, no dimension support (DIM_NONE)
 * - SQ counters: Per shader engine/array/WGP (DIM_SE_SA_WGP)
 * - GL2C counters: Per shader engine/array (DIM_SE_SA)
 * - TA counters: Per shader engine/array/WGP (DIM_SE_SA_WGP)
 *
 * When a user specifies dimensions not supported by a counter, the event
 * creation will fail with an appropriate error message.
 */
#define DIM_NONE        0x00  /* Global counter, no dimension support */
#define DIM_XCC         0x01  /* Per-XCC (Infinity Fabric block) */
#define DIM_SE          0x02  /* Per-Shader Engine */
#define DIM_SA          0x04  /* Per-Shader Array */
#define DIM_WGP         0x08  /* Per-Work Group Processor */
#define DIM_CU          0x10  /* Per-Compute Unit */
#define DIM_ALL         (DIM_XCC | DIM_SE | DIM_SA | DIM_WGP | DIM_CU)
#define DIM_SE_SA_WGP   (DIM_SE | DIM_SA | DIM_WGP)
#define DIM_SE_SA       (DIM_SE | DIM_SA)

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
    uint32_t supported_dimensions;   /* Bitmap of supported dimension flags (DIM_*) */
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
 * @param out_event_id Output: hardware event ID (valid if return is 0)
 * @return 0 on success, negative error code on failure (-EINVAL, -ENOENT, -ERANGE)
 */
int lookup_event_id(const counter_def_t* counter, const arch_t* arch, uint32_t* out_event_id);

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

/**
 * @brief Validate that a counter supports the requested dimensions
 *
 * Checks whether the specified counter supports the dimensions requested
 * in the dimension coordinates structure. This is used during event
 * initialization to reject invalid dimension specifications early.
 *
 * @param counter Pointer to counter definition
 * @param dims Requested dimension coordinates
 * @return 0 on success, -EINVAL if counter doesn't support requested dimensions
 *
 * Example:
 *   A GRBM counter (DIM_NONE) will fail if any dimension is specified.
 *   An SQ counter (DIM_SE_SA_WGP) will fail if XCC or CU is specified.
 */
int pmu_validate_counter_dimensions(const counter_def_t* counter,
                                    const struct pmu_dimension_coords* dims);

#endif /* COUNTER_REGISTRY_H */