/**
 * @file counter_registry.c
 * @brief Static counter registry implementation
 */

#include "counter_registry.h"
#include "gfx12_events.h"

#ifdef USERSPACE_BUILD
#include <string.h>
#include <stddef.h>
#else
#include <linux/string.h>
#include <linux/kernel.h>
#endif

/* Base counter definitions - architecture agnostic */
static const counter_def_t base_counters[] = {
    /* GL2C Block Counters */
    {COUNTER_GL2C_EA_RDREQ, "gl2c_ea_rdreq", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_RDREQ_128B, "gl2c_ea_rdreq_128b", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_RDREQ_32B, "gl2c_ea_rdreq_32b", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_RDREQ_64B, "gl2c_ea_rdreq_64b", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_WRREQ, "gl2c_ea_wrreq", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_WRREQ_64B, "gl2c_ea_wrreq_64b", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_WRREQ_STALL, "gl2c_ea_wrreq_stall", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_HIT, "gl2c_hit", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_MISS, "gl2c_miss", HW_IP_BLOCK_GL2C},

    /* SQ Block Counters */
    {COUNTER_SQC_LDS_BANK_CONFLICT, "sqc_lds_bank_conflict", HW_IP_BLOCK_SQ},
    {COUNTER_SQC_LDS_IDX_ACTIVE, "sqc_lds_idx_active", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_ACCUM_PREV, "sq_accum_prev", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_BUSY_CYCLES, "sq_busy_cycles", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_FLAT, "sq_insts_flat", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_LDS, "sq_insts_lds", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_SALU, "sq_insts_salu", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_SMEM, "sq_insts_smem", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_TEX_LOAD, "sq_insts_tex_load", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_TEX_STORE, "sq_insts_tex_store", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_VALU, "sq_insts_valu", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_WAVE32, "sq_insts_wave32", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_WAVE32_LDS, "sq_insts_wave32_lds", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_WAVE32_VALU, "sq_insts_wave32_valu", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INST_CYCLES_VMEM, "sq_inst_cycles_vmem", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INST_LEVEL_LDS, "sq_inst_level_lds", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAIT_ANY, "sq_wait_any", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAIT_INST_ANY, "sq_wait_inst_any", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAVE32_INSTS, "sq_wave32_insts", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAVE64_INSTS, "sq_wave64_insts", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAVES, "sq_waves", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAVE_CYCLES, "sq_wave_cycles", HW_IP_BLOCK_SQ},

    /* TA Block Counters */
    {COUNTER_TA_BUFFER_LOAD_WAVEFRONTS, "ta_buffer_load_wavefronts", HW_IP_BLOCK_TA},
    {COUNTER_TA_BUFFER_STORE_WAVEFRONTS, "ta_buffer_store_wavefronts", HW_IP_BLOCK_TA},
    {COUNTER_TA_TA_BUSY, "ta_ta_busy", HW_IP_BLOCK_TA},

    /* GRBM Block Counters */
    {COUNTER_GRBM_COUNT, "grbm_count", HW_IP_BLOCK_GRBM},
    {COUNTER_GRBM_GUI_ACTIVE, "grbm_gui_active", HW_IP_BLOCK_GRBM},
};

#define BASE_COUNTER_COUNT (sizeof(base_counters) / sizeof(counter_def_t))


/* Lookup functions */

/**
 * @brief Look up a counter definition by its string name
 *
 * Searches the base counter registry for a counter matching the given name.
 * Counter names are architecture-agnostic identifiers like "SQ_WAVES" or
 * "GL2C_HIT" that map to hardware-specific event IDs via the event map.
 *
 * This function provides a simpler C-based alternative to the event lookup
 * mechanisms in projects/aqlprofile/src/core/pm4_factory.h:158-184 which
 * uses C++ BlockInfoMap and GpuBlockInfo structures.
 *
 * @param counter_name String name of the counter (e.g., "SQ_WAVES")
 * @return Pointer to counter definition, or NULL if not found
 *
 * @note Counter names are case-sensitive
 * @note The returned pointer is valid for the lifetime of the program
 * @see lookup_counter_by_id(), lookup_event_id(), BlockInfoMap::Find in
 *      projects/aqlprofile/src/core/pm4_factory.h:89
 */
const counter_def_t* lookup_counter_by_name(const char* counter_name) {
    if (!counter_name) return NULL;

    for (size_t i = 0; i < BASE_COUNTER_COUNT; i++) {
        if (strcmp(base_counters[i].name, counter_name) == 0) {
            return &base_counters[i];
        }
    }
    return NULL;
}

/**
 * @brief Look up a counter definition by its numeric ID
 *
 * Searches the base counter registry for a counter with the specified ID.
 * Counter IDs are stable numeric identifiers defined in the counter_id_t enum.
 *
 * @param id Numeric counter identifier from counter_id_t enum
 * @return Pointer to counter definition, or NULL if not found or ID out of range
 *
 * @note The returned pointer is valid for the lifetime of the program
 * @see lookup_counter_by_name(), lookup_event_id()
 */
const counter_def_t* lookup_counter_by_id(counter_id_t id) {
    if (id <= 0 || id >= COUNTER_ID_LAST) return NULL;

    for (size_t i = 0; i < BASE_COUNTER_COUNT; i++) {
        if (base_counters[i].id == id) {
            return &base_counters[i];
        }
    }
    return NULL;
}

/**
 * @brief Look up the architecture-specific hardware event ID for a counter
 *
 * Translates an architecture-agnostic counter definition to the specific
 * hardware event ID that should be programmed into the counter's SELECT register
 * for the given GPU architecture. Different architectures may use different
 * event IDs for the same logical counter.
 *
 * This function is analogous to the event ID lookup in GpuBlockInfo structures
 * in projects/aqlprofile/def/gpu_block_info.h and the select_value callback
 * functions used in projects/aqlprofile/src/pm4/pmc_builder.h:325-326
 *
 * @param counter Pointer to counter definition (from lookup_counter_by_name/id)
 * @param arch Architecture information containing the event map
 * @return Hardware event ID for this counter on this architecture, or 0 if not found
 *
 * @note Event ID 0 typically indicates an error or unmapped counter
 * @note The event map is architecture-specific (e.g., GFX12 has different IDs than GFX11)
 * @see lookup_counter_by_name(), GpuBlockInfo::select_value in
 *      projects/aqlprofile/def/gpu_block_info.h
 */
uint32_t lookup_event_id(const counter_def_t* counter, const arch_t* arch) {
    if (!counter || !arch) return 0;

    /* Use the event map from the architecture structure */
    if (!arch->event_map || arch->event_count == 0) return 0;

    for (size_t i = 0; i < arch->event_count; i++) {
        if (arch->event_map[i].counter_id == counter->id) {
            return arch->event_map[i].event_id;
        }
    }
    return 0;
}

/**
 * @brief Get pointer to the full array of counter definitions
 *
 * Returns a pointer to the base counter array containing all registered
 * architecture-agnostic counter definitions.
 *
 * @return Pointer to array of counter definitions
 *
 * @note Array size can be obtained via get_counter_count()
 * @note Array is valid for the lifetime of the program
 * @see get_counter_count()
 */
const counter_def_t* get_all_counters(void) {
    return base_counters;
}

/**
 * @brief Get the total number of registered counters
 *
 * @return Number of entries in the base counter array
 *
 * @see get_all_counters()
 */
size_t get_counter_count(void) {
    return BASE_COUNTER_COUNT;
}