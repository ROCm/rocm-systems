/**
 * @file gfx12_events.c
 * @brief GFX12 architecture-specific event mappings implementation
 */

#include "gfx12_events.h"

#ifdef USERSPACE_BUILD
#include <stddef.h>
#else
#include <linux/kernel.h>
#endif

/* GFX12 architecture-specific event mappings */
static const struct arch_event_map gfx12_events[] = {
    /* GL2C Block Events */
    {COUNTER_GL2C_EA_RDREQ, 140},
    {COUNTER_GL2C_EA_RDREQ_128B, 148},
    {COUNTER_GL2C_EA_RDREQ_32B, 146},
    {COUNTER_GL2C_EA_RDREQ_64B, 147},
    {COUNTER_GL2C_EA_WRREQ, 108},
    {COUNTER_GL2C_EA_WRREQ_64B, 114},
    {COUNTER_GL2C_EA_WRREQ_STALL, 122},
    {COUNTER_GL2C_HIT, 41},
    {COUNTER_GL2C_MISS, 42},

    /* SQ Block Events */
    {COUNTER_SQC_LDS_BANK_CONFLICT, 288},
    {COUNTER_SQC_LDS_IDX_ACTIVE, 293},
    {COUNTER_SQ_ACCUM_PREV, 1},
    {COUNTER_SQ_BUSY_CYCLES, 3},
    {COUNTER_SQ_INSTS_FLAT, 44},
    {COUNTER_SQ_INSTS_LDS, 45},
    {COUNTER_SQ_INSTS_SALU, 46},
    {COUNTER_SQ_INSTS_SMEM, 47},
    {COUNTER_SQ_INSTS_TEX_LOAD, 54},
    {COUNTER_SQ_INSTS_TEX_STORE, 55},
    {COUNTER_SQ_INSTS_VALU, 50},
    {COUNTER_SQ_INSTS_WAVE32, 58},
    {COUNTER_SQ_INSTS_WAVE32_LDS, 60},
    {COUNTER_SQ_INSTS_WAVE32_VALU, 61},
    {COUNTER_SQ_INST_CYCLES_VMEM, 102},
    {COUNTER_SQ_INST_LEVEL_LDS, 75},
    {COUNTER_SQ_WAIT_ANY, 27},
    {COUNTER_SQ_WAIT_INST_ANY, 26},
    {COUNTER_SQ_WAVE32_INSTS, 70},
    {COUNTER_SQ_WAVE64_INSTS, 71},
    {COUNTER_SQ_WAVES, 4},
    {COUNTER_SQ_WAVE_CYCLES, 24},

    /* TA Block Events */
    {COUNTER_TA_BUFFER_LOAD_WAVEFRONTS, 45},
    {COUNTER_TA_BUFFER_STORE_WAVEFRONTS, 46},
    {COUNTER_TA_TA_BUSY, 15},

    /* GRBM Block Events */
    {COUNTER_GRBM_COUNT, 0},
    {COUNTER_GRBM_GUI_ACTIVE, 2},
};

#define GFX12_EVENT_COUNT (sizeof(gfx12_events) / sizeof(struct arch_event_map))

/**
 * @brief Get the GFX12 event mapping array
 *
 * Returns a pointer to the static array mapping architecture-agnostic counter
 * IDs to GFX12-specific hardware event IDs. This array is used by the counter
 * registry to look up the correct event ID for programming SELECT registers.
 *
 * This function is analogous to the event ID lookup tables in
 * projects/aqlprofile/def/gfx12/ event definition files.
 *
 * @return Pointer to const array of arch_event_map structures
 *
 * @note Array size can be obtained via get_gfx12_event_count()
 * @note Array is valid for the lifetime of the program
 * @see get_gfx12_event_count(), lookup_event_id() in counter_registry.c
 */
const struct arch_event_map* get_gfx12_events(void) {
    return gfx12_events;
}

/**
 * @brief Get the number of event mappings in the GFX12 event array
 *
 * Returns the total count of counter-to-event mappings defined for GFX12.
 * Used to iterate over the event map or validate array bounds.
 *
 * @return Number of event mappings in the GFX12 array
 *
 * @see get_gfx12_events()
 */
size_t get_gfx12_event_count(void) {
    return GFX12_EVENT_COUNT;
}