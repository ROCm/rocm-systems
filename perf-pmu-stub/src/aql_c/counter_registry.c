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
    {COUNTER_GL2C_EA_RDREQ, "GL2C_EA_RDREQ", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_RDREQ_128B, "GL2C_EA_RDREQ_128B", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_RDREQ_32B, "GL2C_EA_RDREQ_32B", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_RDREQ_64B, "GL2C_EA_RDREQ_64B", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_WRREQ, "GL2C_EA_WRREQ", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_WRREQ_64B, "GL2C_EA_WRREQ_64B", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_EA_WRREQ_STALL, "GL2C_EA_WRREQ_STALL", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_HIT, "GL2C_HIT", HW_IP_BLOCK_GL2C},
    {COUNTER_GL2C_MISS, "GL2C_MISS", HW_IP_BLOCK_GL2C},

    /* SQ Block Counters */
    {COUNTER_SQC_LDS_BANK_CONFLICT, "SQC_LDS_BANK_CONFLICT", HW_IP_BLOCK_SQ},
    {COUNTER_SQC_LDS_IDX_ACTIVE, "SQC_LDS_IDX_ACTIVE", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_ACCUM_PREV, "SQ_ACCUM_PREV", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_BUSY_CYCLES, "SQ_BUSY_CYCLES", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_FLAT, "SQ_INSTS_FLAT", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_LDS, "SQ_INSTS_LDS", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_SALU, "SQ_INSTS_SALU", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_SMEM, "SQ_INSTS_SMEM", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_TEX_LOAD, "SQ_INSTS_TEX_LOAD", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_TEX_STORE, "SQ_INSTS_TEX_STORE", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_VALU, "SQ_INSTS_VALU", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_WAVE32, "SQ_INSTS_WAVE32", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_WAVE32_LDS, "SQ_INSTS_WAVE32_LDS", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INSTS_WAVE32_VALU, "SQ_INSTS_WAVE32_VALU", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INST_CYCLES_VMEM, "SQ_INST_CYCLES_VMEM", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_INST_LEVEL_LDS, "SQ_INST_LEVEL_LDS", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAIT_ANY, "SQ_WAIT_ANY", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAIT_INST_ANY, "SQ_WAIT_INST_ANY", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAVE32_INSTS, "SQ_WAVE32_INSTS", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAVE64_INSTS, "SQ_WAVE64_INSTS", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAVES, "SQ_WAVES", HW_IP_BLOCK_SQ},
    {COUNTER_SQ_WAVE_CYCLES, "SQ_WAVE_CYCLES", HW_IP_BLOCK_SQ},

    /* TA Block Counters */
    {COUNTER_TA_BUFFER_LOAD_WAVEFRONTS, "TA_BUFFER_LOAD_WAVEFRONTS", HW_IP_BLOCK_TA},
    {COUNTER_TA_BUFFER_STORE_WAVEFRONTS, "TA_BUFFER_STORE_WAVEFRONTS", HW_IP_BLOCK_TA},
    {COUNTER_TA_TA_BUSY, "TA_TA_BUSY", HW_IP_BLOCK_TA},

    /* GRBM Block Counters */
    {COUNTER_GRBM_COUNT, "GRBM_COUNT", HW_IP_BLOCK_GRBM},
    {COUNTER_GRBM_GUI_ACTIVE, "GRBM_GUI_ACTIVE", HW_IP_BLOCK_GRBM},
};

#define BASE_COUNTER_COUNT (sizeof(base_counters) / sizeof(counter_def_t))


/* Lookup functions */
const counter_def_t* lookup_counter_by_name(const char* counter_name) {
    if (!counter_name) return NULL;

    for (size_t i = 0; i < BASE_COUNTER_COUNT; i++) {
        if (strcmp(base_counters[i].name, counter_name) == 0) {
            return &base_counters[i];
        }
    }
    return NULL;
}

const counter_def_t* lookup_counter_by_id(counter_id_t id) {
    if (id <= 0 || id >= COUNTER_ID_LAST) return NULL;

    for (size_t i = 0; i < BASE_COUNTER_COUNT; i++) {
        if (base_counters[i].id == id) {
            return &base_counters[i];
        }
    }
    return NULL;
}

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

const counter_def_t* get_all_counters(void) {
    return base_counters;
}

size_t get_counter_count(void) {
    return BASE_COUNTER_COUNT;
}