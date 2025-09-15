/**
 * @file aql_gfx9_ops.c
 * @brief GFX9 (Vega) Architecture Operations Implementation
 *
 * This file implements the architecture-specific operations for GFX9 (Vega).
 * Key differences from GFX12:
 * - Supports PRED_EXEC packets (unique to GFX9)
 * - Uses 7-dword ACQUIRE_MEM packets instead of 8-dword
 * - Has UCONFIG register space
 * - Different cache coherence mechanisms
 */

#include "aql_arch_ops.h"
#include "aql_gfx9_defs.h"
#include "aql_cmd_buffer.h"

#ifdef __KERNEL__
#include <linux/kernel.h>
#include <linux/string.h>
#else
#include <string.h>
#include <stdio.h>
#endif

/*
 * Forward declarations for GFX9 operations
 */
static aql_result_t aql_gfx9_build_write_uconfig_reg(aql_cmd_buffer_t* buf,
                                                     uint32_t addr, uint32_t value);
static aql_result_t aql_gfx9_build_write_config_reg(aql_cmd_buffer_t* buf,
                                                    uint32_t addr, uint32_t value);
static aql_result_t aql_gfx9_build_write_sh_reg(aql_cmd_buffer_t* buf,
                                                uint32_t addr, uint32_t value);
static aql_result_t aql_gfx9_build_copy_reg_data(aql_cmd_buffer_t* buf,
                                                 uint32_t src_addr, void* dst_addr,
                                                 uint32_t size, bool wait);
static aql_result_t aql_gfx9_build_indirect_buffer(aql_cmd_buffer_t* buf,
                                                   const void* cmd_addr, size_t cmd_size);
static aql_result_t aql_gfx9_build_wait_reg_mem(aql_cmd_buffer_t* buf, bool mem_space,
                                                uint64_t wait_addr, bool func_eq,
                                                uint32_t mask_val, uint32_t wait_val);
static aql_result_t aql_gfx9_build_wait_idle(aql_cmd_buffer_t* buf);
static aql_result_t aql_gfx9_build_nop(aql_cmd_buffer_t* buf, uint32_t num_dwords);
static aql_result_t aql_gfx9_build_cache_flush(aql_cmd_buffer_t* buf,
                                               size_t addr, size_t size);
static aql_result_t aql_gfx9_build_pred_exec(aql_cmd_buffer_t* buf,
                                             uint32_t xcc_select, uint32_t exec_count);

/* Architecture-specific helper functions */
static uint32_t aql_gfx9_make_grbm_index(uint32_t se, uint32_t sa,
                                         uint32_t wgp, uint32_t instance);
static uint32_t aql_gfx9_get_cache_policy(void);
static uint64_t aql_gfx9_get_smn_address(uint64_t base_addr, uint32_t aid_index);

/* Counter and register management functions */
static bool aql_gfx9_is_valid_counter_reg(aql_block_id_t block_id, uint32_t reg_addr);
static const aql_counter_block_regs_t* aql_gfx9_get_counter_block_regs(aql_block_id_t block_id);
static aql_result_t aql_gfx9_get_counter_register_addr(aql_block_id_t block_id,
                                                       uint32_t counter_id,
                                                       uint32_t reg_type,
                                                       uint32_t* reg_addr);
static aql_result_t aql_gfx9_validate_counter_request(const aql_counter_request_t* request);

/*
 * GFX9 Operations Table
 */
const aql_arch_ops_t aql_gfx9_ops = {
    /* Architecture identification */
    .arch_name = AQL_GFX9_ARCH_NAME,
    .arch_type = AQL_ARCH_GFX9,
    .gfx_version = AQL_GFX9_VERSION,

    /* Architecture capabilities */
    .has_pred_exec = AQL_GFX9_HAS_PRED_EXEC,
    .has_uconfig_space = AQL_GFX9_HAS_UCONFIG_SPACE,
    .has_dual_sdma = AQL_GFX9_HAS_DUAL_SDMA,
    .has_gl_cache_hierarchy = AQL_GFX9_HAS_GL_CACHE_HIERARCHY,
    .has_smn_addressing = AQL_GFX9_HAS_SMN_ADDRESSING,

    /* Register space definitions */
    .register_spaces = {
        .config_space_start = AQL_GFX9_CONFIG_SPACE_START,
        .config_space_end = AQL_GFX9_CONFIG_SPACE_END,
        .uconfig_space_start = AQL_GFX9_UCONFIG_SPACE_START,
        .uconfig_space_end = AQL_GFX9_UCONFIG_SPACE_END,
        .has_uconfig_space = true
    },

    /* Counter block information - now using real hardware register definitions */
    .counter_blocks = (const void*)aql_gfx9_counter_blocks,
    .counter_block_count = (sizeof(aql_gfx9_counter_blocks) / sizeof(aql_gfx9_counter_blocks[0])) - 1, /* Exclude terminator */

    /* PM4 command generation functions */
    .build_write_uconfig_reg = aql_gfx9_build_write_uconfig_reg,
    .build_write_config_reg = aql_gfx9_build_write_config_reg,
    .build_write_sh_reg = aql_gfx9_build_write_sh_reg,
    .build_copy_reg_data = aql_gfx9_build_copy_reg_data,
    .build_indirect_buffer = aql_gfx9_build_indirect_buffer,
    .build_wait_reg_mem = aql_gfx9_build_wait_reg_mem,
    .build_wait_idle = aql_gfx9_build_wait_idle,
    .build_nop = aql_gfx9_build_nop,
    .build_cache_flush = aql_gfx9_build_cache_flush,
    .build_pred_exec = aql_gfx9_build_pred_exec,

    /* Architecture-specific helper functions */
    .make_grbm_index = aql_gfx9_make_grbm_index,
    .get_cache_policy = aql_gfx9_get_cache_policy,
    .get_smn_address = aql_gfx9_get_smn_address,

    /* Counter and register management functions */
    .is_valid_counter_reg = aql_gfx9_is_valid_counter_reg,
    .get_counter_block_regs = aql_gfx9_get_counter_block_regs,
    .get_counter_register_addr = aql_gfx9_get_counter_register_addr,
    .validate_counter_request = aql_gfx9_validate_counter_request
};

/*
 * PM4 Command Generation Functions
 */

static aql_result_t aql_gfx9_build_write_uconfig_reg(aql_cmd_buffer_t* buf,
                                                     uint32_t addr, uint32_t value) {
    uint32_t packet[AQL_GFX9_PKT3_WRITE_UCONFIG_REG_SIZE];

    if (!buf) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Validate UCONFIG register address */
    if (addr < AQL_GFX9_UCONFIG_SPACE_START || addr > AQL_GFX9_UCONFIG_SPACE_END) {
        return AQL_ERROR_INVALID_REGISTER;
    }

    /* Build WRITE_UCONFIG_REG packet */
    packet[0] = AQL_GFX9_PKT3_HEADER(AQL_GFX9_WRITE_UCONFIG_REG,
                                     AQL_GFX9_PKT3_WRITE_UCONFIG_REG_SIZE - 1);
    packet[1] = (addr - AQL_GFX9_UCONFIG_SPACE_START) >> 2;
    packet[2] = value;

    return aql_cmd_buffer_append_dwords(buf, packet, AQL_GFX9_PKT3_WRITE_UCONFIG_REG_SIZE);
}

static aql_result_t aql_gfx9_build_write_config_reg(aql_cmd_buffer_t* buf,
                                                    uint32_t addr, uint32_t value) {
    uint32_t packet[AQL_GFX9_PKT3_WRITE_CONFIG_REG_SIZE];

    if (!buf) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Validate CONFIG register address */
    if (addr < AQL_GFX9_CONFIG_SPACE_START || addr > AQL_GFX9_CONFIG_SPACE_END) {
        return AQL_ERROR_INVALID_REGISTER;
    }

    /* Build WRITE_CONFIG_REG packet */
    packet[0] = AQL_GFX9_PKT3_HEADER(AQL_GFX9_WRITE_CONFIG_REG,
                                     AQL_GFX9_PKT3_WRITE_CONFIG_REG_SIZE - 1);
    packet[1] = (addr - AQL_GFX9_CONFIG_SPACE_START) >> 2;
    packet[2] = value;

    return aql_cmd_buffer_append_dwords(buf, packet, AQL_GFX9_PKT3_WRITE_CONFIG_REG_SIZE);
}

static aql_result_t aql_gfx9_build_write_sh_reg(aql_cmd_buffer_t* buf,
                                                uint32_t addr, uint32_t value) {
    uint32_t packet[AQL_GFX9_PKT3_WRITE_SH_REG_SIZE];

    if (!buf) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* For GFX9, SH registers use direct addressing */
    /* Build WRITE_SH_REG packet */
    packet[0] = AQL_GFX9_PKT3_HEADER(AQL_GFX9_WRITE_SH_REG,
                                     AQL_GFX9_PKT3_WRITE_SH_REG_SIZE - 1);
    packet[1] = addr >> 2;
    packet[2] = value;

    return aql_cmd_buffer_append_dwords(buf, packet, AQL_GFX9_PKT3_WRITE_SH_REG_SIZE);
}

static aql_result_t aql_gfx9_build_copy_reg_data(aql_cmd_buffer_t* buf,
                                                 uint32_t src_addr, void* dst_addr,
                                                 uint32_t size, bool wait) {
    uint32_t packet[AQL_GFX9_PKT3_COPY_REG_DATA_SIZE];
    uint64_t dst_addr_64 = (uint64_t)(uintptr_t)dst_addr;

    if (!buf || !dst_addr || size == 0) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Build COPY_REG_DATA packet */
    packet[0] = AQL_GFX9_PKT3_HEADER(AQL_GFX9_PKT3_COPY_REG_DATA,
                                     AQL_GFX9_PKT3_COPY_REG_DATA_SIZE - 1);
    packet[1] = AQL_GFX9_COPY_REG_DATA_SRC_SEL_REG |
                AQL_GFX9_COPY_REG_DATA_DST_SEL_MEM |
                (wait ? AQL_GFX9_COPY_REG_DATA_WR_CONFIRM : 0);
    packet[2] = src_addr >> 2;
    packet[3] = (uint32_t)(dst_addr_64 & 0xFFFFFFFF);
    packet[4] = (uint32_t)(dst_addr_64 >> 32);

    return aql_cmd_buffer_append_dwords(buf, packet, AQL_GFX9_PKT3_COPY_REG_DATA_SIZE);
}

static aql_result_t aql_gfx9_build_indirect_buffer(aql_cmd_buffer_t* buf,
                                                   const void* cmd_addr, size_t cmd_size) {
    uint32_t packet[AQL_GFX9_PKT3_INDIRECT_BUFFER_SIZE];
    uint64_t cmd_addr_64 = (uint64_t)(uintptr_t)cmd_addr;
    uint32_t ib_size_dwords;

    if (!buf || !cmd_addr || cmd_size == 0) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    if (cmd_size % 4 != 0) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    ib_size_dwords = (uint32_t)(cmd_size / 4);

    /* Build INDIRECT_BUFFER packet */
    packet[0] = AQL_GFX9_PKT3_HEADER(AQL_GFX9_PKT3_INDIRECT_BUFFER,
                                     AQL_GFX9_PKT3_INDIRECT_BUFFER_SIZE - 1);
    packet[1] = (uint32_t)(cmd_addr_64 & 0xFFFFFFFF);
    packet[2] = (uint32_t)(cmd_addr_64 >> 32);
    packet[3] = ib_size_dwords & 0xFFFFFF;

    return aql_cmd_buffer_append_dwords(buf, packet, AQL_GFX9_PKT3_INDIRECT_BUFFER_SIZE);
}

static aql_result_t aql_gfx9_build_wait_reg_mem(aql_cmd_buffer_t* buf, bool mem_space,
                                                uint64_t wait_addr, bool func_eq,
                                                uint32_t mask_val, uint32_t wait_val) {
    uint32_t packet[AQL_GFX9_PKT3_WAIT_REG_MEM_SIZE];
    uint32_t function = func_eq ? AQL_GFX9_WAIT_REG_MEM_FUNCTION_EQ :
                                  AQL_GFX9_WAIT_REG_MEM_FUNCTION_NE;
    uint32_t space = mem_space ? AQL_GFX9_WAIT_REG_MEM_SPACE_MEM :
                                 AQL_GFX9_WAIT_REG_MEM_SPACE_REG;

    if (!buf) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Build WAIT_REG_MEM packet */
    packet[0] = AQL_GFX9_PKT3_HEADER(AQL_GFX9_PKT3_WAIT_REG_MEM,
                                     AQL_GFX9_PKT3_WAIT_REG_MEM_SIZE - 1);
    packet[1] = (function << 0) | (space << 4);
    packet[2] = (uint32_t)(wait_addr & 0xFFFFFFFF);
    packet[3] = (uint32_t)(wait_addr >> 32);
    packet[4] = wait_val;
    packet[5] = mask_val;
    packet[6] = 4; /* Poll interval */

    return aql_cmd_buffer_append_dwords(buf, packet, AQL_GFX9_PKT3_WAIT_REG_MEM_SIZE);
}

static aql_result_t aql_gfx9_build_wait_idle(aql_cmd_buffer_t* buf) {
    /* For GFX9, we use ACQUIRE_MEM with appropriate cache flush bits */
    return aql_gfx9_build_cache_flush(buf, 0, 0);
}

static aql_result_t aql_gfx9_build_nop(aql_cmd_buffer_t* buf, uint32_t num_dwords) {
    uint32_t header;
    uint32_t i;
    aql_result_t result;

    if (!buf || num_dwords < AQL_GFX9_PKT3_NOP_MIN_SIZE) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Build NOP packet header */
    header = AQL_GFX9_PKT3_HEADER(AQL_GFX9_PKT3_NOP, num_dwords - 1);
    result = aql_cmd_buffer_append_dwords(buf, &header, 1);
    if (result != AQL_SUCCESS) {
        return result;
    }

    /* Fill remaining dwords with zeros */
    for (i = 1; i < num_dwords; i++) {
        uint32_t zero = 0;
        result = aql_cmd_buffer_append_dwords(buf, &zero, 1);
        if (result != AQL_SUCCESS) {
            return result;
        }
    }

    return AQL_SUCCESS;
}

static aql_result_t aql_gfx9_build_cache_flush(aql_cmd_buffer_t* buf,
                                               size_t addr, size_t size) {
    uint32_t packet[AQL_GFX9_PKT3_ACQUIRE_MEM_SIZE];
    uint32_t coher_cntl;

    if (!buf) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* GFX9 uses comprehensive cache flush via ACQUIRE_MEM */
    coher_cntl = AQL_GFX9_ACQUIRE_MEM_TCL1_ACTION_ENA |
                 AQL_GFX9_ACQUIRE_MEM_TC_ACTION_ENA |
                 AQL_GFX9_ACQUIRE_MEM_CB_ACTION_ENA |
                 AQL_GFX9_ACQUIRE_MEM_DB_ACTION_ENA |
                 AQL_GFX9_ACQUIRE_MEM_SH_KCACHE_ACTION_ENA |
                 AQL_GFX9_ACQUIRE_MEM_SH_ICACHE_ACTION_ENA;

    /* Build 7-dword ACQUIRE_MEM packet (GFX9 specific) */
    packet[0] = AQL_GFX9_PKT3_HEADER(AQL_GFX9_PKT3_ACQUIRE_MEM,
                                     AQL_GFX9_PKT3_ACQUIRE_MEM_SIZE - 1);
    packet[1] = coher_cntl;
    packet[2] = (uint32_t)(size & 0xFFFFFFFF);
    packet[3] = (uint32_t)((addr >> 8) & 0xFFFFFFFF);
    packet[4] = (uint32_t)((addr >> 40) & 0xFF);
    packet[5] = 0; /* POLL_INTERVAL */
    packet[6] = AQL_GFX9_ACQUIRE_MEM_ENGINE_ME;

    return aql_cmd_buffer_append_dwords(buf, packet, AQL_GFX9_PKT3_ACQUIRE_MEM_SIZE);
}

static aql_result_t aql_gfx9_build_pred_exec(aql_cmd_buffer_t* buf,
                                             uint32_t xcc_select, uint32_t exec_count) {
    uint32_t packet[AQL_GFX9_PKT3_PRED_EXEC_SIZE];

    if (!buf) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Build PRED_EXEC packet (GFX9 specific) */
    packet[0] = AQL_GFX9_PKT3_HEADER(AQL_GFX9_PKT3_PRED_EXEC,
                                     AQL_GFX9_PKT3_PRED_EXEC_SIZE - 1);
    packet[1] = (xcc_select & AQL_GFX9_PRED_EXEC_XCC_MASK);
    packet[2] = (exec_count << 8) & AQL_GFX9_PRED_EXEC_EXEC_COUNT_MASK;

    return aql_cmd_buffer_append_dwords(buf, packet, AQL_GFX9_PKT3_PRED_EXEC_SIZE);
}

/*
 * Architecture-Specific Helper Functions
 */

static uint32_t aql_gfx9_make_grbm_index(uint32_t se, uint32_t sa,
                                         uint32_t wgp, uint32_t instance) {
    /* GFX9 doesn't use WGP, only SE/SA/Instance */
    (void)wgp;

    return (se << AQL_GFX9_GRBM_GFX_INDEX_SE_SHIFT) |
           (sa << AQL_GFX9_GRBM_GFX_INDEX_SA_SHIFT) |
           (instance << AQL_GFX9_GRBM_GFX_INDEX_INSTANCE_SHIFT);
}

static uint32_t aql_gfx9_get_cache_policy(void) {
    return AQL_GFX9_CACHE_POLICY_LRU;
}

static uint64_t aql_gfx9_get_smn_address(uint64_t base_addr, uint32_t aid_index) {
    return AQL_GFX9_SMN_BASE + base_addr + ((uint64_t)aid_index << AQL_GFX9_SMN_AID_SHIFT);
}

/*
 * Counter and Register Management Functions
 */

static bool aql_gfx9_is_valid_counter_reg(aql_block_id_t block_id, uint32_t reg_addr) {
    const aql_gfx9_counter_block_info_t* block_info;

    /* Find block information */
    for (block_info = aql_gfx9_counter_blocks; block_info->block_name != NULL; block_info++) {
        if (block_info->block_id == block_id) {
            /* Check if register is within the block's range */
            uint32_t block_base = block_info->perf_sel_base;
            uint32_t block_end = block_base + (block_info->counter_count * 4);
            return (reg_addr >= block_base && reg_addr < block_end);
        }
    }

    return false;
}

static const aql_counter_block_regs_t* aql_gfx9_get_counter_block_regs(aql_block_id_t block_id) {
    /* For now, return NULL - would need full counter block definitions */
    (void)block_id;
    return NULL;
}

static aql_result_t aql_gfx9_get_counter_register_addr(aql_block_id_t block_id,
                                                       uint32_t counter_id,
                                                       uint32_t reg_type,
                                                       uint32_t* reg_addr) {
    const aql_gfx9_counter_block_info_t* block_info;

    if (!reg_addr) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Find block information */
    for (block_info = aql_gfx9_counter_blocks; block_info->block_name != NULL; block_info++) {
        if (block_info->block_id == block_id) {
            if (counter_id >= block_info->counter_count) {
                return AQL_ERROR_INVALID_COUNTER;
            }

            /* For real hardware registers, use the direct register address
             * The perf_sel_base now contains the actual hardware register address
             * for counter 0, and other counters are offset by 1 (not 4) */
            switch (reg_type) {
                case AQL_REG_TYPE_SELECT:
                    *reg_addr = block_info->perf_sel_base + counter_id;
                    break;
                case AQL_REG_TYPE_ENABLE:
                    /* Enable register typically at base + 1 offset per counter */
                    *reg_addr = block_info->perf_sel_base + counter_id + 1;
                    break;
                case AQL_REG_TYPE_RESULT:
                    /* Result register typically at base + 2 offset per counter */
                    *reg_addr = block_info->perf_sel_base + counter_id + 2;
                    break;
                default:
                    return AQL_ERROR_INVALID_ARGUMENT;
            }
            return AQL_SUCCESS;
        }
    }

    return AQL_ERROR_INVALID_BLOCK;
}

static aql_result_t aql_gfx9_validate_counter_request(const aql_counter_request_t* request) {
    const aql_gfx9_counter_block_info_t* block_info;

    if (!request) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Find and validate block */
    for (block_info = aql_gfx9_counter_blocks; block_info->block_name != NULL; block_info++) {
        if (block_info->block_id == request->block_id) {
            /* Validate counter ID */
            if (request->counter_id >= block_info->counter_count) {
                return AQL_ERROR_INVALID_COUNTER;
            }

            /* Validate block instance */
            if (request->block_instance >= AQL_GFX9_MAX_SHADER_ENGINES) {
                return AQL_ERROR_INVALID_INSTANCE;
            }

            return AQL_SUCCESS;
        }
    }

    return AQL_ERROR_INVALID_BLOCK;
}