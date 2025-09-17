/**
 * @file aql_gfx12_ops.c
 * @brief GFX12 (RDNA3) architecture operations implementation
 *
 * This file implements the architecture-specific operations for GFX12 GPUs.
 * GFX12 is the RDNA3 architecture with enhanced cache hierarchy, dual SDMA
 * engines, and workgroup processor support.
 *
 * Key GFX12 characteristics:
 * - No PRED_EXEC support (RDNA3 feature removal)
 * - Enhanced GL1/GL2 cache hierarchy
 * - Workgroup Processor (WGP) addressing support
 * - 8-dword ACQUIRE_MEM packets (vs 7 in GFX9)
 * - Dual SDMA engines
 * - UCONFIG register space available
 */

#include "aql_types.h"
#include "aql_arch_ops.h"
#include "aql_cmd_buffer.h"
#include "aql_gfx12_defs.h"

/*
 * Forward declarations for internal functions
 */
static aql_result_t aql_gfx12_build_write_uconfig_reg(aql_cmd_buffer_t* buf,
                                                     uint32_t addr, uint32_t value);
static aql_result_t aql_gfx12_build_write_config_reg(aql_cmd_buffer_t* buf,
                                                    uint32_t addr, uint32_t value);
static aql_result_t aql_gfx12_build_write_sh_reg(aql_cmd_buffer_t* buf,
                                                uint32_t addr, uint32_t value);
static aql_result_t aql_gfx12_build_copy_reg_data(aql_cmd_buffer_t* buf,
                                                 uint32_t src_addr, void* dst_addr,
                                                 uint32_t size, bool wait);
static aql_result_t aql_gfx12_build_indirect_buffer(aql_cmd_buffer_t* buf,
                                                   const void* cmd_addr, size_t cmd_size);
static aql_result_t aql_gfx12_build_wait_reg_mem(aql_cmd_buffer_t* buf, bool mem_space,
                                                uint64_t wait_addr, bool func_eq,
                                                uint32_t mask_val, uint32_t wait_val);
static aql_result_t aql_gfx12_build_wait_idle(aql_cmd_buffer_t* buf);
static aql_result_t aql_gfx12_build_nop(aql_cmd_buffer_t* buf, uint32_t num_dwords);
static aql_result_t aql_gfx12_build_cache_flush(aql_cmd_buffer_t* buf,
                                               size_t addr, size_t size);
static aql_result_t aql_gfx12_build_pred_exec(aql_cmd_buffer_t* buf,
                                             uint32_t xcc_select, uint32_t exec_count);

static uint32_t aql_gfx12_make_grbm_index_impl(uint32_t se, uint32_t sa,
                                              uint32_t wgp, uint32_t instance);
static uint32_t aql_gfx12_get_cache_policy_impl(void);
static uint64_t aql_gfx12_get_smn_address_impl(uint64_t base_addr, uint32_t aid_index);

static bool aql_gfx12_is_valid_counter_reg(aql_block_id_t block_id, uint32_t reg_addr);
static const aql_counter_block_regs_t* aql_gfx12_get_counter_block_regs(aql_block_id_t block_id);
static aql_result_t aql_gfx12_get_counter_register_addr(aql_block_id_t block_id,
                                                       uint32_t counter_id,
                                                       uint32_t reg_type,
                                                       uint32_t* reg_addr);
static aql_result_t aql_gfx12_validate_counter_request(const aql_counter_request_t* request);

/*
 * GFX12 Counter Block Register Definitions
 */
static const aql_counter_block_regs_t gfx12_counter_blocks[] = {
    /* AQL_BLOCK_CB - Color Buffer */
    {
        .base_addr = 0x28000,
        .counter_count = 4,
        .counter_stride = 4,
        .select_reg_offset = 0x000,
        .enable_reg_offset = 0x010,
        .result_reg_offset = 0x020,
        .block_attributes = AQL_BLOCK_ATTR_SE_SPECIFIC
    },
    /* AQL_BLOCK_CPC - Command Processor Compute */
    {
        .base_addr = 0x26000,
        .counter_count = 2,
        .counter_stride = 4,
        .select_reg_offset = 0x000,
        .enable_reg_offset = 0x008,
        .result_reg_offset = 0x010,
        .block_attributes = 0
    },
    /* AQL_BLOCK_SQ - Shader Quad (with accumulation support) */
    {
        .base_addr = 0x28300,
        .counter_count = 16,
        .counter_stride = 4,
        .select_reg_offset = 0x000,
        .enable_reg_offset = 0x040,
        .result_reg_offset = 0x080,
        .block_attributes = AQL_BLOCK_ATTR_SE_SPECIFIC | AQL_BLOCK_ATTR_ACCUMULATION
    },
    /* Add more counter blocks as needed */
};

#define GFX12_NUM_COUNTER_BLOCKS (sizeof(gfx12_counter_blocks) / sizeof(gfx12_counter_blocks[0]))

/* Supported block IDs for GFX12 */
static const aql_block_id_t gfx12_supported_blocks[] = AQL_GFX12_SUPPORTED_BLOCKS;

/*
 * GFX12 Architecture Operations Table
 */
const aql_arch_ops_t aql_gfx12_ops = {
    /* Architecture identification */
    .arch_name = "gfx12",
    .arch_type = AQL_ARCH_GFX12,
    .gfx_version = 12,

    /* Architecture capabilities */
    .has_pred_exec = AQL_GFX12_HAS_PRED_EXEC,
    .has_uconfig_space = AQL_GFX12_HAS_UCONFIG_SPACE,
    .has_dual_sdma = AQL_GFX12_HAS_DUAL_SDMA,
    .has_gl_cache_hierarchy = AQL_GFX12_HAS_GL_CACHE_HIERARCHY,
    .has_smn_addressing = AQL_GFX12_HAS_SMN_ADDRESSING,

    /* Register space definitions */
    .register_spaces = {
        .config_space_start = AQL_GFX12_CONFIG_SPACE_START,
        .config_space_end = AQL_GFX12_CONFIG_SPACE_END,
        .uconfig_space_start = AQL_GFX12_UCONFIG_SPACE_START,
        .uconfig_space_end = AQL_GFX12_UCONFIG_SPACE_END,
        .has_uconfig_space = AQL_GFX12_HAS_UCONFIG_SPACE
    },

    /* Counter block information */
    .counter_blocks = gfx12_counter_blocks,
    .counter_block_count = GFX12_NUM_COUNTER_BLOCKS,

    /* PM4 command generation functions */
    .build_write_uconfig_reg = aql_gfx12_build_write_uconfig_reg,
    .build_write_config_reg = aql_gfx12_build_write_config_reg,
    .build_write_sh_reg = aql_gfx12_build_write_sh_reg,
    .build_copy_reg_data = aql_gfx12_build_copy_reg_data,
    .build_indirect_buffer = aql_gfx12_build_indirect_buffer,
    .build_wait_reg_mem = aql_gfx12_build_wait_reg_mem,
    .build_wait_idle = aql_gfx12_build_wait_idle,
    .build_nop = aql_gfx12_build_nop,
    .build_cache_flush = aql_gfx12_build_cache_flush,
    .build_pred_exec = aql_gfx12_build_pred_exec,

    /* Architecture-specific helper functions */
    .make_grbm_index = aql_gfx12_make_grbm_index_impl,
    .get_cache_policy = aql_gfx12_get_cache_policy_impl,
    .get_smn_address = aql_gfx12_get_smn_address_impl,

    /* Register and counter management */
    .is_valid_counter_reg = aql_gfx12_is_valid_counter_reg,
    .get_counter_block_regs = aql_gfx12_get_counter_block_regs,
    .get_counter_register_addr = aql_gfx12_get_counter_register_addr,
    .validate_counter_request = aql_gfx12_validate_counter_request
};

/*
 * PM4 Command Generation Functions
 */

static aql_result_t aql_gfx12_build_write_uconfig_reg(aql_cmd_buffer_t* buf,
                                                     uint32_t addr, uint32_t value) {
    uint32_t cmd[AQL_GFX12_SET_REG_SIZE_DWORDS];

    if (!buf) return AQL_ERROR_INVALID_ARGUMENT;

    AQL_CHECK_BUFFER_SPACE(buf, AQL_GFX12_SET_REG_SIZE_DWORDS);

    /* Build PACKET3_SET_UCONFIG_REG packet */
    cmd[0] = AQL_GFX12_PACKET3(AQL_GFX12_PACKET3_SET_UCONFIG_REG,
                              AQL_GFX12_SET_REG_SIZE_DWORDS * sizeof(uint32_t));
    cmd[1] = AQL_GFX12_SET_REG__REG_OFFSET(addr - AQL_GFX12_UCONFIG_SPACE_START);
    cmd[2] = value;

    AQL_APPEND_COMMAND(buf, cmd, AQL_GFX12_SET_REG_SIZE_DWORDS, "SET_UCONFIG_REG");
    return AQL_SUCCESS;
}

static aql_result_t aql_gfx12_build_write_config_reg(aql_cmd_buffer_t* buf,
                                                    uint32_t addr, uint32_t value) {
    uint32_t cmd[AQL_GFX12_SET_REG_SIZE_DWORDS];

    if (!buf) return AQL_ERROR_INVALID_ARGUMENT;

    AQL_CHECK_BUFFER_SPACE(buf, AQL_GFX12_SET_REG_SIZE_DWORDS);

    /* Build PACKET3_SET_CONFIG_REG packet */
    cmd[0] = AQL_GFX12_PACKET3(AQL_GFX12_PACKET3_SET_CONFIG_REG,
                              AQL_GFX12_SET_REG_SIZE_DWORDS * sizeof(uint32_t));
    cmd[1] = AQL_GFX12_SET_REG__REG_OFFSET(addr - AQL_GFX12_CONFIG_SPACE_START);
    cmd[2] = value;

    AQL_APPEND_COMMAND(buf, cmd, AQL_GFX12_SET_REG_SIZE_DWORDS, "SET_CONFIG_REG");
    return AQL_SUCCESS;
}

static aql_result_t aql_gfx12_build_write_sh_reg(aql_cmd_buffer_t* buf,
                                                uint32_t addr, uint32_t value) {
    uint32_t cmd[AQL_GFX12_SET_REG_SIZE_DWORDS];

    if (!buf) return AQL_ERROR_INVALID_ARGUMENT;

    AQL_CHECK_BUFFER_SPACE(buf, AQL_GFX12_SET_REG_SIZE_DWORDS);

    /* Build PACKET3_SET_SH_REG packet */
    cmd[0] = AQL_GFX12_PACKET3(AQL_GFX12_PACKET3_SET_SH_REG,
                              AQL_GFX12_SET_REG_SIZE_DWORDS * sizeof(uint32_t));
    cmd[1] = AQL_GFX12_SET_REG__REG_OFFSET(addr);  /* SH regs use direct offset */
    cmd[2] = value;

    AQL_APPEND_COMMAND(buf, cmd, AQL_GFX12_SET_REG_SIZE_DWORDS, "SET_SH_REG");
    return AQL_SUCCESS;
}

static aql_result_t aql_gfx12_build_copy_reg_data(aql_cmd_buffer_t* buf,
                                                 uint32_t src_addr, void* dst_addr,
                                                 uint32_t size, bool wait) {
    uint32_t cmd[AQL_GFX12_WRITE_DATA_MIN_SIZE_DWORDS + 1]; /* +1 for data dword */
    uint32_t control;

    if (!buf || !dst_addr || size == 0) return AQL_ERROR_INVALID_ARGUMENT;

    AQL_CHECK_BUFFER_SPACE(buf, AQL_GFX12_WRITE_DATA_MIN_SIZE_DWORDS + 1);

    /* Build control field */
    control = AQL_GFX12_WRITE_DATA__DST_SEL(AQL_GFX12_WRITE_DATA_DST_MMREG) |
              AQL_GFX12_WRITE_DATA__WR_CONFIRM(wait ? 1 : 0);

    /* Build PACKET3_WRITE_DATA packet */
    cmd[0] = AQL_GFX12_PACKET3(AQL_GFX12_PACKET3_WRITE_DATA,
                              (AQL_GFX12_WRITE_DATA_MIN_SIZE_DWORDS + 1) * sizeof(uint32_t));
    cmd[1] = control;
    cmd[2] = src_addr;                           /* Source register address */
    cmd[3] = (uint32_t)((uintptr_t)dst_addr);    /* Destination address low */
    cmd[4] = (uint32_t)((uintptr_t)dst_addr >> 32); /* Destination address high */

    AQL_APPEND_COMMAND(buf, cmd, AQL_GFX12_WRITE_DATA_MIN_SIZE_DWORDS + 1, "COPY_REG_DATA");
    return AQL_SUCCESS;
}

static aql_result_t aql_gfx12_build_indirect_buffer(aql_cmd_buffer_t* buf,
                                                   const void* cmd_addr, size_t cmd_size) {
    uint32_t cmd[AQL_GFX12_INDIRECT_BUFFER_SIZE_DWORDS];
    uintptr_t addr = (uintptr_t)cmd_addr;

    if (!buf || !cmd_addr || cmd_size == 0) return AQL_ERROR_INVALID_ARGUMENT;

    AQL_CHECK_BUFFER_SPACE(buf, AQL_GFX12_INDIRECT_BUFFER_SIZE_DWORDS);

    /* Build PACKET3_INDIRECT_BUFFER packet */
    cmd[0] = AQL_GFX12_PACKET3(AQL_GFX12_PACKET3_INDIRECT_BUFFER,
                              AQL_GFX12_INDIRECT_BUFFER_SIZE_DWORDS * sizeof(uint32_t));
    cmd[1] = AQL_GFX12_IB__IB_BASE_LO((uint32_t)addr);
    cmd[2] = AQL_GFX12_IB__IB_BASE_HI((uint32_t)(addr >> 32));
    cmd[3] = AQL_GFX12_IB__IB_SIZE(cmd_size / sizeof(uint32_t));

    AQL_APPEND_COMMAND(buf, cmd, AQL_GFX12_INDIRECT_BUFFER_SIZE_DWORDS, "INDIRECT_BUFFER");
    return AQL_SUCCESS;
}

static aql_result_t aql_gfx12_build_wait_reg_mem(aql_cmd_buffer_t* buf, bool mem_space,
                                                uint64_t wait_addr, bool func_eq,
                                                uint32_t mask_val, uint32_t wait_val) {
    uint32_t cmd[AQL_GFX12_WAIT_REG_MEM_SIZE_DWORDS];
    uint32_t operation, mem_space_field, function;

    if (!buf) return AQL_ERROR_INVALID_ARGUMENT;

    AQL_CHECK_BUFFER_SPACE(buf, AQL_GFX12_WAIT_REG_MEM_SIZE_DWORDS);

    /* Build control fields */
    operation = AQL_GFX12_WAIT_REG_MEM__OPERATION(AQL_GFX12_WAIT_REG_MEM_OP_WAIT_REG_MEM);
    mem_space_field = AQL_GFX12_WAIT_REG_MEM__MEM_SPACE(
        mem_space ? AQL_GFX12_WAIT_REG_MEM_SPACE_MEMORY : AQL_GFX12_WAIT_REG_MEM_SPACE_REGISTER);
    function = AQL_GFX12_WAIT_REG_MEM__FUNCTION(
        func_eq ? AQL_GFX12_WAIT_REG_MEM_FUNC_EQUAL : AQL_GFX12_WAIT_REG_MEM_FUNC_NOT_EQUAL);

    /* Build PACKET3_WAIT_REG_MEM packet */
    cmd[0] = AQL_GFX12_PACKET3(AQL_GFX12_PACKET3_WAIT_REG_MEM,
                              AQL_GFX12_WAIT_REG_MEM_SIZE_DWORDS * sizeof(uint32_t));
    cmd[1] = operation | mem_space_field | function;
    cmd[2] = (uint32_t)wait_addr;                /* Address low */
    cmd[3] = (uint32_t)(wait_addr >> 32);        /* Address high */
    cmd[4] = wait_val;                           /* Reference value */
    cmd[5] = mask_val;                           /* Mask value */
    cmd[6] = AQL_GFX12_WAIT_REG_MEM__POLL_INTERVAL(0x04); /* Poll interval */

    AQL_APPEND_COMMAND(buf, cmd, AQL_GFX12_WAIT_REG_MEM_SIZE_DWORDS, "WAIT_REG_MEM");
    return AQL_SUCCESS;
}

static aql_result_t aql_gfx12_build_wait_idle(aql_cmd_buffer_t* buf) {
    uint32_t cmd[AQL_GFX12_EVENT_WRITE_SIZE_DWORDS];

    if (!buf) return AQL_ERROR_INVALID_ARGUMENT;

    AQL_CHECK_BUFFER_SPACE(buf, AQL_GFX12_EVENT_WRITE_SIZE_DWORDS);

    /* Build PACKET3_EVENT_WRITE packet for CS_PARTIAL_FLUSH */
    cmd[0] = AQL_GFX12_PACKET3(AQL_GFX12_PACKET3_EVENT_WRITE,
                              AQL_GFX12_EVENT_WRITE_SIZE_DWORDS * sizeof(uint32_t));
    cmd[1] = AQL_GFX12_EVENT_WRITE__EVENT_TYPE(AQL_GFX12_CS_PARTIAL_FLUSH) |
             AQL_GFX12_EVENT_WRITE__EVENT_INDEX(AQL_GFX12_EVENT_INDEX_CS_PARTIAL_FLUSH);

    AQL_APPEND_COMMAND(buf, cmd, AQL_GFX12_EVENT_WRITE_SIZE_DWORDS, "WAIT_IDLE");
    return AQL_SUCCESS;
}

static aql_result_t aql_gfx12_build_nop(aql_cmd_buffer_t* buf, uint32_t num_dwords) {
    uint32_t cmd[1];

    if (!buf || num_dwords == 0) return AQL_ERROR_INVALID_ARGUMENT;

    AQL_CHECK_BUFFER_SPACE(buf, num_dwords);

    /* Build PACKET3_NOP packet */
    cmd[0] = AQL_GFX12_PACKET3(AQL_GFX12_PACKET3_NOP, num_dwords * sizeof(uint32_t));

    AQL_APPEND_COMMAND(buf, cmd, 1, "NOP_HEADER");

    /* Pad remaining dwords with zeros */
    for (uint32_t i = 1; i < num_dwords; i++) {
        aql_cmd_buffer_append_dword(buf, 0);
    }

    return AQL_SUCCESS;
}

static aql_result_t aql_gfx12_build_cache_flush(aql_cmd_buffer_t* buf,
                                               size_t addr, size_t size) {
    uint32_t cmd[AQL_GFX12_ACQUIRE_MEM_SIZE_DWORDS];
    uint32_t gcr_cntl;

    if (!buf) return AQL_ERROR_INVALID_ARGUMENT;

    AQL_CHECK_BUFFER_SPACE(buf, AQL_GFX12_ACQUIRE_MEM_SIZE_DWORDS);

    /* Calculate size in 256-byte chunks */
    size = ((addr % 256 + size) >> 8) + ((size + 0xFF) >> 8) - (size >> 8);

    /* Build GCR control field for RDNA3 */
    gcr_cntl = (AQL_GFX12_GCR_CNTL_SEQ_FORWARD & AQL_GFX12_GCR_CNTL_SEQ_MASK) |
               AQL_GFX12_GCR_CNTL_GL2_WB_MASK;

    /* Build PACKET3_ACQUIRE_MEM packet (8 dwords for GFX12) */
    cmd[0] = AQL_GFX12_PACKET3(AQL_GFX12_PACKET3_ACQUIRE_MEM,
                              AQL_GFX12_ACQUIRE_MEM_SIZE_DWORDS * sizeof(uint32_t));
    cmd[1] = 0;  /* Reserved */
    cmd[2] = AQL_GFX12_ACQUIRE_MEM__COHER_SIZE((uint32_t)size);
    cmd[3] = AQL_GFX12_ACQUIRE_MEM__COHER_SIZE_HI((uint32_t)(size >> 32));
    cmd[4] = AQL_GFX12_ACQUIRE_MEM__COHER_BASE_LO((uint32_t)(addr >> 8));
    cmd[5] = AQL_GFX12_ACQUIRE_MEM__COHER_BASE_HI((uint8_t)(addr >> 40));
    cmd[6] = AQL_GFX12_ACQUIRE_MEM__POLL_INTERVAL(0x10);
    cmd[7] = AQL_GFX12_ACQUIRE_MEM__GCR_CNTL(gcr_cntl);

    AQL_APPEND_COMMAND(buf, cmd, AQL_GFX12_ACQUIRE_MEM_SIZE_DWORDS, "CACHE_FLUSH");
    return AQL_SUCCESS;
}

static aql_result_t aql_gfx12_build_pred_exec(aql_cmd_buffer_t* buf,
                                             uint32_t xcc_select, uint32_t exec_count) {
    /* PRED_EXEC is not supported on GFX12/RDNA3 */
    (void)buf;
    (void)xcc_select;
    (void)exec_count;
    return AQL_ERROR_UNSUPPORTED_ARCH;
}

/*
 * Architecture-Specific Helper Functions
 */

static uint32_t aql_gfx12_make_grbm_index_impl(uint32_t se, uint32_t sa,
                                              uint32_t wgp, uint32_t instance) {
    return aql_gfx12_make_grbm_index(se, sa, wgp, instance);
}

static uint32_t aql_gfx12_get_cache_policy_impl(void) {
    return AQL_GFX12_CACHE_POLICY;
}

static uint64_t aql_gfx12_get_smn_address_impl(uint64_t base_addr, uint32_t aid_index) {
    /* SMN addressing not supported on GFX12 */
    (void)base_addr;
    (void)aid_index;
    return 0;
}

/*
 * Register and Counter Management Functions
 */

static bool aql_gfx12_is_valid_counter_reg(aql_block_id_t block_id, uint32_t reg_addr) {
    const aql_counter_block_regs_t* block_regs;

    block_regs = aql_gfx12_get_counter_block_regs(block_id);
    if (!block_regs) return false;

    /* Check if register is within the block's address range */
    return (reg_addr >= block_regs->base_addr) &&
           (reg_addr < (block_regs->base_addr + (block_regs->counter_count * block_regs->counter_stride)));
}

static const aql_counter_block_regs_t* aql_gfx12_get_counter_block_regs(aql_block_id_t block_id) {
    /* Simple linear search - could be optimized with hash table for large numbers of blocks */
    for (uint32_t i = 0; i < GFX12_NUM_COUNTER_BLOCKS; i++) {
        if (gfx12_supported_blocks[i] == block_id) {
            return &gfx12_counter_blocks[i];
        }
    }
    return NULL;
}

static aql_result_t aql_gfx12_get_counter_register_addr(aql_block_id_t block_id,
                                                       uint32_t counter_id,
                                                       uint32_t reg_type,
                                                       uint32_t* reg_addr) {
    const aql_counter_block_regs_t* block_regs;
    uint32_t offset;

    if (!reg_addr) return AQL_ERROR_INVALID_ARGUMENT;

    block_regs = aql_gfx12_get_counter_block_regs(block_id);
    if (!block_regs) return AQL_ERROR_NOT_FOUND;

    if (counter_id >= block_regs->counter_count) return AQL_ERROR_INVALID_COUNTER;

    /* Calculate register address based on type */
    switch (reg_type) {
        case AQL_REG_TYPE_SELECT:
            offset = block_regs->select_reg_offset;
            break;
        case AQL_REG_TYPE_ENABLE:
            offset = block_regs->enable_reg_offset;
            break;
        case AQL_REG_TYPE_RESULT:
            offset = block_regs->result_reg_offset;
            break;
        default:
            return AQL_ERROR_INVALID_ARGUMENT;
    }

    *reg_addr = block_regs->base_addr + offset + (counter_id * block_regs->counter_stride);
    return AQL_SUCCESS;
}

static aql_result_t aql_gfx12_validate_counter_request(const aql_counter_request_t* request) {
    const aql_counter_block_regs_t* block_regs;

    if (!request) return AQL_ERROR_INVALID_ARGUMENT;

    /* Check if block is supported on GFX12 */
    block_regs = aql_gfx12_get_counter_block_regs(request->block_id);
    if (!block_regs) return AQL_ERROR_UNSUPPORTED_ARCH;

    /* Check counter ID range */
    if (request->counter_id >= block_regs->counter_count) {
        return AQL_ERROR_INVALID_COUNTER;
    }

    /* Validate block instance if block is instance-specific */
    if (block_regs->block_attributes & AQL_BLOCK_ATTR_INSTANCE_SPECIFIC) {
        /* Add instance validation logic here */
    }

    return AQL_SUCCESS;
}