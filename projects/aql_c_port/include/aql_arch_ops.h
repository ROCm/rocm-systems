/**
 * @file aql_arch_ops.h
 * @brief Architecture-specific operations interface
 *
 * This file defines the architecture operations interface that provides
 * abstraction for different GPU architectures (GFX9, GFX10, GFX11, GFX12).
 * It replaces the virtual function interface from the C++ implementation
 * with function pointer tables.
 *
 * The design follows these principles:
 * - One operations table per supported architecture
 * - All functions are stateless and thread-safe
 * - Consistent error handling across all operations
 * - Support for architecture-specific features and limitations
 */

#ifndef AQL_ARCH_OPS_H
#define AQL_ARCH_OPS_H

#include "aql_types.h"

/**
 * @brief Architecture-specific operations table
 *
 * This structure contains function pointers for all architecture-specific
 * operations. Each supported GPU architecture provides its own implementation
 * of these functions.
 */
typedef struct aql_arch_ops {
    /* Architecture identification */
    const char* arch_name;              /**< Architecture name (e.g., "gfx942") */
    aql_arch_type_t arch_type;          /**< Architecture enumeration */
    uint32_t gfx_version;               /**< Numeric GFX version */

    /* Architecture capabilities */
    bool has_pred_exec;                 /**< Supports PRED_EXEC packets */
    bool has_uconfig_space;             /**< Has UCONFIG register space */
    bool has_dual_sdma;                 /**< Has dual SDMA engines */
    bool has_gl_cache_hierarchy;        /**< Has GL1/GL2 cache blocks */
    bool has_smn_addressing;            /**< Supports SMN addressing */

    /* Register space definitions */
    aql_register_spaces_t register_spaces; /**< Register space layout */

    /* Counter block information */
    const aql_counter_block_regs_t* counter_blocks; /**< Counter block registers */
    uint32_t counter_block_count;       /**< Number of counter blocks */

    /*
     * PM4 Command Generation Functions
     *
     * These functions generate PM4 commands specific to the architecture.
     * All functions follow the pattern:
     * - Return aql_result_t (AQL_SUCCESS on success)
     * - First parameter is command buffer to append to
     * - Remaining parameters are command-specific
     */

    /**
     * @brief Build WRITE_UCONFIG_REG packet
     * @param buf Command buffer to append to
     * @param addr Register address
     * @param value Value to write
     * @return AQL_SUCCESS or error code
     */
    aql_result_t (*build_write_uconfig_reg)(aql_cmd_buffer_t* buf,
                                           uint32_t addr, uint32_t value);

    /**
     * @brief Build WRITE_CONFIG_REG packet
     * @param buf Command buffer to append to
     * @param addr Register address
     * @param value Value to write
     * @return AQL_SUCCESS or error code
     */
    aql_result_t (*build_write_config_reg)(aql_cmd_buffer_t* buf,
                                          uint32_t addr, uint32_t value);

    /**
     * @brief Build WRITE_SH_REG packet
     * @param buf Command buffer to append to
     * @param addr Register address
     * @param value Value to write
     * @return AQL_SUCCESS or error code
     */
    aql_result_t (*build_write_sh_reg)(aql_cmd_buffer_t* buf,
                                      uint32_t addr, uint32_t value);

    /**
     * @brief Build COPY_REG_DATA packet
     * @param buf Command buffer to append to
     * @param src_addr Source register address
     * @param dst_addr Destination memory address
     * @param size Size of data to copy in bytes
     * @param wait Wait for completion if true
     * @return AQL_SUCCESS or error code
     */
    aql_result_t (*build_copy_reg_data)(aql_cmd_buffer_t* buf,
                                       uint32_t src_addr, void* dst_addr,
                                       uint32_t size, bool wait);

    /**
     * @brief Build INDIRECT_BUFFER packet
     * @param buf Command buffer to append to
     * @param cmd_addr Address of command buffer to execute
     * @param cmd_size Size of command buffer in bytes
     * @return AQL_SUCCESS or error code
     */
    aql_result_t (*build_indirect_buffer)(aql_cmd_buffer_t* buf,
                                         const void* cmd_addr, size_t cmd_size);

    /**
     * @brief Build WAIT_REG_MEM packet
     * @param buf Command buffer to append to
     * @param mem_space True for memory space, false for register space
     * @param wait_addr Address to poll
     * @param func_eq True for equal comparison, false for not-equal
     * @param mask_val Mask to apply to read value
     * @param wait_val Value to compare against
     * @return AQL_SUCCESS or error code
     */
    aql_result_t (*build_wait_reg_mem)(aql_cmd_buffer_t* buf, bool mem_space,
                                      uint64_t wait_addr, bool func_eq,
                                      uint32_t mask_val, uint32_t wait_val);

    /**
     * @brief Build WAIT_IDLE/BARRIER packet
     * @param buf Command buffer to append to
     * @return AQL_SUCCESS or error code
     */
    aql_result_t (*build_wait_idle)(aql_cmd_buffer_t* buf);

    /**
     * @brief Build NOP packet for padding
     * @param buf Command buffer to append to
     * @param num_dwords Total packet size in dwords (including header)
     * @return AQL_SUCCESS or error code
     */
    aql_result_t (*build_nop)(aql_cmd_buffer_t* buf, uint32_t num_dwords);

    /**
     * @brief Build CACHE_FLUSH packet (architecture-specific)
     * @param buf Command buffer to append to
     * @param addr Base address to flush
     * @param size Size of memory region to flush
     * @return AQL_SUCCESS or error code
     */
    aql_result_t (*build_cache_flush)(aql_cmd_buffer_t* buf,
                                     size_t addr, size_t size);

    /**
     * @brief Build PRED_EXEC packet (GFX9 only)
     * @param buf Command buffer to append to
     * @param xcc_select XCC to target
     * @param exec_count Execution count
     * @return AQL_SUCCESS or error code, AQL_ERROR_UNSUPPORTED_ARCH if not supported
     */
    aql_result_t (*build_pred_exec)(aql_cmd_buffer_t* buf,
                                   uint32_t xcc_select, uint32_t exec_count);

    /*
     * Architecture-Specific Helper Functions
     */

    /**
     * @brief Create GRBM index value for register programming
     * @param se Shader Engine index
     * @param sa Shader Array index
     * @param wgp Workgroup Processor index (RDNA only)
     * @param instance Instance index
     * @return GRBM index value
     */
    uint32_t (*make_grbm_index)(uint32_t se, uint32_t sa,
                               uint32_t wgp, uint32_t instance);

    /**
     * @brief Get cache policy value for memory operations
     * @return Cache policy value for this architecture
     */
    uint32_t (*get_cache_policy)(void);

    /**
     * @brief Get SMN address for a given base address and AID
     * @param base_addr Base SMN address
     * @param aid_index AID index
     * @return Translated SMN address, or 0 if not supported
     */
    uint64_t (*get_smn_address)(uint64_t base_addr, uint32_t aid_index);

    /*
     * Register and Counter Management
     */

    /**
     * @brief Validate if a register address is valid for counter access
     * @param block_id Counter block identifier
     * @param reg_addr Register address to validate
     * @return True if register is valid and accessible
     */
    bool (*is_valid_counter_reg)(aql_block_id_t block_id, uint32_t reg_addr);

    /**
     * @brief Get counter block register information
     * @param block_id Counter block identifier
     * @return Pointer to register info, or NULL if not found
     */
    const aql_counter_block_regs_t* (*get_counter_block_regs)(aql_block_id_t block_id);

    /**
     * @brief Calculate register address for a specific counter
     * @param block_id Counter block identifier
     * @param counter_id Counter index within block
     * @param reg_type Register type (select, enable, result)
     * @param reg_addr Output: calculated register address
     * @return AQL_SUCCESS or error code
     */
    aql_result_t (*get_counter_register_addr)(aql_block_id_t block_id,
                                             uint32_t counter_id,
                                             uint32_t reg_type,
                                             uint32_t* reg_addr);

    /**
     * @brief Validate counter request for this architecture
     * @param request Counter request to validate
     * @return AQL_SUCCESS if valid, error code otherwise
     */
    aql_result_t (*validate_counter_request)(const aql_counter_request_t* request);

} aql_arch_ops_t;

/**
 * @brief Register type enumeration for get_counter_register_addr
 */
typedef enum {
    AQL_REG_TYPE_SELECT = 0,            /**< Event select register */
    AQL_REG_TYPE_ENABLE = 1,            /**< Counter enable register */
    AQL_REG_TYPE_RESULT = 2,            /**< Counter result register */
} aql_register_type_t;

/*
 * Architecture Operations Tables
 *
 * Each supported architecture provides a static operations table.
 * These are defined in separate source files (aql_gfx9_ops.c, etc.)
 */

extern const aql_arch_ops_t aql_gfx9_ops;
extern const aql_arch_ops_t aql_gfx10_ops;
extern const aql_arch_ops_t aql_gfx11_ops;
extern const aql_arch_ops_t aql_gfx12_ops;

/*
 * Architecture Detection and Selection
 */

/**
 * @brief Detect GPU architecture from name string
 * @param gfx_name GPU architecture name (e.g., "gfx942", "gfx1101")
 * @return Architecture operations table, or NULL if not supported
 */
const aql_arch_ops_t* aql_detect_architecture(const char* gfx_name);

/**
 * @brief Get architecture operations by type
 * @param arch_type Architecture type enumeration
 * @return Architecture operations table, or NULL if invalid
 */
const aql_arch_ops_t* aql_get_arch_ops(aql_arch_type_t arch_type);

/**
 * @brief List all supported architectures
 * @param ops_list Output array of operation tables
 * @param max_count Maximum number of entries in ops_list
 * @return Number of architectures written to ops_list
 */
uint32_t aql_list_supported_architectures(const aql_arch_ops_t** ops_list,
                                         uint32_t max_count);

/*
 * Utility Functions for Architecture Operations
 */

/**
 * @brief Check if architecture supports a specific feature
 * @param ops Architecture operations table
 * @param feature_flag Feature flag to check
 * @return True if feature is supported
 */
static inline bool aql_arch_has_feature(const aql_arch_ops_t* ops, uint32_t feature_flag) {
    if (!ops) return false;

    switch (feature_flag) {
        case 0x01: return ops->has_pred_exec;
        case 0x02: return ops->has_uconfig_space;
        case 0x04: return ops->has_dual_sdma;
        case 0x08: return ops->has_gl_cache_hierarchy;
        case 0x10: return ops->has_smn_addressing;
        default: return false;
    }
}

/**
 * @brief Get human-readable architecture name
 * @param ops Architecture operations table
 * @return Architecture name string, or "unknown" if ops is NULL
 */
static inline const char* aql_arch_get_name(const aql_arch_ops_t* ops) {
    return ops ? ops->arch_name : "unknown";
}

/**
 * @brief Check if register address is in privileged CONFIG space
 * @param ops Architecture operations table
 * @param addr Register address to check
 * @return True if address is in CONFIG space
 */
static inline bool aql_is_config_reg(const aql_arch_ops_t* ops, uint32_t addr) {
    if (!ops) return false;
    return (addr >= ops->register_spaces.config_space_start) &&
           (addr <= ops->register_spaces.config_space_end);
}

/**
 * @brief Check if register address is in UCONFIG space (GFX9 only)
 * @param ops Architecture operations table
 * @param addr Register address to check
 * @return True if address is in UCONFIG space
 */
static inline bool aql_is_uconfig_reg(const aql_arch_ops_t* ops, uint32_t addr) {
    if (!ops || !ops->has_uconfig_space) return false;
    return (addr >= ops->register_spaces.uconfig_space_start) &&
           (addr <= ops->register_spaces.uconfig_space_end);
}

/*
 * Additional utility structures and functions
 */

/**
 * @brief Architecture capabilities structure for querying
 */
typedef struct {
    bool has_pred_exec;                 /**< Supports PRED_EXEC packets */
    bool has_uconfig_space;             /**< Has UCONFIG register space */
    bool has_dual_sdma;                 /**< Has dual SDMA engines */
    bool has_gl_cache_hierarchy;        /**< Has GL1/GL2 cache blocks */
    bool has_smn_addressing;            /**< Supports SMN addressing */
} aql_arch_capabilities_t;

/**
 * @brief Get architecture capabilities
 * @param gfx_name GPU architecture name
 * @param capabilities Output structure for capabilities
 * @return AQL_SUCCESS if architecture is supported
 */
aql_result_t aql_get_arch_capabilities(const char* gfx_name,
                                      aql_arch_capabilities_t* capabilities);

/**
 * @brief Get register space information for architecture
 * @param gfx_name GPU architecture name
 * @param register_spaces Output structure for register spaces
 * @return AQL_SUCCESS if architecture is supported
 */
aql_result_t aql_get_register_spaces(const char* gfx_name,
                                    aql_register_spaces_t* register_spaces);

#endif /* AQL_ARCH_OPS_H */