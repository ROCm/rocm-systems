/**
 * @file aql_types.h
 * @brief Core data types and structures for AQL packet generation
 *
 * This file defines the fundamental data structures used throughout the AQL
 * packet generation library. These structures replace the C++ classes from
 * the original AQLProfile implementation and are designed for kernel module
 * compatibility.
 *
 * Key design principles:
 * - All structures are plain C (no function pointers in basic types)
 * - Fixed-size allocations where possible for kernel compatibility
 * - Clear separation between data and operations
 * - Comprehensive error handling via return codes
 */

#ifndef AQL_TYPES_H
#define AQL_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/errno.h>
#include <linux/string.h>
#include <linux/slab.h>
#else
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#endif

/* Maximum supported counters per request */
#define AQL_MAX_COUNTERS_PER_REQUEST    128

/* Maximum command buffer size in dwords */
#define AQL_MAX_CMD_BUFFER_DWORDS      4096

/* Maximum architecture name length */
#define AQL_MAX_ARCH_NAME_LEN           32

/* AQL packet size in bytes (fixed) */
#define AQL_PACKET_SIZE                 64

/* PM4 indirect buffer command size in dwords */
#define AQL_PM4_IB_COMMAND_DWORDS       4

/**
 * @brief Error codes for AQL operations
 */
typedef enum {
    AQL_SUCCESS = 0,                    /**< Operation successful */
    AQL_ERROR_INVALID_ARGUMENT = -1,    /**< Invalid function argument */
    AQL_ERROR_BUFFER_TOO_SMALL = -2,    /**< Buffer too small for operation */
    AQL_ERROR_UNSUPPORTED_ARCH = -3,    /**< Unsupported GPU architecture */
    AQL_ERROR_INVALID_COUNTER = -4,     /**< Invalid counter specification */
    AQL_ERROR_COUNTER_LIMIT = -5,       /**< Too many counters requested */
    AQL_ERROR_HARDWARE_FAULT = -6,      /**< Hardware fault detected */
    AQL_ERROR_TIMEOUT = -7,             /**< Operation timeout */
    AQL_ERROR_NO_MEMORY = -8,           /**< Memory allocation failed */
    AQL_ERROR_INVALID_STATE = -9,       /**< Invalid operation state */
    AQL_ERROR_NOT_FOUND = -10,          /**< Resource not found */
    AQL_ERROR_INVALID_REGISTER = -11,   /**< Invalid register address */
    AQL_ERROR_INVALID_BLOCK = -12,      /**< Invalid counter block */
    AQL_ERROR_INVALID_INSTANCE = -13,   /**< Invalid block instance */
} aql_result_t;

/**
 * @brief GPU architecture enumeration
 */
typedef enum {
    AQL_ARCH_UNKNOWN = 0,
    AQL_ARCH_GFX9,                      /**< GFX9 (Vega) */
    AQL_ARCH_GFX10,                     /**< GFX10 (RDNA1) */
    AQL_ARCH_GFX11,                     /**< GFX11 (RDNA2) */
    AQL_ARCH_GFX12,                     /**< GFX12 (RDNA3) */
    AQL_ARCH_COUNT
} aql_arch_type_t;

/**
 * @brief Counter block enumeration (unified across architectures)
 */
typedef enum {
    AQL_BLOCK_CB = 0,                   /**< Color Buffer */
    AQL_BLOCK_CPC,                      /**< Command Processor Compute */
    AQL_BLOCK_CPF,                      /**< Command Processor Fetch */
    AQL_BLOCK_CPG,                      /**< Command Processor Graphics */
    AQL_BLOCK_DB,                       /**< Depth Buffer */
    AQL_BLOCK_GDS,                      /**< Global Data Share */
    AQL_BLOCK_GRBM,                     /**< Graphics Register Bus Manager */
    AQL_BLOCK_GRBM_SE,                  /**< GRBM Shader Engine */
    AQL_BLOCK_IA,                       /**< Input Assembler (GFX9 only) */
    AQL_BLOCK_PA_SC,                    /**< PA Setup/Clipping (GFX9 only) */
    AQL_BLOCK_PA_SU,                    /**< PA Setup Unit (GFX9 only) */
    AQL_BLOCK_SPI,                      /**< Shader Processor Input */
    AQL_BLOCK_SQ,                       /**< Shader Quad */
    AQL_BLOCK_SQ_GS,                    /**< SQ Geometry Shader */
    AQL_BLOCK_SQ_VS,                    /**< SQ Vertex Shader (GFX9 only) */
    AQL_BLOCK_SQ_PS,                    /**< SQ Pixel Shader */
    AQL_BLOCK_SQ_HS,                    /**< SQ Hull Shader */
    AQL_BLOCK_SQ_CS,                    /**< SQ Compute Shader */
    AQL_BLOCK_SX,                       /**< Shader Export */
    AQL_BLOCK_TA,                       /**< Texture Addresser */
    AQL_BLOCK_TCA,                      /**< Texture Cache Arbiter (GFX9 only) */
    AQL_BLOCK_TCC,                      /**< Texture Cache Controller (GFX9 only) */
    AQL_BLOCK_TCP,                      /**< Texture Cache Per-pipe */
    AQL_BLOCK_TCS,                      /**< Texture Cache System (GFX9 only) */
    AQL_BLOCK_TD,                       /**< Texture Data */
    AQL_BLOCK_VGT,                      /**< Vertex Grouper/Tessellator (GFX9 only) */
    AQL_BLOCK_WD,                       /**< Workgroup Distributor (GFX9 only) */

    /* Memory Controller blocks */
    AQL_BLOCK_GCEA,                     /**< Graphics Command Engine Arbiter */
    AQL_BLOCK_ATC,                      /**< Address Translation Cache (GFX9 only) */
    AQL_BLOCK_ATC_L2,                   /**< ATC L2 Cache (GFX9 only) */
    AQL_BLOCK_MC_VM_L2,                 /**< MC VM L2 (GFX9 only) */
    AQL_BLOCK_RPB,                      /**< Render Backend */
    AQL_BLOCK_RMI,                      /**< Read/Write Memory Interface */

    /* RDNA specific blocks */
    AQL_BLOCK_GL1A,                     /**< GL1 Texture Cache Arbiter (RDNA+) */
    AQL_BLOCK_GL1C,                     /**< GL1 Texture Cache Controller (RDNA+) */
    AQL_BLOCK_GL2A,                     /**< GL2 Cache Arbiter (RDNA+) */
    AQL_BLOCK_GL2C,                     /**< GL2 Cache Controller (RDNA+) */
    AQL_BLOCK_GCR,                      /**< Graphics Command Ring (RDNA+) */
    AQL_BLOCK_GUS,                      /**< Graphics Unified Scheduler (RDNA+) */

    /* System blocks */
    AQL_BLOCK_SDMA,                     /**< System DMA (single engine) */
    AQL_BLOCK_SDMA0,                    /**< System DMA Engine 0 (RDNA+) */
    AQL_BLOCK_SDMA1,                    /**< System DMA Engine 1 (RDNA+) */
    AQL_BLOCK_UMC,                      /**< Unified Memory Controller */
    AQL_BLOCK_IOMMU_V2,                 /**< IOMMU Version 2 */

    AQL_BLOCK_COUNT,                    /**< Total number of block types */
    AQL_BLOCK_UNKNOWN = 0xFFFFFFFF      /**< Unknown/invalid block type */
} aql_block_id_t;

/**
 * @brief Command buffer for PM4 command generation
 *
 * This structure replaces the C++ CmdBuffer class and provides a simple
 * buffer for accumulating PM4 commands. The buffer uses pre-allocated
 * storage to avoid dynamic allocation in kernel context.
 */
typedef struct {
    uint32_t* data;                     /**< Command data buffer */
    size_t capacity;                    /**< Total capacity in dwords */
    size_t used;                        /**< Used size in dwords */
    bool is_external;                   /**< True if data buffer is externally owned */
} aql_cmd_buffer_t;

/**
 * @brief Register address specification
 */
typedef struct {
    uint32_t offset;                    /**< Register offset from base */
    uint32_t base_index;                /**< IP block base index */
} aql_register_addr_t;

/**
 * @brief Counter register layout for a hardware block
 */
typedef struct {
    uint32_t base_addr;                 /**< Base register address */
    uint32_t counter_count;             /**< Number of counters in block */
    uint32_t counter_stride;            /**< Bytes between counter registers */
    uint32_t select_reg_offset;         /**< Event select register offset */
    uint32_t enable_reg_offset;         /**< Counter enable register offset */
    uint32_t result_reg_offset;         /**< Counter result register offset */
    uint32_t block_attributes;          /**< Block capability flags */
} aql_counter_block_regs_t;

/**
 * @brief Block attribute flags
 */
#define AQL_BLOCK_ATTR_SE_SPECIFIC      (1U << 0)   /**< Shader Engine specific */
#define AQL_BLOCK_ATTR_AID_SPECIFIC     (1U << 1)   /**< AID specific */
#define AQL_BLOCK_ATTR_INSTANCE_SPECIFIC (1U << 2)  /**< Instance specific */
#define AQL_BLOCK_ATTR_ACCUMULATION     (1U << 3)   /**< Supports accumulation mode */
#define AQL_BLOCK_ATTR_SMN_ADDRESSING   (1U << 4)   /**< Uses SMN addressing */

/**
 * @brief Register space definition
 */
typedef struct {
    uint32_t config_space_start;        /**< CONFIG space start address */
    uint32_t config_space_end;          /**< CONFIG space end address */
    uint32_t uconfig_space_start;       /**< UCONFIG space start (GFX9 only) */
    uint32_t uconfig_space_end;         /**< UCONFIG space end (GFX9 only) */
    bool has_uconfig_space;             /**< True if UCONFIG space exists */
} aql_register_spaces_t;

/**
 * @brief Counter request specification
 */
typedef struct {
    aql_block_id_t block_id;            /**< Counter block identifier */
    uint32_t block_instance;            /**< Block instance number */
    uint32_t counter_id;                /**< Counter within block */
    uint32_t event_select;              /**< Event to count */
    uint32_t flags;                     /**< Counter-specific flags */

    /* Runtime assignment (filled by allocation logic) */
    uint32_t assigned_register;         /**< Assigned hardware register */
    void* result_location;              /**< Where to store counter result */

    /* Debug information */
    const char* block_name;             /**< Human-readable block name */
    const char* counter_name;           /**< Human-readable counter name */
} aql_counter_request_t;

/**
 * @brief Counter flags for special modes
 */
#define AQL_COUNTER_FLAG_ACCUMULATION_NONE      0
#define AQL_COUNTER_FLAG_ACCUMULATION_LO_RES    (1U << 0)
#define AQL_COUNTER_FLAG_ACCUMULATION_HI_RES    (1U << 1)

/**
 * @brief GPU agent information
 */
typedef struct {
    char gfx_name[AQL_MAX_ARCH_NAME_LEN]; /**< GPU architecture name */
    aql_arch_type_t arch_type;          /**< Detected architecture type */
    uint32_t xcc_count;                 /**< Number of XCCs */
    uint32_t se_count;                  /**< Number of Shader Engines */
    uint32_t cu_count;                  /**< Number of Compute Units */
    uint32_t shader_arrays_per_se;      /**< Shader arrays per SE */
    uint32_t domain;                    /**< PCI domain */
    uint32_t location_id;               /**< PCI BDF */
} aql_agent_info_t;

/**
 * @brief AQL PM4 indirect buffer packet structure
 *
 * This structure represents the AQL packet format used to submit PM4
 * commands to the GPU. It's a direct translation of the C++ structure
 * with explicit field documentation.
 */
typedef struct {
    uint16_t header;                    /**< AQL packet header */
    uint16_t pm4_ib_format;             /**< PM4 IB format (always 1) */
    uint32_t pm4_ib_command[AQL_PM4_IB_COMMAND_DWORDS]; /**< PM4 IB command */
    uint32_t dw_count_remain;           /**< Remaining dword count (always 10) */
    uint32_t reserved[8];               /**< Reserved fields (must be zero) */
    uint64_t completion_signal;         /**< Completion signal handle */
} aql_pm4_ib_packet_t;

/* Ensure packet is exactly 64 bytes */
_Static_assert(sizeof(aql_pm4_ib_packet_t) == AQL_PACKET_SIZE,
               "AQL packet must be exactly 64 bytes");

/**
 * @brief AQL context for packet generation
 *
 * This structure maintains the state for AQL packet generation operations.
 * It replaces the various C++ classes and provides a unified context for
 * all packet generation activities.
 */
typedef struct {
    /* Agent information */
    aql_agent_info_t agent_info;        /**< GPU agent details */

    /* Architecture-specific operations (forward declaration) */
    const struct aql_arch_ops* arch_ops; /**< Architecture operations table */

    /* Counter configuration */
    aql_counter_request_t* counters;     /**< Array of counter requests */
    uint32_t counter_count;             /**< Number of active counters */
    uint32_t max_counters;              /**< Maximum counters supported */

    /* Command buffer management */
    aql_cmd_buffer_t cmd_buffer;        /**< Primary command buffer */
    aql_cmd_buffer_t ib_buffer;         /**< Indirect buffer commands */

    /* Runtime state */
    uint32_t current_xcc;              /**< Current XCC being programmed */
    uint32_t current_grbm_index;       /**< Current GRBM index setting */

    /* Error tracking */
    aql_result_t last_error;           /**< Last operation error */
    char error_msg[256];               /**< Detailed error message */

    /* Debug flags */
    uint32_t debug_flags;              /**< Debug output control */
} aql_context_t;

/**
 * @brief Debug flag definitions
 */
#define AQL_DEBUG_COMMANDS             (1U << 0)    /**< Debug command generation */
#define AQL_DEBUG_PACKETS              (1U << 1)    /**< Debug packet contents */
#define AQL_DEBUG_REGISTERS            (1U << 2)    /**< Debug register programming */
#define AQL_DEBUG_COUNTERS             (1U << 3)    /**< Debug counter assignment */

/**
 * @brief Memory allocation callback for AQL operations
 *
 * This callback is used to request memory allocation for buffers that need
 * to be accessible by both CPU and GPU. In kernel context, this would
 * typically allocate DMA-coherent memory.
 */
typedef aql_result_t (*aql_memory_alloc_cb_t)(void** ptr, size_t size,
                                              uint32_t flags, void* userdata);

/**
 * @brief Memory deallocation callback
 */
typedef void (*aql_memory_dealloc_cb_t)(void* ptr, void* userdata);

/**
 * @brief Memory allocation flags
 */
#define AQL_MEM_FLAG_HOST_ACCESS       (1U << 0)    /**< CPU accessible */
#define AQL_MEM_FLAG_DEVICE_ACCESS     (1U << 1)    /**< GPU accessible */
#define AQL_MEM_FLAG_COHERENT          (1U << 2)    /**< Cache coherent */
#define AQL_MEM_FLAG_UNCACHED          (1U << 3)    /**< Uncached access */

/* Forward declarations for function pointer tables */
struct aql_arch_ops;

#endif /* AQL_TYPES_H */