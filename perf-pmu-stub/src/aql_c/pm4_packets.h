/**
 * @file pm4_packets.h
 * @brief PM4 packet builder interface for constructing GPU command buffers
 *
 * This provides a builder pattern API for constructing PM4 packets that matches
 * the Rust implementation's approach while maintaining exact binary compatibility.
 */

#ifndef PM4_PACKETS_H
#define PM4_PACKETS_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/slab.h>
#else
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#endif

/* PM4 Opcodes - must match hardware specification */
#define PM4_SET_UCONFIG_REG_OPCODE  0x79
#define PM4_EVENT_WRITE_OPCODE       0x46  /* decimal 70 */
#define PM4_COPY_DATA_OPCODE         0x40  /* decimal 64 */
#define PM4_ACQUIRE_MEM_OPCODE       0x58  /* Flush/Invalidate cache */
#define PM4_WRITE_SH_REG_OPCODE      0x76

/* Register space bases */
#define UCONFIG_SPACE_START          0x0000C000
#define PERSISTENT_SPACE_START       0x00002C00

/* Event types */
#define VGT_EVENT_TYPE_CS_PARTIAL_FLUSH  0x07

/* PM4 packet header construction
 *
 * IMPORTANT: This uses the same count calculation as aqlprofile implementation:
 * count = (packet_size_bytes / 4) - 2
 *
 * This is intentionally different from the AMD PM4 specification which would use:
 * count = total_dwords - 1 (number of DWORDs following the header)
 *
 * For a 3-DWORD packet (header + 2 data DWORDs):
 * - AMD spec would use: count = 3 - 1 = 2
 * - aqlprofile uses: count = 12/4 - 2 = 1
 * - We match aqlprofile for compatibility: count = 1
 */
#define PM4_TYPE_3_HEADER(opcode, packet_size_bytes) \
    (3u << 30 | ((opcode) & 0xFF) << 8 | (((packet_size_bytes / 4) - 2) & 0x3FFF) << 16)

/**
 * PM4 command buffer structure
 * Manages a growable buffer for accumulating PM4 packets
 */
typedef struct {
    uint32_t *data;       /* Buffer holding PM4 commands */
    size_t capacity;      /* Buffer capacity in DWORDs */
    size_t size;          /* Current size in DWORDs */
#ifdef __KERNEL__
    gfp_t gfp_flags;     /* Kernel memory allocation flags */
#endif
} pm4_buffer_t;

/**
 * Copy Data packet flags structure
 * Matches Rust CopyData bitfield exactly
 */
typedef union {
    uint32_t raw;
    struct {
        uint32_t src_sel : 4;          /* bits 0-3 */
        uint32_t reserved1 : 4;        /* bits 4-7 */
        uint32_t dst_sel : 4;          /* bits 8-11 */
        uint32_t reserved2 : 1;        /* bit 12 */
        uint32_t src_temporal : 2;     /* bits 13-14 */
        uint32_t reserved3 : 1;        /* bit 15 */
        uint32_t count_sel : 1;        /* bit 16 */
        uint32_t reserved4 : 3;        /* bits 17-19 */
        uint32_t wr_confirm : 1;       /* bit 20 */
        uint32_t mode : 1;             /* bit 21 */
        uint32_t reserved5 : 1;        /* bit 22 */
        uint32_t aid_id : 2;           /* bits 23-24 */
        uint32_t dst_temporal : 2;     /* bits 25-26 */
        uint32_t reserved6 : 2;        /* bits 27-28 */
        uint32_t pq_exe_status : 1;    /* bit 29 */
        uint32_t reserved7 : 2;        /* bits 30-31 */
    } bits;
} pm4_copy_data_flags_t;

/**
 * GRBM GFX Index structure
 * Matches Rust GrbmGFXIndex bitfield exactly
 */
typedef union {
    uint32_t raw;
    struct {
        uint32_t instance_index : 7;           /* bits 0-6 */
        uint32_t reserved1 : 1;                /* bit 7 */
        uint32_t sa_index : 2;                 /* bits 8-9 */
        uint32_t reserved2 : 6;                /* bits 10-15 */
        uint32_t se_index : 4;                 /* bits 16-19 */
        uint32_t reserved3 : 9;                /* bits 20-28 */
        uint32_t sa_broadcast_writes : 1;      /* bit 29 */
        uint32_t instance_broadcast_writes : 1;/* bit 30 */
        uint32_t se_broadcast_writes : 1;      /* bit 31 */
    } bits;
} pm4_grbm_gfx_index_t;

/**
 * Barrier Event structure
 * Matches Rust BarrierEvent bitfield exactly
 */
typedef union {
    uint32_t raw;
    struct {
        uint32_t event_type : 6;           /* bits 0-5 */
        uint32_t reserved1 : 2;            /* bits 6-7 */
        uint32_t event_index : 4;          /* bits 8-11 */
        uint32_t reserved2 : 9;            /* bits 12-20 */
        uint32_t samp_plst_cntr_mode : 2;  /* bits 21-22 */
        uint32_t offload_enable : 1;       /* bit 23 */
        uint32_t reserved3 : 8;            /* bits 24-31 */
    } bits;
} pm4_barrier_event_t;

/* Buffer management functions */

/**
 * @brief Create a new PM4 command buffer
 *
 * Allocates and initializes a growable buffer for accumulating PM4 packets.
 *
 * @param initial_capacity Initial buffer capacity in DWORDs (0 = use default)
 * @param flags Kernel memory allocation flags (kernel mode only)
 * @return Pointer to allocated buffer, or NULL on allocation failure
 *
 * @note Caller must call pm4_buffer_destroy() to free the buffer
 * @see pm4_buffer_destroy()
 */
#ifdef __KERNEL__
pm4_buffer_t* pm4_buffer_create(size_t initial_capacity, gfp_t flags);
#else
pm4_buffer_t* pm4_buffer_create(size_t initial_capacity);
#endif

/**
 * @brief Destroy a PM4 command buffer and free all memory
 *
 * @param buffer Pointer to buffer to destroy, or NULL (no-op)
 * @see pm4_buffer_create()
 */
void pm4_buffer_destroy(pm4_buffer_t *buffer);

/**
 * @brief Reset buffer to empty state without deallocating memory
 *
 * @param buffer Pointer to buffer to reset
 * @return 0 on success, -1 if buffer is NULL
 * @see pm4_buffer_create()
 */
int pm4_buffer_reset(pm4_buffer_t *buffer);

/**
 * @brief Ensure buffer has capacity for additional DWORDs
 *
 * Expands buffer capacity if needed to accommodate additional data.
 *
 * @param buffer Pointer to buffer to expand
 * @param required_dwords Number of additional DWORDs that will be appended
 * @return 0 on success, -1 on allocation failure or NULL buffer
 */
int pm4_buffer_ensure_capacity(pm4_buffer_t *buffer, size_t required_dwords);

/**
 * @brief Get pointer to the buffer's data array
 *
 * @param buffer Pointer to buffer
 * @return Pointer to data array, or NULL if buffer is NULL
 */
uint32_t* pm4_buffer_get_data(pm4_buffer_t *buffer);

/**
 * @brief Get current buffer size in DWORDs
 *
 * @param buffer Pointer to buffer
 * @return Current size in DWORDs, or 0 if buffer is NULL
 */
size_t pm4_buffer_get_size(pm4_buffer_t *buffer);

/**
 * @brief Get current buffer size in bytes
 *
 * @param buffer Pointer to buffer
 * @return Current size in bytes, or 0 if buffer is NULL
 */
size_t pm4_buffer_get_size_bytes(pm4_buffer_t *buffer);

/* PM4 packet builder functions - matches Rust PM4CommandWriter trait */

/* SetUConfigReg packet - writes to UCONFIG register space */
int pm4_append_set_uconfig_reg(pm4_buffer_t *buffer,
                                uint32_t reg_offset,
                                uint32_t value);

/* WriteSHRegister packet - writes to SH (persistent) register space */
int pm4_append_write_sh_reg(pm4_buffer_t *buffer,
                             uint32_t reg_offset,
                             uint32_t value,
                             uint8_t vmid_shift,
                             uint8_t index);

/* EventWrite packet - triggers GPU events */
int pm4_append_event_write(pm4_buffer_t *buffer,
                            uint32_t event_type,
                            uint32_t event_index);

/* CopyData packet - copies data from registers to memory */
int pm4_append_copy_data(pm4_buffer_t *buffer,
                          pm4_copy_data_flags_t flags,
                          uint32_t src_reg_lo,
                          uint32_t src_reg_hi,
                          uint64_t dst_addr);

/* AcquireMem (FlushCache) packet - ensures cache coherency */
int pm4_append_acquire_mem(pm4_buffer_t *buffer,
                            uint64_t base_addr,
                            uint64_t size,
                            uint32_t gcr_cntl);

/* Higher-level helper functions that match Rust implementation */

/**
 * @brief Set GRBM to broadcast mode for writing to all GPU instances
 *
 * @param buffer PM4 buffer to append packet to
 * @param grbm_gfx_index_reg GRBM_GFX_INDEX register offset
 * @return 0 on success, negative on error
 */
int pm4_grbm_broadcast(pm4_buffer_t *buffer, uint32_t grbm_gfx_index_reg);

/**
 * @brief Set specific GRBM index for targeted writes
 *
 * Configures GRBM_GFX_INDEX to target a specific SE/SA/WGP location.
 *
 * @param buffer PM4 buffer to append packet to
 * @param grbm_gfx_index_reg GRBM_GFX_INDEX register offset
 * @param wg_index Work group index (shifted by 2 before use)
 * @param sa_index Shader Array index
 * @param se_index Shader Engine index
 * @return 0 on success, negative on error
 */
int pm4_set_grbm_index(pm4_buffer_t *buffer,
                        uint32_t grbm_gfx_index_reg,
                        uint32_t wg_index,
                        uint32_t sa_index,
                        uint32_t se_index);

/**
 * @brief Enable or disable performance monitoring
 *
 * Writes to CP_PERFMON_CNTL register to control performance monitoring state.
 *
 * @param buffer PM4 buffer to append packet to
 * @param cp_perfmon_cntl_reg CP_PERFMON_CNTL register offset
 * @param control_value Control value to write (state + sample bit)
 * @return 0 on success, negative on error
 */
int pm4_perfcount_enable(pm4_buffer_t *buffer,
                          uint32_t cp_perfmon_cntl_reg,
                          uint32_t control_value);

/**
 * @brief Trigger CS partial flush event
 *
 * Appends an EVENT_WRITE packet for CS_PARTIAL_FLUSH synchronization.
 *
 * @param buffer PM4 buffer to append packet to
 * @return 0 on success, negative on error
 */
int pm4_cs_partial_flush(pm4_buffer_t *buffer);

/**
 * @brief Calculate cache coherency parameters for ACQUIRE_MEM packet
 *
 * Converts memory address and size into the coher_base/coher_size format
 * required by ACQUIRE_MEM packets. Uses 256-byte granularity.
 *
 * @param addr Base memory address to flush
 * @param size Size of memory range in bytes
 * @param coher_size_lo Output: coherency size low 32 bits
 * @param coher_size_hi Output: coherency size high 8 bits
 * @param coher_base_lo Output: coherency base low 32 bits
 * @param coher_base_hi Output: coherency base high 24 bits
 */
void pm4_calculate_cache_coher_params(uint64_t addr,
                                       uint64_t size,
                                       uint32_t *coher_size_lo,
                                       uint32_t *coher_size_hi,
                                       uint32_t *coher_base_lo,
                                       uint32_t *coher_base_hi);

/* Debug/utility functions */

/**
 * @brief Dump a PM4 packet to console for debugging
 *
 * @param packet Pointer to packet data
 * @param size_dwords Size of packet in DWORDs
 */
void pm4_dump_packet(const uint32_t *packet, size_t size_dwords);

/**
 * @brief Convert PM4 opcode to string name
 *
 * @param opcode PM4 opcode value
 * @return String name of opcode, or "UNKNOWN"
 */
const char* pm4_opcode_to_string(uint8_t opcode);

/**
 * @brief Validate a PM4 packet structure
 *
 * Checks packet header and size for correctness.
 *
 * @param packet Pointer to packet data
 * @param size_dwords Size of packet in DWORDs
 * @return 0 if valid, -1 if invalid
 */
int pm4_validate_packet(const uint32_t *packet, size_t size_dwords);

/**
 * PM4 Operation Union
 * Contains all supported PM4 packet types with discriminator
 * Used for handling variable-sized PM4 packets in a type-safe manner
 */
typedef enum {
    PM4_OP_SET_UCONFIG_REG,
    PM4_OP_EVENT_WRITE,
    PM4_OP_COPY_DATA,
    PM4_OP_FLUSH_CACHE,
    PM4_OP_WRITE_SH_REG,
    PM4_OP_INVALID
} pm4_op_type_t;

typedef struct {
    pm4_op_type_t type;
    size_t size_dwords;
    union {
        /* SetUConfigReg packet (3 DWORDs) */
        struct {
            uint32_t header;
            uint16_t reg_offset;
            uint16_t reserved;
            uint32_t reg_value;
        } set_uconfig_reg;

        /* EventWrite packet (2 DWORDs) */
        struct {
            uint32_t header;
            uint32_t event;
        } event_write;

        /* CopyData packet (6 DWORDs) */
        struct {
            uint32_t header;
            uint32_t copy_data;
            uint32_t src_reg_offset_lo;
            uint32_t src_reg_offset_hi;
            uint32_t dst_reg_offset_lo;
            uint32_t dst_reg_offset_hi;
        } copy_data;

        /* FlushCache packet (8 DWORDs) */
        struct {
            uint32_t header;
            uint32_t reserved;
            uint32_t coher_size;
            uint32_t coher_size_hi;
            uint32_t coher_base_lo;
            uint32_t coher_base_hi;
            uint32_t poll_interval;
            uint32_t gcr_cntl;
        } flush_cache;

        /* WriteSHRegister packet (3 DWORDs) */
        struct {
            uint32_t header;
            uint32_t word1;  /* Contains reg_offset, vmid_shift, index fields */
            uint32_t reg_value;
        } write_sh_reg;

        /* Raw access for unknown packet types */
        uint32_t raw[8];  /* Max packet size is 8 DWORDs */
    } packet;
} pm4_op_t;

/* Helper functions for pm4_op_t */

/**
 * @brief Parse PM4 packet data into a pm4_op_t structure
 *
 * @param data Pointer to PM4 packet data
 * @param size_dwords Size of packet in DWORDs
 * @return Parsed pm4_op_t structure (type set to PM4_OP_INVALID on error)
 */
pm4_op_t pm4_op_from_buffer(const uint32_t* data, size_t size_dwords);

/**
 * @brief Serialize a pm4_op_t structure to buffer
 *
 * @param op Pointer to PM4 operation structure
 * @param buffer Output buffer for packet data
 * @param buffer_size Size of output buffer in DWORDs
 * @return Number of DWORDs written, or -1 on error
 */
int pm4_op_to_buffer(const pm4_op_t* op, uint32_t* buffer, size_t buffer_size);

/**
 * @brief Get size of a PM4 operation in DWORDs
 *
 * @param op Pointer to PM4 operation structure
 * @return Size in DWORDs, or 0 if op is NULL
 */
size_t pm4_op_get_size(const pm4_op_t* op);

/**
 * @brief Convert PM4 operation type to string name
 *
 * @param type PM4 operation type
 * @return String name of operation type
 */
const char* pm4_op_type_to_string(pm4_op_type_t type);

#endif /* PM4_PACKETS_H */