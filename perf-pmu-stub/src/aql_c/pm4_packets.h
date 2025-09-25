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
#ifdef __KERNEL__
pm4_buffer_t* pm4_buffer_create(size_t initial_capacity, gfp_t flags);
#else
pm4_buffer_t* pm4_buffer_create(size_t initial_capacity);
#endif
void pm4_buffer_destroy(pm4_buffer_t *buffer);
int pm4_buffer_reset(pm4_buffer_t *buffer);
int pm4_buffer_ensure_capacity(pm4_buffer_t *buffer, size_t required_dwords);
uint32_t* pm4_buffer_get_data(pm4_buffer_t *buffer);
size_t pm4_buffer_get_size(pm4_buffer_t *buffer);
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

/* Set GRBM to broadcast mode */
int pm4_grbm_broadcast(pm4_buffer_t *buffer, uint32_t grbm_gfx_index_reg);

/* Set specific GRBM index for targeted writes */
int pm4_set_grbm_index(pm4_buffer_t *buffer,
                        uint32_t grbm_gfx_index_reg,
                        uint32_t wg_index,
                        uint32_t sa_index,
                        uint32_t se_index);

/* Enable performance counters */
int pm4_perfcount_enable(pm4_buffer_t *buffer,
                          uint32_t cp_perfmon_cntl_reg,
                          uint32_t control_value);

/* Trigger CS partial flush */
int pm4_cs_partial_flush(pm4_buffer_t *buffer);

/* Calculate cache coherency parameters (helper) */
void pm4_calculate_cache_coher_params(uint64_t addr,
                                       uint64_t size,
                                       uint32_t *coher_size_lo,
                                       uint32_t *coher_size_hi,
                                       uint32_t *coher_base_lo,
                                       uint32_t *coher_base_hi);

/* Debug/utility functions */
void pm4_dump_packet(const uint32_t *packet, size_t size_dwords);
const char* pm4_opcode_to_string(uint8_t opcode);
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
pm4_op_t pm4_op_from_buffer(const uint32_t* data, size_t size_dwords);
int pm4_op_to_buffer(const pm4_op_t* op, uint32_t* buffer, size_t buffer_size);
size_t pm4_op_get_size(const pm4_op_t* op);
const char* pm4_op_type_to_string(pm4_op_type_t type);

#endif /* PM4_PACKETS_H */