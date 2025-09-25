/**
 * @file pm4_packets.c
 * @brief PM4 packet builder implementation for constructing GPU command buffers
 *
 * Implements the builder pattern for PM4 packets with exact binary compatibility
 * to the Rust implementation in rocprofiler-oop/kfd/amd/amdgpu/gfx12/impls.rs
 */

#ifdef __KERNEL__
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>
#else
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#endif

#include "pm4_packets.h"

/* Buffer management implementation */

#ifdef __KERNEL__
pm4_buffer_t* pm4_buffer_create(size_t initial_capacity, gfp_t flags)
#else
pm4_buffer_t* pm4_buffer_create(size_t initial_capacity)
#endif
{
    pm4_buffer_t *buffer;

    if (initial_capacity == 0) {
        initial_capacity = 256;  /* Default 256 DWORDs = 1KB */
    }

#ifdef __KERNEL__
    buffer = kmalloc(sizeof(pm4_buffer_t), flags);
    if (!buffer)
        return NULL;

    buffer->data = kmalloc(initial_capacity * sizeof(uint32_t), flags);
    buffer->gfp_flags = flags;
#else
    buffer = malloc(sizeof(pm4_buffer_t));
    if (!buffer)
        return NULL;

    buffer->data = malloc(initial_capacity * sizeof(uint32_t));
#endif

    if (!buffer->data) {
#ifdef __KERNEL__
        kfree(buffer);
#else
        free(buffer);
#endif
        return NULL;
    }

    buffer->capacity = initial_capacity;
    buffer->size = 0;

    return buffer;
}

void pm4_buffer_destroy(pm4_buffer_t *buffer)
{
    if (!buffer)
        return;

    if (buffer->data) {
#ifdef __KERNEL__
        kfree(buffer->data);
#else
        free(buffer->data);
#endif
    }

#ifdef __KERNEL__
    kfree(buffer);
#else
    free(buffer);
#endif
}

int pm4_buffer_reset(pm4_buffer_t *buffer)
{
    if (!buffer)
        return -1;

    buffer->size = 0;
    return 0;
}

int pm4_buffer_ensure_capacity(pm4_buffer_t *buffer, size_t required_dwords)
{
    size_t new_capacity;
    uint32_t *new_data;

    if (!buffer)
        return -1;

    if (buffer->size + required_dwords <= buffer->capacity)
        return 0;  /* Already have enough space */

    /* Double capacity or add required amount, whichever is larger */
    new_capacity = buffer->capacity * 2;
    if (new_capacity < buffer->size + required_dwords) {
        new_capacity = buffer->size + required_dwords + 256;  /* Add some extra */
    }

#ifdef __KERNEL__
    new_data = krealloc(buffer->data, new_capacity * sizeof(uint32_t), buffer->gfp_flags);
#else
    new_data = realloc(buffer->data, new_capacity * sizeof(uint32_t));
#endif

    if (!new_data)
        return -1;  /* Failed to allocate */

    buffer->data = new_data;
    buffer->capacity = new_capacity;

    return 0;
}

uint32_t* pm4_buffer_get_data(pm4_buffer_t *buffer)
{
    return buffer ? buffer->data : NULL;
}

size_t pm4_buffer_get_size(pm4_buffer_t *buffer)
{
    return buffer ? buffer->size : 0;
}

size_t pm4_buffer_get_size_bytes(pm4_buffer_t *buffer)
{
    return buffer ? buffer->size * sizeof(uint32_t) : 0;
}

/* Internal helper to append DWORDs to buffer */
static int pm4_buffer_append(pm4_buffer_t *buffer, const uint32_t *data, size_t count)
{
    if (!buffer || !data)
        return -1;

    if (pm4_buffer_ensure_capacity(buffer, count) < 0)
        return -1;

    memcpy(&buffer->data[buffer->size], data, count * sizeof(uint32_t));
    buffer->size += count;

    return 0;
}

/* PM4 packet builder implementations */

int pm4_append_set_uconfig_reg(pm4_buffer_t *buffer,
                                uint32_t reg_offset,
                                uint32_t value)
{
    uint32_t packet[3];
    uint16_t offset;

    if (!buffer)
        return -1;

    /* Calculate register offset from UCONFIG base */
    offset = (uint16_t)((reg_offset - UCONFIG_SPACE_START) & 0xFFFF);

    /* Build packet matching aqlprofile SetUConfigReg structure */
    packet[0] = PM4_TYPE_3_HEADER(PM4_SET_UCONFIG_REG_OPCODE, 12);  /* 3 DWORDs = 12 bytes */
    packet[1] = offset & 0xFFFF;  /* reg_offset in lower 16 bits */
    packet[2] = value;

    return pm4_buffer_append(buffer, packet, 3);
}

int pm4_append_write_sh_reg(pm4_buffer_t *buffer,
                             uint32_t reg_offset,
                             uint32_t value,
                             uint8_t vmid_shift,
                             uint8_t index)
{
    uint32_t packet[3];
    uint16_t offset;
    uint32_t word1;

    if (!buffer)
        return -1;

    /* Calculate register offset from PERSISTENT base */
    offset = (uint16_t)((reg_offset - PERSISTENT_SPACE_START) & 0xFFFF);

    /* Build word1 with bitfield layout matching Rust WriteSHRegister */
    word1 = (offset & 0xFFFF) |           /* bits 0-15: reg_offset */
            ((vmid_shift & 0x1F) << 23) |  /* bits 23-27: vmid_shift */
            ((index & 0xF) << 28);          /* bits 28-31: index */

    /* Build packet */
    packet[0] = PM4_TYPE_3_HEADER(PM4_WRITE_SH_REG_OPCODE, 12);  /* 3 DWORDs = 12 bytes */
    packet[1] = word1;
    packet[2] = value;

    return pm4_buffer_append(buffer, packet, 3);
}

int pm4_append_event_write(pm4_buffer_t *buffer,
                            uint32_t event_type,
                            uint32_t event_index)
{
    uint32_t packet[2];
    pm4_barrier_event_t event = {0};

    if (!buffer)
        return -1;

    /* Build event structure matching Rust BarrierEvent */
    event.bits.event_type = event_type & 0x3F;
    event.bits.event_index = event_index & 0xF;

    /* Build packet matching aqlprofile EventWrite */
    packet[0] = PM4_TYPE_3_HEADER(PM4_EVENT_WRITE_OPCODE, 8);  /* 2 DWORDs = 8 bytes */
    packet[1] = event.raw & 0xFFFF;  /* Only lower 16 bits used */

    return pm4_buffer_append(buffer, packet, 2);
}

int pm4_append_copy_data(pm4_buffer_t *buffer,
                          pm4_copy_data_flags_t flags,
                          uint32_t src_reg_lo,
                          uint32_t src_reg_hi,
                          uint64_t dst_addr)
{
    uint32_t packet[6];

    if (!buffer)
        return -1;

    /* Build packet matching aqlprofile CopyData */
    packet[0] = PM4_TYPE_3_HEADER(PM4_COPY_DATA_OPCODE, 24);  /* 6 DWORDs = 24 bytes */
    packet[1] = flags.raw;
    packet[2] = src_reg_lo;
    packet[3] = src_reg_hi;
    packet[4] = (uint32_t)(dst_addr & 0xFFFFFFFF);
    packet[5] = (uint32_t)(dst_addr >> 32);

    return pm4_buffer_append(buffer, packet, 6);
}

int pm4_append_acquire_mem(pm4_buffer_t *buffer,
                            uint64_t base_addr,
                            uint64_t size,
                            uint32_t gcr_cntl)
{
    uint32_t packet[8];
    uint32_t coher_size_lo, coher_size_hi;
    uint32_t coher_base_lo, coher_base_hi;

    if (!buffer)
        return -1;

    /* Calculate coherency parameters matching Rust cache_cohere */
    pm4_calculate_cache_coher_params(base_addr, size,
                                      &coher_size_lo, &coher_size_hi,
                                      &coher_base_lo, &coher_base_hi);

    /* Build packet matching aqlprofile AcquireMem */
    packet[0] = PM4_TYPE_3_HEADER(PM4_ACQUIRE_MEM_OPCODE, 32);  /* 8 DWORDs = 32 bytes */
    packet[1] = 0;  /* Reserved */
    packet[2] = coher_size_lo;
    packet[3] = coher_size_hi & 0xFF;  /* Only 8 bits for size_hi */
    packet[4] = coher_base_lo;
    packet[5] = coher_base_hi & 0xFFFFFF;  /* Only 24 bits for base_hi */
    packet[6] = 0x10 & 0xFFFF;  /* poll_interval = 0x10 (poll every 4K) */
    packet[7] = gcr_cntl & 0x7FFFF;  /* 19 bits for gcr_cntl */

    return pm4_buffer_append(buffer, packet, 8);
}

/* Higher-level helper functions */

int pm4_grbm_broadcast(pm4_buffer_t *buffer, uint32_t grbm_gfx_index_reg)
{
    pm4_grbm_gfx_index_t index = {0};

    /* Set broadcast mode for all units */
    index.bits.instance_index = 0;
    index.bits.sa_index = 0;
    index.bits.se_index = 0;
    index.bits.sa_broadcast_writes = 1;
    index.bits.instance_broadcast_writes = 1;
    index.bits.se_broadcast_writes = 1;

    return pm4_append_set_uconfig_reg(buffer, grbm_gfx_index_reg, index.raw);
}

int pm4_set_grbm_index(pm4_buffer_t *buffer,
                        uint32_t grbm_gfx_index_reg,
                        uint32_t wg_index,
                        uint32_t sa_index,
                        uint32_t se_index)
{
    pm4_grbm_gfx_index_t index = {0};

    /* Set specific index values matching Rust set_grbm_index */
    index.bits.instance_index = (wg_index << 2) & 0x7F;  /* wg_index shifted by 2 */
    index.bits.sa_index = sa_index & 0x3;
    index.bits.se_index = se_index & 0xF;

    return pm4_append_set_uconfig_reg(buffer, grbm_gfx_index_reg, index.raw);
}

int pm4_perfcount_enable(pm4_buffer_t *buffer,
                          uint32_t cp_perfmon_cntl_reg,
                          uint32_t control_value)
{
    return pm4_append_set_uconfig_reg(buffer, cp_perfmon_cntl_reg, control_value);
}

int pm4_cs_partial_flush(pm4_buffer_t *buffer)
{
    /* Trigger CS partial flush event */
    return pm4_append_event_write(buffer, VGT_EVENT_TYPE_CS_PARTIAL_FLUSH, 4);
}

void pm4_calculate_cache_coher_params(uint64_t addr,
                                       uint64_t size,
                                       uint32_t *coher_size_lo,
                                       uint32_t *coher_size_hi,
                                       uint32_t *coher_base_lo,
                                       uint32_t *coher_base_hi)
{
    uint64_t align;
    uint64_t boundaries_crossed;
    uint64_t ceiling;
    uint64_t divided;
    uint64_t coher_size;

    /* Match Rust cache_cohere calculation exactly */
    align = addr % 256;
    boundaries_crossed = (align + size) >> 8;
    ceiling = (size + 0xFF) >> 8;  /* Ceiling of size/256 */
    divided = size >> 8;

    coher_size = boundaries_crossed + ceiling - divided;

    /* Set output parameters */
    *coher_size_lo = (uint32_t)(coher_size & 0xFFFFFFFF);
    *coher_size_hi = (uint32_t)(coher_size >> 32);
    *coher_base_lo = (uint32_t)((addr >> 8) & 0xFFFFFFFF);
    *coher_base_hi = (uint32_t)((addr >> 40) & 0xFFFFFF);
}

/* Debug/utility functions */

void pm4_dump_packet(const uint32_t *packet, size_t size_dwords)
{
    size_t i;

    if (!packet || size_dwords == 0)
        return;

#ifdef __KERNEL__
    printk(KERN_INFO "PM4 Packet (%zu DWORDs):\n", size_dwords);
    for (i = 0; i < size_dwords; i++) {
        printk(KERN_INFO "  [%zu]: 0x%08X\n", i, packet[i]);
    }
#else
    printf("PM4 Packet (%zu DWORDs):\n", size_dwords);
    for (i = 0; i < size_dwords; i++) {
        printf("  [%zu]: 0x%08X\n", i, packet[i]);
    }
#endif
}

const char* pm4_opcode_to_string(uint8_t opcode)
{
    switch (opcode) {
        case PM4_SET_UCONFIG_REG_OPCODE:  return "SET_UCONFIG_REG";
        case PM4_EVENT_WRITE_OPCODE:       return "EVENT_WRITE";
        case PM4_COPY_DATA_OPCODE:         return "COPY_DATA";
        case PM4_ACQUIRE_MEM_OPCODE:       return "ACQUIRE_MEM";
        case PM4_WRITE_SH_REG_OPCODE:      return "WRITE_SH_REG";
        default:                           return "UNKNOWN";
    }
}

int pm4_validate_packet(const uint32_t *packet, size_t size_dwords)
{
    uint32_t header;
    uint8_t type;
    uint8_t opcode;
    uint16_t count;
    size_t expected_size;

    if (!packet || size_dwords == 0)
        return -1;

    header = packet[0];
    type = (header >> 30) & 0x3;

    if (type != 3) {
        /* Not a Type 3 packet */
        return -1;
    }

    opcode = (header >> 8) & 0xFF;
    count = ((header >> 16) & 0x3FFF) + 1;  /* Count field is size-1 */
    expected_size = count + 1;  /* +1 for header */

    if (expected_size != size_dwords) {
        /* Size mismatch */
        return -1;
    }

    /* Validate known opcodes */
    switch (opcode) {
        case PM4_SET_UCONFIG_REG_OPCODE:
            return (expected_size == 3) ? 0 : -1;
        case PM4_EVENT_WRITE_OPCODE:
            return (expected_size == 2) ? 0 : -1;
        case PM4_COPY_DATA_OPCODE:
            return (expected_size == 6) ? 0 : -1;
        case PM4_ACQUIRE_MEM_OPCODE:
            return (expected_size == 8) ? 0 : -1;
        case PM4_WRITE_SH_REG_OPCODE:
            return (expected_size == 3) ? 0 : -1;
        default:
            /* Unknown opcode, but structure is valid */
            return 0;
    }
}

/* pm4_op_t helper functions implementation */

pm4_op_t pm4_op_from_buffer(const uint32_t* data, size_t size_dwords)
{
    pm4_op_t op = {0};
    uint32_t header;
    uint8_t opcode;

    if (!data || size_dwords == 0) {
        op.type = PM4_OP_INVALID;
        return op;
    }

    header = data[0];
    opcode = (header >> 8) & 0xFF;
    op.size_dwords = size_dwords;

    /* Determine packet type from opcode */
    switch (opcode) {
        case PM4_SET_UCONFIG_REG_OPCODE:
            op.type = PM4_OP_SET_UCONFIG_REG;
            if (size_dwords >= 3) {
                op.packet.set_uconfig_reg.header = data[0];
                op.packet.set_uconfig_reg.reg_offset = data[1] & 0xFFFF;
                op.packet.set_uconfig_reg.reserved = (data[1] >> 16) & 0xFFFF;
                op.packet.set_uconfig_reg.reg_value = data[2];
            }
            break;

        case PM4_EVENT_WRITE_OPCODE:
            op.type = PM4_OP_EVENT_WRITE;
            if (size_dwords >= 2) {
                op.packet.event_write.header = data[0];
                op.packet.event_write.event = data[1];
            }
            break;

        case PM4_COPY_DATA_OPCODE:
            op.type = PM4_OP_COPY_DATA;
            if (size_dwords >= 6) {
                op.packet.copy_data.header = data[0];
                op.packet.copy_data.copy_data = data[1];
                op.packet.copy_data.src_reg_offset_lo = data[2];
                op.packet.copy_data.src_reg_offset_hi = data[3];
                op.packet.copy_data.dst_reg_offset_lo = data[4];
                op.packet.copy_data.dst_reg_offset_hi = data[5];
            }
            break;

        case PM4_ACQUIRE_MEM_OPCODE:
            op.type = PM4_OP_FLUSH_CACHE;
            if (size_dwords >= 8) {
                op.packet.flush_cache.header = data[0];
                op.packet.flush_cache.reserved = data[1];
                op.packet.flush_cache.coher_size = data[2];
                op.packet.flush_cache.coher_size_hi = data[3];
                op.packet.flush_cache.coher_base_lo = data[4];
                op.packet.flush_cache.coher_base_hi = data[5];
                op.packet.flush_cache.poll_interval = data[6];
                op.packet.flush_cache.gcr_cntl = data[7];
            }
            break;

        case PM4_WRITE_SH_REG_OPCODE:
            op.type = PM4_OP_WRITE_SH_REG;
            if (size_dwords >= 3) {
                op.packet.write_sh_reg.header = data[0];
                op.packet.write_sh_reg.word1 = data[1];
                op.packet.write_sh_reg.reg_value = data[2];
            }
            break;

        default:
            op.type = PM4_OP_INVALID;
            /* Copy raw data up to available space */
            size_t copy_size = (size_dwords < 8) ? size_dwords : 8;
            memcpy(op.packet.raw, data, copy_size * sizeof(uint32_t));
            break;
    }

    return op;
}

int pm4_op_to_buffer(const pm4_op_t* op, uint32_t* buffer, size_t buffer_size)
{
    if (!op || !buffer || buffer_size == 0)
        return -1;

    if (op->size_dwords > buffer_size)
        return -1;  /* Buffer too small */

    switch (op->type) {
        case PM4_OP_SET_UCONFIG_REG:
            if (buffer_size < 3) return -1;
            buffer[0] = op->packet.set_uconfig_reg.header;
            buffer[1] = op->packet.set_uconfig_reg.reg_offset |
                       (op->packet.set_uconfig_reg.reserved << 16);
            buffer[2] = op->packet.set_uconfig_reg.reg_value;
            return 3;

        case PM4_OP_EVENT_WRITE:
            if (buffer_size < 2) return -1;
            buffer[0] = op->packet.event_write.header;
            buffer[1] = op->packet.event_write.event;
            return 2;

        case PM4_OP_COPY_DATA:
            if (buffer_size < 6) return -1;
            buffer[0] = op->packet.copy_data.header;
            buffer[1] = op->packet.copy_data.copy_data;
            buffer[2] = op->packet.copy_data.src_reg_offset_lo;
            buffer[3] = op->packet.copy_data.src_reg_offset_hi;
            buffer[4] = op->packet.copy_data.dst_reg_offset_lo;
            buffer[5] = op->packet.copy_data.dst_reg_offset_hi;
            return 6;

        case PM4_OP_FLUSH_CACHE:
            if (buffer_size < 8) return -1;
            buffer[0] = op->packet.flush_cache.header;
            buffer[1] = op->packet.flush_cache.reserved;
            buffer[2] = op->packet.flush_cache.coher_size;
            buffer[3] = op->packet.flush_cache.coher_size_hi;
            buffer[4] = op->packet.flush_cache.coher_base_lo;
            buffer[5] = op->packet.flush_cache.coher_base_hi;
            buffer[6] = op->packet.flush_cache.poll_interval;
            buffer[7] = op->packet.flush_cache.gcr_cntl;
            return 8;

        case PM4_OP_WRITE_SH_REG:
            if (buffer_size < 3) return -1;
            buffer[0] = op->packet.write_sh_reg.header;
            buffer[1] = op->packet.write_sh_reg.word1;
            buffer[2] = op->packet.write_sh_reg.reg_value;
            return 3;

        case PM4_OP_INVALID:
        default:
            /* Copy raw data */
            size_t copy_size = (op->size_dwords < buffer_size) ? op->size_dwords : buffer_size;
            memcpy(buffer, op->packet.raw, copy_size * sizeof(uint32_t));
            return (int)copy_size;
    }
}

size_t pm4_op_get_size(const pm4_op_t* op)
{
    if (!op)
        return 0;

    switch (op->type) {
        case PM4_OP_SET_UCONFIG_REG:    return 3;
        case PM4_OP_EVENT_WRITE:        return 2;
        case PM4_OP_COPY_DATA:          return 6;
        case PM4_OP_FLUSH_CACHE:        return 8;
        case PM4_OP_WRITE_SH_REG:       return 3;
        case PM4_OP_INVALID:
        default:                        return op->size_dwords;
    }
}

const char* pm4_op_type_to_string(pm4_op_type_t type)
{
    switch (type) {
        case PM4_OP_SET_UCONFIG_REG:    return "SET_UCONFIG_REG";
        case PM4_OP_EVENT_WRITE:        return "EVENT_WRITE";
        case PM4_OP_COPY_DATA:          return "COPY_DATA";
        case PM4_OP_FLUSH_CACHE:        return "FLUSH_CACHE";
        case PM4_OP_WRITE_SH_REG:       return "WRITE_SH_REG";
        case PM4_OP_INVALID:
        default:                        return "INVALID";
    }
}