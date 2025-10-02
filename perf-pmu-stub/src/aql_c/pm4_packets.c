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

/**
 * @brief Create a new PM4 command buffer with specified capacity
 *
 * Allocates and initializes a growable buffer for accumulating PM4 packets.
 * The buffer will automatically expand as needed when packets are appended.
 *
 * This function is analogous to CmdBuffer::CmdBuffer() in
 * projects/aqlprofile/src/pm4/cmd_builder.h:90-132, but implemented in C
 * with explicit memory management instead of using std::vector.
 *
 * @param initial_capacity Initial buffer capacity in DWORDs (32-bit words)
 *                        If 0, defaults to 256 DWORDs (1KB)
 * @param flags Kernel memory allocation flags (kernel mode only)
 * @return Pointer to allocated buffer, or NULL on allocation failure
 *
 * @note Caller is responsible for freeing the buffer with pm4_buffer_destroy()
 * @see pm4_buffer_destroy(), CmdBuffer in projects/aqlprofile/src/pm4/cmd_builder.h:90
 */
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

/**
 * @brief Destroy a PM4 command buffer and free all associated memory
 *
 * Frees both the buffer's internal data array and the buffer structure itself.
 * Safe to call with NULL pointer (no-op).
 *
 * This function is analogous to CmdBuffer::~CmdBuffer() (implicit destructor) in
 * projects/aqlprofile/src/pm4/cmd_builder.h:131, but implemented explicitly
 * in C since there are no destructors.
 *
 * @param buffer Pointer to buffer to destroy, or NULL
 *
 * @note After calling this function, the buffer pointer is invalid
 * @see pm4_buffer_create(), CmdBuffer::~CmdBuffer in projects/aqlprofile/src/pm4/cmd_builder.h
 */
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

/**
 * @brief Reset buffer to empty state without deallocating memory
 *
 * Resets the buffer's size to 0, allowing it to be reused while retaining
 * the allocated capacity. Useful for reusing buffers across multiple operations.
 *
 * This function is analogous to CmdBuffer::Clear() in
 * projects/aqlprofile/src/pm4/cmd_builder.h:127
 *
 * @param buffer Pointer to buffer to reset
 * @return 0 on success, -1 if buffer is NULL
 *
 * @see pm4_buffer_create(), CmdBuffer::Clear in projects/aqlprofile/src/pm4/cmd_builder.h:127
 */
int pm4_buffer_reset(pm4_buffer_t *buffer)
{
    if (!buffer)
        return -1;

    buffer->size = 0;
    return 0;
}

/**
 * @brief Ensure buffer has capacity for additional DWORDs
 *
 * Expands the buffer's capacity if needed to accommodate additional data.
 * Uses a doubling strategy for efficient amortized growth.
 *
 * This function implements the automatic growth behavior implicit in
 * CmdBuffer::Append() via std::vector::resize() in
 * projects/aqlprofile/src/pm4/cmd_builder.h:96-98
 *
 * @param buffer Pointer to buffer to expand
 * @param required_dwords Number of additional DWORDs that will be appended
 * @return 0 on success, -1 on allocation failure or NULL buffer
 *
 * @note Uses doubling strategy: new_capacity = max(capacity*2, size+required+256)
 * @see pm4_buffer_append(), CmdBuffer::Append in projects/aqlprofile/src/pm4/cmd_builder.h:96
 */
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

/**
 * @brief Get pointer to the buffer's data array
 *
 * @param buffer Pointer to PM4 buffer
 * @return Pointer to data array, or NULL if buffer is NULL
 */
uint32_t* pm4_buffer_get_data(pm4_buffer_t *buffer)
{
    return buffer ? buffer->data : NULL;
}

/**
 * @brief Get current buffer size in DWORDs
 *
 * @param buffer Pointer to PM4 buffer
 * @return Current size in DWORDs, or 0 if buffer is NULL
 */
size_t pm4_buffer_get_size(pm4_buffer_t *buffer)
{
    return buffer ? buffer->size : 0;
}

/**
 * @brief Get current buffer size in bytes
 *
 * @param buffer Pointer to PM4 buffer
 * @return Current size in bytes, or 0 if buffer is NULL
 */
size_t pm4_buffer_get_size_bytes(pm4_buffer_t *buffer)
{
    return buffer ? buffer->size * sizeof(uint32_t) : 0;
}

/**
 * @brief Internal helper to append DWORDs to buffer
 *
 * Appends raw DWORD data to the buffer, ensuring capacity and copying data.
 * Used internally by all pm4_append_* functions.
 *
 * @param buffer PM4 buffer to append to
 * @param data Pointer to data to append
 * @param count Number of DWORDs to append
 * @return 0 on success, -1 on failure
 */
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

/**
 * @brief Append a SET_UCONFIG_REG packet to write a UCONFIG register
 *
 * Builds and appends a Type 3 PM4 packet to write a value to a UCONFIG
 * (User Configuration) register space. UCONFIG registers are typically used
 * for performance counters and GPU configuration.
 *
 * This function is analogous to Gfx12CmdBuilder::BuildWriteUConfigRegPacket() in
 * projects/aqlprofile/src/pm4/gfx12_cmd_builder.h:185-195
 *
 * @param buffer PM4 buffer to append packet to
 * @param reg_offset Absolute register offset (will be adjusted to UCONFIG space base)
 * @param value 32-bit value to write to the register
 * @return 0 on success, -1 on failure
 *
 * @note Packet format: 3 DWORDs (header + offset + value)
 * @note Register offset is automatically adjusted by subtracting UCONFIG_SPACE_START
 * @see pm4_append_write_sh_reg(), Gfx12CmdBuilder::BuildWriteUConfigRegPacket in
 *      projects/aqlprofile/src/pm4/gfx12_cmd_builder.h:185
 */
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

/**
 * @brief Append a WRITE_SH_REG packet to write a persistent (SH) register
 *
 * Builds and appends a Type 3 PM4 packet to write a value to the SH
 * (Shader/Persistent) register space. These registers maintain their values
 * across shader dispatches within a queue.
 *
 * This function is analogous to Gfx12CmdBuilder::BuildWriteShRegPacket() in
 * projects/aqlprofile/src/pm4/gfx12_cmd_builder.h:173-183
 *
 * @param buffer PM4 buffer to append packet to
 * @param reg_offset Absolute register offset (will be adjusted to PERSISTENT space base)
 * @param value 32-bit value to write to the register
 * @param vmid_shift VMID shift value for multi-VMID contexts (bits 23-27)
 * @param index Register index field (bits 28-31), typically 0 for default
 * @return 0 on success, -1 on failure
 *
 * @note Packet format: 3 DWORDs (header + word1 + value)
 * @note word1 encodes offset, vmid_shift, and index as bitfields
 * @see pm4_append_set_uconfig_reg(), Gfx12CmdBuilder::BuildWriteShRegPacket in
 *      projects/aqlprofile/src/pm4/gfx12_cmd_builder.h:173
 */
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

/**
 * @brief Append an EVENT_WRITE packet to trigger a GPU event
 *
 * Builds and appends a Type 3 PM4 packet to signal a GPU event such as
 * cache flushes, synchronization barriers, or other GPU-side operations.
 * Commonly used for CS_PARTIAL_FLUSH and other synchronization events.
 *
 * This function is analogous to Gfx12CmdBuilder::BuildBarrierCommand() and
 * BuildThreadTraceCommand() in projects/aqlprofile/src/pm4/gfx12_cmd_builder.h:72-88
 *
 * @param buffer PM4 buffer to append packet to
 * @param event_type Event type code (e.g., CS_PARTIAL_FLUSH = 0x07)
 * @param event_index Event index field for routing (typically 4 for flush events)
 * @return 0 on success, -1 on failure
 *
 * @note Packet format: 2 DWORDs (header + event data)
 * @note Event data encodes both event_type (bits 0-5) and event_index (bits 8-11)
 * @see pm4_cs_partial_flush(), Gfx12CmdBuilder::BuildBarrierCommand in
 *      projects/aqlprofile/src/pm4/gfx12_cmd_builder.h:72
 */
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

/**
 * @brief Append a COPY_DATA packet to copy data from register to memory
 *
 * Builds and appends a Type 3 PM4 packet to copy counter data from GPU registers
 * to system memory. Used extensively for reading performance counter values.
 *
 * This function is analogous to Gfx12CmdBuilder::BuildCopyRegDataPacket() and
 * BuildCopyCounterDataPacket() in projects/aqlprofile/src/pm4/gfx12_cmd_builder.h:227-279
 *
 * @param buffer PM4 buffer to append packet to
 * @param flags Copy operation flags (source/dest selectors, cache policy, etc.)
 * @param src_reg_lo Source register address (low 32 bits)
 * @param src_reg_hi Source register address (high 32 bits, typically 0 for registers)
 * @param dst_addr Destination memory address (64-bit GPU address)
 * @return 0 on success, -1 on failure
 *
 * @note Packet format: 6 DWORDs (header + flags + src_lo + src_hi + dst_lo + dst_hi)
 * @note flags.bits.src_sel=0 for registers, dst_sel=2 for TC_L2 memory
 * @see pm4_set_grbm_index(), Gfx12CmdBuilder::BuildCopyRegDataPacket in
 *      projects/aqlprofile/src/pm4/gfx12_cmd_builder.h:227
 */
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

/**
 * @brief Append an ACQUIRE_MEM packet for cache coherency and synchronization
 *
 * Builds and appends a Type 3 PM4 packet to ensure cache coherency by flushing
 * and invalidating GPU caches for a specified memory range. Critical for ensuring
 * that counter data written to memory is visible to the CPU.
 *
 * This function is analogous to Gfx12CmdBuilder::BuildCacheFlushPacket() in
 * projects/aqlprofile/src/pm4/gfx12_cmd_builder.h:92-124
 *
 * @param buffer PM4 buffer to append packet to
 * @param base_addr Base address of memory range to flush (256-byte aligned)
 * @param size Size of memory range in bytes
 * @param gcr_cntl GCR control value for cache operations (GL2 writeback, etc.)
 * @return 0 on success, -1 on failure
 *
 * @note Packet format: 8 DWORDs (header + reserved + coher_size + coher_size_hi +
 *       coher_base_lo + coher_base_hi + poll_interval + gcr_cntl)
 * @note Address and size are converted to 256-byte granularity internally
 * @see pm4_calculate_cache_coher_params(), Gfx12CmdBuilder::BuildCacheFlushPacket in
 *      projects/aqlprofile/src/pm4/gfx12_cmd_builder.h:92
 */
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

/**
 * @brief Set GRBM to broadcast mode for writing to all GPU instances
 *
 * Configures GRBM_GFX_INDEX register to broadcast writes to all Shader Engines,
 * Shader Arrays, and instances simultaneously. Essential before configuring
 * performance counters globally.
 *
 * @param buffer PM4 buffer to append packet to
 * @param grbm_gfx_index_reg GRBM_GFX_INDEX register offset
 * @return 0 on success, negative on error
 *
 * @see pm4_set_grbm_index()
 */
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

/**
 * @brief Set specific GRBM index for targeted register writes
 *
 * Configures GRBM_GFX_INDEX to target a specific SE/SA/WGP location.
 * Used when reading counters from specific hardware instances.
 *
 * @param buffer PM4 buffer to append packet to
 * @param grbm_gfx_index_reg GRBM_GFX_INDEX register offset
 * @param wg_index Work group index (shifted left by 2 before programming)
 * @param sa_index Shader Array index
 * @param se_index Shader Engine index
 * @return 0 on success, negative on error
 *
 * @note wg_index is shifted left by 2 to match hardware expectations
 * @see pm4_grbm_broadcast()
 */
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

/**
 * @brief Enable or disable performance monitoring
 *
 * Writes to CP_PERFMON_CNTL register to control the performance monitoring
 * state machine (disable/enable/stop) and sample bit.
 *
 * @param buffer PM4 buffer to append packet to
 * @param cp_perfmon_cntl_reg CP_PERFMON_CNTL register offset
 * @param control_value Control value (state + sample bit)
 * @return 0 on success, negative on error
 *
 * @see generate_perfmon_enable() in packet_generation.c
 */
int pm4_perfcount_enable(pm4_buffer_t *buffer,
                          uint32_t cp_perfmon_cntl_reg,
                          uint32_t control_value)
{
    return pm4_append_set_uconfig_reg(buffer, cp_perfmon_cntl_reg, control_value);
}

/**
 * @brief Trigger CS partial flush for synchronization
 *
 * Appends an EVENT_WRITE packet that triggers a CS_PARTIAL_FLUSH event,
 * ensuring all compute shader work completes before subsequent commands.
 *
 * @param buffer PM4 buffer to append packet to
 * @return 0 on success, negative on error
 *
 * @see pm4_append_event_write()
 */
int pm4_cs_partial_flush(pm4_buffer_t *buffer)
{
    /* Trigger CS partial flush event */
    return pm4_append_event_write(buffer, VGT_EVENT_TYPE_CS_PARTIAL_FLUSH, 4);
}

/**
 * @brief Calculate cache coherency parameters for ACQUIRE_MEM packet
 *
 * Converts a memory address and size into the coher_base/coher_size format
 * required by ACQUIRE_MEM packets. The hardware uses 256-byte granularity for
 * cache line addresses. This calculation ensures proper alignment and coverage.
 *
 * Algorithm matches the Rust rocprofiler-oop cache_cohere implementation exactly.
 *
 * @param addr Base memory address to flush (any alignment)
 * @param size Size of memory range in bytes
 * @param coher_size_lo Output: coherency size low 32 bits (in 256-byte units)
 * @param coher_size_hi Output: coherency size high 8 bits
 * @param coher_base_lo Output: coherency base address low 32 bits (bits 8-39 of addr)
 * @param coher_base_hi Output: coherency base address high 24 bits (bits 40-63 of addr)
 *
 * @note Addresses are converted to 256-byte granularity by shifting right 8 bits
 * @see pm4_append_acquire_mem()
 */
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

/**
 * @brief Dump a PM4 packet to console for debugging
 *
 * Prints all DWORDs of a PM4 packet in hexadecimal format for inspection.
 * Uses printk() in kernel mode and printf() in userspace.
 *
 * @param packet Pointer to packet data
 * @param size_dwords Size of packet in DWORDs
 */
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

/**
 * @brief Convert PM4 opcode to string name for debugging
 *
 * @param opcode PM4 opcode value (from packet header bits 8-15)
 * @return String name of opcode, or "UNKNOWN" if not recognized
 */
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

/**
 * @brief Validate a PM4 packet structure
 *
 * Checks that the packet header is valid (Type 3) and that the size field
 * matches the actual packet size for known opcodes.
 *
 * @param packet Pointer to packet data
 * @param size_dwords Size of packet in DWORDs
 * @return 0 if valid, -1 if invalid or NULL
 */
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

/**
 * @brief Parse PM4 packet data into a pm4_op_t structure
 *
 * Decodes a raw PM4 packet buffer into a structured pm4_op_t union based on
 * the opcode. Supports all major PM4 packet types used for performance monitoring.
 *
 * @param data Pointer to PM4 packet data
 * @param size_dwords Size of packet in DWORDs
 * @return Parsed pm4_op_t structure (type set to PM4_OP_INVALID on error)
 *
 * @note Returns PM4_OP_INVALID if data is NULL, size is 0, or opcode is unknown
 */
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

/**
 * @brief Serialize a pm4_op_t structure to buffer
 *
 * Converts a pm4_op_t structure back into raw PM4 packet format and writes
 * it to the provided buffer. Used for reconstructing packets from structured data.
 *
 * @param op Pointer to PM4 operation structure to serialize
 * @param buffer Output buffer for packet data
 * @param buffer_size Size of output buffer in DWORDs
 * @return Number of DWORDs written, or -1 on error (NULL pointers or buffer too small)
 */
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

/**
 * @brief Get size of a PM4 operation in DWORDs
 *
 * Returns the size in DWORDs for the given PM4 operation type.
 * Used to determine buffer space needed for serialization.
 *
 * @param op Pointer to PM4 operation structure
 * @return Size in DWORDs (fixed size per type), or 0 if op is NULL
 */
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

/**
 * @brief Convert PM4 operation type enum to string name
 *
 * @param type PM4 operation type from pm4_op_type_t enum
 * @return String name of operation type (e.g., "SET_UCONFIG_REG", "COPY_DATA")
 */
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