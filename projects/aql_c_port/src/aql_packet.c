/**
 * @file aql_packet.c
 * @brief AQL packet population implementation
 *
 * This file implements the AQL packet population logic that was originally
 * in the C++ populate_aql.cpp file. It provides functions to create AQL
 * packets from PM4 commands for submission to AMD GPUs.
 */

#include "aql_types.h"
#include "aql_arch_ops.h"
#include "aql_cmd_buffer.h"

#ifdef __KERNEL__
#include <linux/string.h>
#define AQL_PRINT(fmt, ...) printk(KERN_DEBUG "aql: " fmt, ##__VA_ARGS__)
#else
#include <string.h>
#include <stdio.h>
#define AQL_PRINT(fmt, ...) printf("aql: " fmt, ##__VA_ARGS__)
#endif

/* AQL packet format constants */
#define AQL_PM4_IB_FORMAT                   1
#define AQL_PM4_IB_DW_COUNT_REMAIN         10
#define AQL_PM4_IB_RESERVED_COUNT           8

/**
 * @brief Populate AQL packet with PM4 indirect buffer command
 * @param ib_packet 4-dword PM4 indirect buffer command
 * @param aql_packet AQL packet to populate (header/signal not modified)
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_populate_packet(const uint32_t* ib_packet, aql_pm4_ib_packet_t* aql_packet) {
    if (!ib_packet || !aql_packet) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Set format identifier */
    aql_packet->pm4_ib_format = AQL_PM4_IB_FORMAT;

    /* Copy PM4 command (always 4 dwords for indirect buffer) */
    aql_packet->pm4_ib_command[0] = ib_packet[0];
    aql_packet->pm4_ib_command[1] = ib_packet[1];
    aql_packet->pm4_ib_command[2] = ib_packet[2];
    aql_packet->pm4_ib_command[3] = ib_packet[3];

    /* Set remaining dword count */
    aql_packet->dw_count_remain = AQL_PM4_IB_DW_COUNT_REMAIN;

    /* Zero reserved fields */
    memset(aql_packet->reserved, 0, sizeof(aql_packet->reserved));

#ifdef AQL_DEBUG_TRACE
    aql_debug_print_packet(aql_packet);
#endif

    return AQL_SUCCESS;
}

/**
 * @brief Build IB command and populate AQL packet
 * @param cmd_buffer Command buffer containing PM4 commands
 * @param cmd_size Size of command buffer in bytes
 * @param arch_ops Architecture-specific operations
 * @param aql_packet AQL packet to populate
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_populate_packet_from_buffer(const void* cmd_buffer, uint32_t cmd_size,
                                           const aql_arch_ops_t* arch_ops,
                                           aql_pm4_ib_packet_t* aql_packet) {
    uint32_t ib_cmd[4];
    aql_cmd_buffer_t ib_buffer;
    aql_result_t result;

    if (!cmd_buffer || !cmd_size || !arch_ops || !aql_packet) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Initialize temporary buffer for IB command */
    result = aql_cmd_buffer_init(&ib_buffer, ib_cmd, 4);
    if (result != AQL_SUCCESS) {
        return result;
    }

    /* Build PM4 indirect buffer command */
    result = arch_ops->build_indirect_buffer(&ib_buffer, cmd_buffer, cmd_size);
    if (result != AQL_SUCCESS) {
        return result;
    }

    /* Verify we got exactly 4 dwords */
    if (aql_cmd_buffer_used(&ib_buffer) != 4) {
        return AQL_ERROR_INVALID_STATE;
    }

    /* Populate AQL packet with IB command */
    return aql_populate_packet(ib_cmd, aql_packet);
}

/**
 * @brief Initialize AQL packet header fields
 * @param aql_packet AQL packet to initialize
 * @param packet_type AQL packet type
 * @param barrier Barrier flag
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_init_packet_header(aql_pm4_ib_packet_t* aql_packet,
                                   uint16_t packet_type, bool barrier) {
    if (!aql_packet) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Construct AQL header */
    aql_packet->header = packet_type;
    if (barrier) {
        aql_packet->header |= 0x0100; /* Set barrier bit */
    }

    return AQL_SUCCESS;
}

/**
 * @brief Set completion signal for AQL packet
 * @param aql_packet AQL packet to modify
 * @param signal_handle Signal handle value
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_set_completion_signal(aql_pm4_ib_packet_t* aql_packet,
                                      uint64_t signal_handle) {
    if (!aql_packet) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    aql_packet->completion_signal = signal_handle;
    return AQL_SUCCESS;
}

/**
 * @brief Validate AQL packet structure
 * @param aql_packet AQL packet to validate
 * @return AQL_SUCCESS if valid, error code otherwise
 */
aql_result_t aql_validate_packet(const aql_pm4_ib_packet_t* aql_packet) {
    if (!aql_packet) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Check format field */
    if (aql_packet->pm4_ib_format != AQL_PM4_IB_FORMAT) {
        return AQL_ERROR_INVALID_STATE;
    }

    /* Check dword count */
    if (aql_packet->dw_count_remain != AQL_PM4_IB_DW_COUNT_REMAIN) {
        return AQL_ERROR_INVALID_STATE;
    }

    /* Check that reserved fields are zeroed */
    for (int i = 0; i < AQL_PM4_IB_RESERVED_COUNT; i++) {
        if (aql_packet->reserved[i] != 0) {
            return AQL_ERROR_INVALID_STATE;
        }
    }

    return AQL_SUCCESS;
}

/**
 * @brief Copy AQL packet contents
 * @param src Source AQL packet
 * @param dst Destination AQL packet
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_copy_packet(const aql_pm4_ib_packet_t* src, aql_pm4_ib_packet_t* dst) {
    if (!src || !dst) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    memcpy(dst, src, sizeof(aql_pm4_ib_packet_t));
    return AQL_SUCCESS;
}

/**
 * @brief Get PM4 command from AQL packet
 * @param aql_packet AQL packet
 * @param cmd_buffer Output buffer for PM4 command (must be at least 4 dwords)
 * @return AQL_SUCCESS or error code
 */
aql_result_t aql_extract_pm4_command(const aql_pm4_ib_packet_t* aql_packet,
                                    uint32_t* cmd_buffer) {
    if (!aql_packet || !cmd_buffer) {
        return AQL_ERROR_INVALID_ARGUMENT;
    }

    /* Validate packet first */
    aql_result_t result = aql_validate_packet(aql_packet);
    if (result != AQL_SUCCESS) {
        return result;
    }

    /* Copy PM4 command */
    cmd_buffer[0] = aql_packet->pm4_ib_command[0];
    cmd_buffer[1] = aql_packet->pm4_ib_command[1];
    cmd_buffer[2] = aql_packet->pm4_ib_command[2];
    cmd_buffer[3] = aql_packet->pm4_ib_command[3];

    return AQL_SUCCESS;
}

/*
 * Debug Support Functions
 */

#ifdef AQL_DEBUG_TRACE

void aql_debug_print_packet(const aql_pm4_ib_packet_t* aql_packet) {
    const uint32_t* dwords;
    size_t dword_count;

    if (!aql_packet) return;

    dwords = (const uint32_t*)aql_packet;
    dword_count = sizeof(*aql_packet) / sizeof(uint32_t);

    AQL_PRINT("AQL packet (%zu dwords):", dword_count);
    for (size_t i = 0; i < dword_count; i++) {
        AQL_PRINT(" %08x", dwords[i]);
    }
    AQL_PRINT("\n");

    /* Print detailed breakdown */
    AQL_PRINT("  header=0x%04x, format=%u, dw_remain=%u, signal=0x%016llx\n",
              aql_packet->header, aql_packet->pm4_ib_format,
              aql_packet->dw_count_remain,
              (unsigned long long)aql_packet->completion_signal);

    AQL_PRINT("  pm4_cmd: %08x %08x %08x %08x\n",
              aql_packet->pm4_ib_command[0], aql_packet->pm4_ib_command[1],
              aql_packet->pm4_ib_command[2], aql_packet->pm4_ib_command[3]);
}

void aql_debug_print_pm4_ib_command(const uint32_t* ib_cmd) {
    uint32_t header, opcode, count;

    if (!ib_cmd) return;

    header = ib_cmd[0];
    opcode = header & 0xFF;
    count = (header >> 16) & 0x3FFF;

    AQL_PRINT("PM4 IB command: opcode=0x%02x, count=%u\n", opcode, count);
    AQL_PRINT("  dwords: %08x %08x %08x %08x\n",
              ib_cmd[0], ib_cmd[1], ib_cmd[2], ib_cmd[3]);
}

#endif /* AQL_DEBUG_TRACE */