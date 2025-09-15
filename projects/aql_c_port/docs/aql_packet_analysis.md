# AQL Packet Population Analysis

## Overview

This document analyzes the AQL (Architected Queuing Language) packet population logic from the C++ AQLProfile library, focusing on how PM4 commands are wrapped into AQL packets for submission to AMD GPUs.

## AQL Packet Structure

The core AQL packet structure for PM4 indirect buffer commands is defined in `amd_aql_pm4_ib_packet.h`:

```cpp
typedef struct {
    uint16_t header;                           // AQL packet header
    uint16_t pm4_ib_format;                   // Format identifier (=1)
    uint32_t pm4_ib_command[4];               // 4-dword PM4 indirect buffer command
    uint32_t dw_count_remain;                 // Remaining dword count (=10)
    uint32_t reserved[8];                     // Reserved fields (zeroed)
    hsa_signal_t completion_signal;           // HSA completion signal
} amd_aql_pm4_ib_packet_t;
```

### Field Details:

1. **header** (16-bit): AQL packet header containing packet type and format information
2. **pm4_ib_format** (16-bit): Always set to `AMD_AQL_PM4_IB_FORMAT` (1) - identifies this as PM4 IB format
3. **pm4_ib_command[4]** (4 × 32-bit): Contains the actual PM4 indirect buffer command
4. **dw_count_remain** (32-bit): Always `AMD_AQL_PM4_IB_DW_COUNT_REMAIN` (10) - indicates remaining packet size
5. **reserved[8]** (8 × 32-bit): Reserved space, always zeroed
6. **completion_signal** (64-bit): HSA signal for packet completion notification

## Packet Population Process

### Function 1: `PopulateAql(const uint32_t* ib_packet, packet_t* aql_packet)`

This is the core population function that fills an AQL packet with a 4-dword PM4 indirect buffer command:

```cpp
void PopulateAql(const uint32_t* ib_packet, packet_t* aql_packet) {
    amd_aql_pm4_ib_packet_t* aql_pm4_ib = reinterpret_cast<amd_aql_pm4_ib_packet_t*>(aql_packet);

    // Set format identifier
    aql_pm4_ib->pm4_ib_format = AMD_AQL_PM4_IB_FORMAT;

    // Copy 4-dword PM4 command
    aql_pm4_ib->pm4_ib_command[0] = ib_packet[0];
    aql_pm4_ib->pm4_ib_command[1] = ib_packet[1];
    aql_pm4_ib->pm4_ib_command[2] = ib_packet[2];
    aql_pm4_ib->pm4_ib_command[3] = ib_packet[3];

    // Set remaining dword count
    aql_pm4_ib->dw_count_remain = AMD_AQL_PM4_IB_DW_COUNT_REMAIN;

    // Zero reserved fields
    for (unsigned i = 0; i < AMD_AQL_PM4_IB_RESERVED_COUNT; ++i) {
        aql_pm4_ib->reserved[i] = 0;
    }
}
```

**Key Points:**
- Header and completion signal are NOT set by this function (caller responsibility)
- PM4 indirect buffer command is exactly 4 dwords
- Reserved fields are explicitly zeroed for security/consistency

### Function 2: `PopulateAql(const void* cmd_buffer, uint32_t cmd_size, CmdBuilder* cmd_writer, packet_t* aql_packet)`

This higher-level function first builds a PM4 indirect buffer command, then populates the AQL packet:

```cpp
void PopulateAql(const void* cmd_buffer, uint32_t cmd_size, pm4_builder::CmdBuilder* cmd_writer,
                 packet_t* aql_packet) {
    pm4_builder::CmdBuffer ib_buffer;
    cmd_writer->BuildIndirectBufferCmd(&ib_buffer, cmd_buffer, (size_t)cmd_size);
    PopulateAql((const uint32_t*)ib_buffer.Data(), aql_packet);
}
```

**Process Flow:**
1. Create temporary command buffer
2. Build PM4 indirect buffer command pointing to actual command buffer
3. Extract the IB command and populate AQL packet

## Debug Tracing

The C++ implementation includes debug tracing that dumps packet contents:

```cpp
#if defined(DEBUG_TRACE)
const uint32_t* dwords = (uint32_t*)aql_packet;
const uint32_t dword_count = sizeof(*aql_packet) / sizeof(uint32_t);
// Print all dwords in hex format
for (unsigned idx = 0; idx < dword_count; idx++) {
    std::clog << " " << std::hex << std::setw(8) << std::setfill('0') << dwords[idx];
}
#endif
```

## C Port Design

### Data Structure Definition:
```c
#define AQL_PM4_IB_FORMAT 1
#define AQL_PM4_IB_DW_COUNT_REMAIN 10
#define AQL_PM4_IB_RESERVED_COUNT 8

typedef struct {
    uint16_t header;
    uint16_t pm4_ib_format;
    uint32_t pm4_ib_command[4];
    uint32_t dw_count_remain;
    uint32_t reserved[AQL_PM4_IB_RESERVED_COUNT];
    uint64_t completion_signal;  // Simplified from hsa_signal_t for kernel use
} aql_pm4_ib_packet_t;
```

### Core Population Function:
```c
/**
 * aql_populate_packet - Populate AQL packet with PM4 indirect buffer command
 * @ib_packet: 4-dword PM4 indirect buffer command
 * @aql_packet: AQL packet to populate (header/signal not modified)
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_populate_packet(const uint32_t* ib_packet, aql_pm4_ib_packet_t* aql_packet) {
    if (!ib_packet || !aql_packet) {
        return -EINVAL;
    }

    // Set format identifier
    aql_packet->pm4_ib_format = AQL_PM4_IB_FORMAT;

    // Copy PM4 command (always 4 dwords for indirect buffer)
    aql_packet->pm4_ib_command[0] = ib_packet[0];
    aql_packet->pm4_ib_command[1] = ib_packet[1];
    aql_packet->pm4_ib_command[2] = ib_packet[2];
    aql_packet->pm4_ib_command[3] = ib_packet[3];

    // Set remaining dword count
    aql_packet->dw_count_remain = AQL_PM4_IB_DW_COUNT_REMAIN;

    // Zero reserved fields
    memset(aql_packet->reserved, 0, sizeof(aql_packet->reserved));

    return 0;
}
```

### Extended Population Function:
```c
/**
 * aql_populate_packet_from_buffer - Build IB command and populate AQL packet
 * @cmd_buffer: Command buffer containing PM4 commands
 * @cmd_size: Size of command buffer in bytes
 * @arch_ops: Architecture-specific operations
 * @aql_packet: AQL packet to populate
 *
 * Returns: 0 on success, negative error code on failure
 */
int aql_populate_packet_from_buffer(const void* cmd_buffer, uint32_t cmd_size,
                                   const aql_arch_ops_t* arch_ops,
                                   aql_pm4_ib_packet_t* aql_packet) {
    aql_cmd_buffer_t ib_buffer;
    uint32_t ib_cmd[4];
    int ret;

    if (!cmd_buffer || !cmd_size || !arch_ops || !aql_packet) {
        return -EINVAL;
    }

    // Initialize temporary buffer for IB command
    ret = aql_cmd_buffer_init(&ib_buffer, ib_cmd, sizeof(ib_cmd));
    if (ret) {
        return ret;
    }

    // Build PM4 indirect buffer command
    arch_ops->build_indirect_buffer(&ib_buffer, cmd_buffer, cmd_size);

    // Populate AQL packet with IB command
    return aql_populate_packet(ib_cmd, aql_packet);
}
```

### Debug Support:
```c
#ifdef AQL_DEBUG_TRACE
static void aql_debug_print_packet(const aql_pm4_ib_packet_t* packet) {
    const uint32_t* dwords = (const uint32_t*)packet;
    size_t dword_count = sizeof(*packet) / sizeof(uint32_t);

    printk(KERN_DEBUG "AQL packet (%zu dwords):", dword_count);
    for (size_t i = 0; i < dword_count; i++) {
        printk(KERN_CONT " %08x", dwords[i]);
    }
    printk(KERN_CONT "\n");
}
#else
#define aql_debug_print_packet(packet) do {} while(0)
#endif
```

## Key Insights for Kernel Implementation

1. **Fixed Structure**: AQL packets have a fixed 64-byte structure, simplifying kernel memory management
2. **IB Command Format**: PM4 indirect buffer commands are always exactly 4 dwords
3. **No Dynamic Allocation**: Packet population requires no dynamic memory allocation
4. **Header Management**: AQL header and completion signal are managed separately from packet population
5. **Security**: Reserved fields must be zeroed to prevent information leakage
6. **Validation**: Input validation is critical for kernel security

## Integration Points

1. **Memory Management**: AQL packets can be pre-allocated in kernel memory pools
2. **Command Building**: Integrates with PM4 command builder functions
3. **Queue Submission**: Populated packets are submitted to GPU command queues
4. **Synchronization**: Completion signals enable asynchronous operation tracking

## Next Steps

1. Design C data structures for counter management
2. Map register programming sequences
3. Implement command buffer management
4. Create architecture function pointer tables