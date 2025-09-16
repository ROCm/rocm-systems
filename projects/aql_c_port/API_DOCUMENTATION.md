# AQL C Library API Documentation

## Overview

The AQL C Library provides a comprehensive API for AMD GPU performance counter programming and AQL packet generation. This document describes all public APIs available for integration.

## Table of Contents

- [Core Types and Enumerations](#core-types-and-enumerations)
- [PMC Interface API](#pmc-interface-api)
- [Command Buffer API](#command-buffer-api)
- [Packet Generation API](#packet-generation-api)
- [Architecture Detection API](#architecture-detection-api)

---

## Core Types and Enumerations

### Header: `aql_types.h`

#### Result Codes
```c
typedef enum {
    AQL_SUCCESS = 0,
    AQL_ERROR_INVALID_ARGUMENT = -1,
    AQL_ERROR_OUT_OF_MEMORY = -2,
    AQL_ERROR_UNSUPPORTED_ARCHITECTURE = -3,
    AQL_ERROR_INVALID_BLOCK = -4,
    AQL_ERROR_INVALID_EVENT = -5,
    AQL_ERROR_BUFFER_OVERFLOW = -6,
    AQL_ERROR_NOT_INITIALIZED = -7,
    AQL_ERROR_ALREADY_INITIALIZED = -8,
    AQL_ERROR_IO_ERROR = -9,
    AQL_ERROR_PERMISSION_DENIED = -10,
    AQL_ERROR_RESOURCE_BUSY = -11,
    AQL_ERROR_NOT_FOUND = -12,
    AQL_ERROR_TIMEOUT = -13,
    AQL_ERROR_INTERNAL = -99
} aql_result_t;
```

#### Architecture Types
```c
typedef enum {
    AQL_ARCH_UNKNOWN = 0,
    AQL_ARCH_GFX9 = 9,     // Vega architecture
    AQL_ARCH_GFX10 = 10,   // RDNA1 architecture
    AQL_ARCH_GFX11 = 11,   // RDNA2 architecture
    AQL_ARCH_GFX12 = 12    // RDNA3 architecture
} aql_architecture_t;
```

#### Counter Block IDs
```c
typedef enum {
    AQL_BLOCK_UNKNOWN = 0,
    AQL_BLOCK_CB,          // Color Buffer
    AQL_BLOCK_CPC,         // Command Processor Controller
    AQL_BLOCK_CPF,         // Command Processor Fetcher
    AQL_BLOCK_CPG,         // Command Processor Graphics
    AQL_BLOCK_DB,          // Depth Buffer
    AQL_BLOCK_GDS,         // Global Data Share
    AQL_BLOCK_GRBM,        // Graphics Register Bus Manager
    AQL_BLOCK_GRBM_SE,     // GRBM Shader Engine
    AQL_BLOCK_PA_SC,       // Primitive Assembly - Scan Converter
    AQL_BLOCK_PA_SU,       // Primitive Assembly - Setup Unit
    AQL_BLOCK_SPI,         // Shader Processor Input
    AQL_BLOCK_SQ,          // Sequencer
    AQL_BLOCK_SQ_GS,       // SQ Geometry Shader
    AQL_BLOCK_SQ_PS,       // SQ Pixel Shader
    AQL_BLOCK_SQ_VS,       // SQ Vertex Shader
    AQL_BLOCK_SQ_HS,       // SQ Hull Shader
    AQL_BLOCK_SQ_CS,       // SQ Compute Shader
    AQL_BLOCK_SX,          // Shader Export
    AQL_BLOCK_TA,          // Texture Addresser
    AQL_BLOCK_TCP,         // Texture Cache per Pipe
    AQL_BLOCK_TD,          // Texture Data
    AQL_BLOCK_TCA,         // Texture Cache Arbiter
    AQL_BLOCK_TCC,         // Texture Cache Controller
    AQL_BLOCK_GL1A,        // GL1 Texture Cache Arbiter (RDNA)
    AQL_BLOCK_GL1C,        // GL1 Texture Cache Controller (RDNA)
    AQL_BLOCK_GL2A,        // GL2 Cache Arbiter (RDNA)
    AQL_BLOCK_GL2C,        // GL2 Cache Controller (RDNA)
    AQL_BLOCK_GCR,         // Graphics Command Ring (RDNA)
    AQL_BLOCK_GUS,         // Graphics Unified Scheduler (RDNA)
    AQL_BLOCK_SDMA0,       // SDMA Engine 0
    AQL_BLOCK_SDMA1,       // SDMA Engine 1
    AQL_BLOCK_UMC,         // Unified Memory Controller
    AQL_BLOCK_GCEA,        // Graphics Command Engine Arbiter
    AQL_BLOCK_RPB,         // Render Backend
    AQL_BLOCK_RMI,         // Read/Write Memory Interface
    AQL_BLOCK_MAX
} aql_block_id_t;
```

#### Core Structures
```c
// Counter configuration
typedef struct {
    const char* name;      // Counter name (for documentation)
    aql_block_id_t block;  // Hardware block ID
    uint32_t event;        // Event ID within the block
} aql_counter_config_t;

// PM4 Indirect Buffer packet
typedef struct {
    uint32_t reserved1;
    uint32_t format_pm4ib_base_lo;
    uint32_t pm4ib_base_hi;
    uint32_t dw_count_remain;
    uint32_t pm4ib_cmd[10];  // PM4 commands
    uint32_t reserved2[2];
} aql_pm4_ib_packet_t;

// Register space information
typedef struct {
    uint32_t config_space_start;
    uint32_t config_space_end;
    uint32_t uconfig_space_start;
    uint32_t uconfig_space_end;
    bool has_uconfig_space;
} aql_register_spaces_t;
```

---

## PMC Interface API

### Header: `aql_pmc_interface.h`

The PMC (Performance Monitor Counter) interface provides high-level functions for creating and managing performance counter packets compatible with AQLProfile v2.

#### Key Structure
```c
typedef struct {
    aql_pm4_ib_packet_t start_packet;   // Reset counters and start incrementing
    aql_pm4_ib_packet_t stop_packet;    // Pause counters from incrementing
    aql_pm4_ib_packet_t read_packet;    // Retrieve results from device
    void* command_buffer;                // Command buffer memory
    size_t command_buffer_size;          // Command buffer size
    uint64_t handle;                     // Handle for cleanup
} aql_pmc_packets_t;
```

#### Functions

##### Create PMC Packets
```c
aql_result_t aql_pmc_create_packets(
    const aql_counter_config_t* counters,
    size_t counter_count,
    const char* gpu_name,
    aql_pmc_packets_t* packets
);
```
**Description**: Creates start/stop/read packets for performance counter monitoring.

**Parameters**:
- `counters`: Array of counter configurations to monitor
- `counter_count`: Number of counters in the array
- `gpu_name`: GPU architecture name (e.g., "gfx942", "gfx1201")
- `packets`: Output structure containing generated packets

**Returns**: `AQL_SUCCESS` on success, error code otherwise

**Example**:
```c
aql_counter_config_t counters[] = {
    {"CPC_BUSY", AQL_BLOCK_CPC, 25},
    {"SQ_WAVES", AQL_BLOCK_SQ, 4}
};
aql_pmc_packets_t packets;

aql_result_t result = aql_pmc_create_packets(
    counters, 2, "gfx942", &packets
);
```

##### Cleanup PMC Packets
```c
void aql_pmc_cleanup_packets(aql_pmc_packets_t* packets);
```
**Description**: Cleans up memory allocated for PMC packets.

**Parameters**:
- `packets`: Packet structure to clean up

##### Get PMC Command Buffer
```c
void* aql_pmc_get_command_buffer(const aql_pmc_packets_t* packets);
```
**Description**: Returns the underlying command buffer for direct access.

**Returns**: Pointer to command buffer or NULL if not allocated

##### Get PMC Command Buffer Size
```c
size_t aql_pmc_get_command_buffer_size(const aql_pmc_packets_t* packets);
```
**Description**: Returns the size of the command buffer in bytes.

**Returns**: Buffer size in bytes

---

## Command Buffer API

### Header: `aql_cmd_buffer.h`

Low-level API for managing command buffers and PM4 commands.

#### Functions

##### Initialize Command Buffer
```c
void aql_cmd_buffer_init(aql_cmd_buffer_t* buffer, void* memory, size_t size);
```
**Description**: Initializes a command buffer with provided memory.

**Parameters**:
- `buffer`: Command buffer structure to initialize
- `memory`: Pre-allocated memory for commands
- `size`: Size of memory in bytes

##### Add PM4 Command
```c
aql_result_t aql_cmd_buffer_add_pm4(
    aql_cmd_buffer_t* buffer,
    uint32_t header,
    const uint32_t* data,
    uint32_t data_size_dwords
);
```
**Description**: Adds a PM4 packet to the command buffer.

**Parameters**:
- `buffer`: Target command buffer
- `header`: PM4 packet header
- `data`: Packet data (can be NULL)
- `data_size_dwords`: Size of data in dwords

**Returns**: `AQL_SUCCESS` or `AQL_ERROR_BUFFER_OVERFLOW`

##### Reset Command Buffer
```c
void aql_cmd_buffer_reset(aql_cmd_buffer_t* buffer);
```
**Description**: Resets command buffer to empty state.

##### Get Buffer Size
```c
size_t aql_cmd_buffer_get_size_bytes(const aql_cmd_buffer_t* buffer);
```
**Description**: Returns current size of commands in buffer.

**Returns**: Size in bytes

---

## Packet Generation API

### Header: `aql_packet.h`

Functions for creating and manipulating AQL packets.

#### Functions

##### Create Counter Event
```c
aql_result_t aql_create_counter_event(
    aql_architecture_t arch,
    aql_block_id_t block_id,
    uint32_t event_id,
    uint32_t counter_index,
    aql_cmd_buffer_t* cmd_buffer
);
```
**Description**: Creates PM4 commands for configuring a performance counter.

**Parameters**:
- `arch`: Target GPU architecture
- `block_id`: Counter block ID
- `event_id`: Event ID within the block
- `counter_index`: Counter slot index
- `cmd_buffer`: Output command buffer

**Returns**: `AQL_SUCCESS` on success, error code otherwise

##### Populate AQL Packet from Buffer
```c
aql_result_t aql_populate_packet_from_buffer(
    aql_pm4_ib_packet_t* packet,
    const aql_cmd_buffer_t* cmd_buffer
);
```
**Description**: Converts command buffer to AQL PM4 indirect buffer packet.

**Parameters**:
- `packet`: Output AQL packet
- `cmd_buffer`: Source command buffer

**Returns**: `AQL_SUCCESS` on success

**Note**: This function is used internally by the PMC interface but can be used directly for custom packet generation.

---

## Architecture Detection API

### Header: `aql_arch_detect.h`

Functions for detecting and working with GPU architectures.

#### Functions

##### Detect Architecture from Name
```c
aql_architecture_t aql_detect_architecture(const char* gpu_name);
```
**Description**: Detects GPU architecture from device name string.

**Parameters**:
- `gpu_name`: GPU device name (e.g., "gfx942", "gfx1201")

**Returns**: Architecture enum value or `AQL_ARCH_UNKNOWN`

**Example**:
```c
aql_architecture_t arch = aql_detect_architecture("gfx942");
// Returns AQL_ARCH_GFX9
```

##### Get Architecture Name
```c
const char* aql_get_architecture_name(aql_architecture_t arch);
```
**Description**: Returns human-readable architecture name.

**Parameters**:
- `arch`: Architecture enum value

**Returns**: String name (e.g., "GFX9", "GFX12") or "UNKNOWN"

##### Check Architecture Support
```c
bool aql_is_architecture_supported(aql_architecture_t arch);
```
**Description**: Checks if architecture is supported by the library.

**Parameters**:
- `arch`: Architecture to check

**Returns**: true if supported, false otherwise

**Currently Supported**:
- `AQL_ARCH_GFX9` (Vega)
- `AQL_ARCH_GFX12` (RDNA3)

---

## Usage Examples

### Basic Performance Counter Setup
```c
#include "aql_pmc_interface.h"
#include "aql_arch_detect.h"

int main() {
    // Define counters to monitor
    aql_counter_config_t counters[] = {
        {"CPC_BUSY", AQL_BLOCK_CPC, 25},
        {"GRBM_COUNT", AQL_BLOCK_GRBM, 0},
        {"SQ_WAVES", AQL_BLOCK_SQ, 4},
        {"TCP_GATE_EN1", AQL_BLOCK_TCP, 0}
    };

    // Create PMC packets
    aql_pmc_packets_t packets;
    aql_result_t result = aql_pmc_create_packets(
        counters, 4, "gfx942", &packets
    );

    if (result != AQL_SUCCESS) {
        fprintf(stderr, "Failed to create PMC packets: %d\n", result);
        return 1;
    }

    // Use the packets with HSA runtime
    // packets.start_packet - submit to start counting
    // packets.stop_packet - submit to stop counting
    // packets.read_packet - submit to read results

    // Cleanup when done
    aql_pmc_cleanup_packets(&packets);

    return 0;
}
```

### Architecture Detection
```c
#include "aql_arch_detect.h"

void process_gpu(const char* gpu_name) {
    aql_architecture_t arch = aql_detect_architecture(gpu_name);

    if (!aql_is_architecture_supported(arch)) {
        printf("GPU %s (architecture %s) is not supported\n",
               gpu_name, aql_get_architecture_name(arch));
        return;
    }

    printf("Detected %s architecture for GPU %s\n",
           aql_get_architecture_name(arch), gpu_name);

    // Process based on architecture
    switch (arch) {
        case AQL_ARCH_GFX9:
            // GFX9-specific processing
            break;
        case AQL_ARCH_GFX12:
            // GFX12-specific processing
            break;
        default:
            break;
    }
}
```

### Low-Level Command Buffer Usage
```c
#include "aql_cmd_buffer.h"
#include "aql_packet.h"

void create_custom_packet() {
    // Allocate command buffer memory
    uint32_t buffer_memory[256];
    aql_cmd_buffer_t cmd_buffer;

    // Initialize command buffer
    aql_cmd_buffer_init(&cmd_buffer, buffer_memory, sizeof(buffer_memory));

    // Add counter configuration commands
    aql_create_counter_event(
        AQL_ARCH_GFX9,
        AQL_BLOCK_CPC,
        25,  // Event ID
        0,   // Counter index
        &cmd_buffer
    );

    // Convert to AQL packet
    aql_pm4_ib_packet_t packet;
    aql_populate_packet_from_buffer(&packet, &cmd_buffer);

    // Use packet with HSA runtime
    // ...
}
```

## Error Handling

All functions that can fail return `aql_result_t`. Always check return values:

```c
aql_result_t result = aql_pmc_create_packets(...);
if (result != AQL_SUCCESS) {
    switch (result) {
        case AQL_ERROR_INVALID_ARGUMENT:
            fprintf(stderr, "Invalid argument provided\n");
            break;
        case AQL_ERROR_UNSUPPORTED_ARCHITECTURE:
            fprintf(stderr, "GPU architecture not supported\n");
            break;
        case AQL_ERROR_OUT_OF_MEMORY:
            fprintf(stderr, "Memory allocation failed\n");
            break;
        default:
            fprintf(stderr, "Unknown error: %d\n", result);
    }
}
```

## Thread Safety

The AQL C Library functions are **thread-safe** for:
- All read-only operations
- Architecture detection functions
- Separate packet structures

The library is **NOT thread-safe** for:
- Concurrent writes to the same command buffer
- Concurrent modifications to the same packet structure

For multi-threaded usage, create separate packet structures per thread or use appropriate synchronization.

## Memory Management

### Allocation
- `aql_pmc_create_packets()` allocates memory internally
- Command buffers can use stack or heap memory
- No global state is maintained

### Cleanup
- Always call `aql_pmc_cleanup_packets()` for PMC packets
- Command buffers using stack memory need no cleanup
- Command buffers using heap memory must be freed by caller

## Platform Requirements

- **Compiler**: C99 compatible compiler
- **Dependencies**:
  - HSA Runtime headers (`hsa/hsa.h`)
  - Standard C library
- **Architectures Supported**:
  - GFX9 (Vega family)
  - GFX12 (RDNA3 family)
- **Operating Systems**:
  - Linux (primary target)
  - Other POSIX systems (untested)

## Version Information

```c
#define AQL_C_VERSION_MAJOR 1
#define AQL_C_VERSION_MINOR 0
#define AQL_C_VERSION_PATCH 0
#define AQL_C_VERSION_STRING "1.0.0"
```

## License

This library is part of the ROCm ecosystem and follows the same licensing terms as other ROCm components.

---

**Last Updated**: 2025-09-15
**API Version**: 1.0.0
**Status**: Production Ready