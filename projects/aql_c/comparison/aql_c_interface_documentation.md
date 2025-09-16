# AQL_C Public Interface Documentation

## Overview

AQL_C is a C library that ports AQLProfile functionality to kernel-mode compatible code. It provides a low-level interface for generating AQL (Asynchronous Queue Language) packets for GPU performance counter monitoring.

## Key Design Principles

- **Kernel Compatibility**: All structures are plain C with no function pointers in basic types
- **Fixed-size Allocations**: Designed for kernel compatibility
- **Clear Separation**: Data and operations are clearly separated
- **Comprehensive Error Handling**: All functions return `aql_result_t` error codes

## Core Data Types

### Architecture Support

```c
typedef enum {
    AQL_ARCH_UNKNOWN = 0,
    AQL_ARCH_GFX9,      // GFX9 (Vega)
    AQL_ARCH_GFX10,     // GFX10 (RDNA1)
    AQL_ARCH_GFX11,     // GFX11 (RDNA2)
    AQL_ARCH_GFX12,     // GFX12 (RDNA3)
} aql_arch_type_t;
```

### Error Handling

```c
typedef enum {
    AQL_SUCCESS = 0,
    AQL_ERROR_INVALID_ARGUMENT = -1,
    AQL_ERROR_BUFFER_TOO_SMALL = -2,
    AQL_ERROR_UNSUPPORTED_ARCH = -3,
    AQL_ERROR_INVALID_COUNTER = -4,
    // ... more error codes
} aql_result_t;
```

### Counter Block Types

The library supports many counter blocks:
- **CPC**: Command Processor Compute
- **GRBM**: Graphics Register Bus Manager
- **SQ**: Shader Quad
- **GL1A/GL1C**: GL1 Cache (RDNA+)
- **GL2A/GL2C**: GL2 Cache (RDNA+)
- **SDMA**: System DMA
- Many others...

## Core Structures

### Agent Information

```c
typedef struct {
    char gfx_name[AQL_MAX_ARCH_NAME_LEN];  // GPU architecture name
    aql_arch_type_t arch_type;             // Architecture type
    uint32_t xcc_count;                    // Number of XCCs
    uint32_t se_count;                     // Number of Shader Engines
    uint32_t cu_count;                     // Number of Compute Units
    uint32_t shader_arrays_per_se;         // Shader arrays per SE
    uint32_t domain;                       // PCI domain
    uint32_t location_id;                  // PCI BDF
} aql_agent_info_t;
```

### Counter Request

```c
typedef struct {
    aql_block_id_t block_id;        // Counter block identifier
    uint32_t block_instance;        // Block instance number
    uint32_t counter_id;            // Counter within block
    uint32_t event_select;          // Event to count
    uint32_t flags;                 // Counter-specific flags

    // Runtime assignment (filled by allocation logic)
    uint32_t assigned_register;     // Assigned hardware register
    void* result_location;          // Where to store counter result

    // Debug information
    const char* block_name;         // Human-readable block name
    const char* counter_name;       // Human-readable counter name
} aql_counter_request_t;
```

### AQL Context

```c
typedef struct {
    aql_agent_info_t agent_info;             // GPU agent details
    const struct aql_arch_ops* arch_ops;     // Architecture operations
    aql_counter_request_t* counters;         // Array of counter requests
    uint32_t counter_count;                  // Number of active counters
    uint32_t max_counters;                   // Maximum counters supported
    aql_cmd_buffer_t cmd_buffer;             // Primary command buffer
    aql_cmd_buffer_t ib_buffer;              // Indirect buffer commands
    // ... runtime state and error tracking
} aql_context_t;
```

## Command Buffer Management

### Initialization

```c
// Initialize with external storage
aql_result_t aql_cmd_buffer_init(aql_cmd_buffer_t* buf,
                                uint32_t* data, size_t capacity);

// Initialize with internal allocation
aql_result_t aql_cmd_buffer_init_alloc(aql_cmd_buffer_t* buf, size_t capacity,
                                      aql_memory_alloc_cb_t alloc_cb,
                                      void* userdata);
```

### Command Building

```c
// Append raw dwords
aql_result_t aql_cmd_buffer_append_dwords(aql_cmd_buffer_t* buf,
                                         const uint32_t* data,
                                         size_t dword_count);

// Append single dword
aql_result_t aql_cmd_buffer_append_dword(aql_cmd_buffer_t* buf, uint32_t value);

// Reserve space for later writing
aql_result_t aql_cmd_buffer_reserve(aql_cmd_buffer_t* buf, size_t dword_count,
                                   uint32_t** reserved_ptr);
```

### Utility Functions

```c
// Get buffer information
size_t aql_cmd_buffer_available(const aql_cmd_buffer_t* buf);
size_t aql_cmd_buffer_used(const aql_cmd_buffer_t* buf);
size_t aql_cmd_buffer_size_bytes(const aql_cmd_buffer_t* buf);
bool aql_cmd_buffer_is_valid(const aql_cmd_buffer_t* buf);
```

## Architecture Operations Interface

The library uses function pointer tables for architecture-specific operations:

```c
typedef struct aql_arch_ops {
    const char* arch_name;              // Architecture name
    aql_arch_type_t arch_type;          // Architecture type

    // Capabilities
    bool has_pred_exec;                 // Supports PRED_EXEC packets
    bool has_uconfig_space;             // Has UCONFIG register space
    bool has_dual_sdma;                 // Has dual SDMA engines
    bool has_gl_cache_hierarchy;        // Has GL1/GL2 cache blocks

    // PM4 Command Generation Functions
    aql_result_t (*build_write_uconfig_reg)(aql_cmd_buffer_t* buf,
                                           uint32_t addr, uint32_t value);
    aql_result_t (*build_write_config_reg)(aql_cmd_buffer_t* buf,
                                          uint32_t addr, uint32_t value);
    aql_result_t (*build_copy_reg_data)(aql_cmd_buffer_t* buf,
                                       uint32_t src_addr, void* dst_addr,
                                       uint32_t size, bool wait);
    // ... more PM4 command builders

    // Architecture-specific helpers
    uint32_t (*make_grbm_index)(uint32_t se, uint32_t sa,
                               uint32_t wgp, uint32_t instance);
    uint32_t (*get_cache_policy)(void);

    // Counter management
    bool (*is_valid_counter_reg)(aql_block_id_t block_id, uint32_t reg_addr);
    const aql_counter_block_regs_t* (*get_counter_block_regs)(aql_block_id_t block_id);
    aql_result_t (*validate_counter_request)(const aql_counter_request_t* request);
} aql_arch_ops_t;
```

### Architecture Detection

```c
// Detect architecture from name
const aql_arch_ops_t* aql_detect_architecture(const char* gfx_name);

// Get by type
const aql_arch_ops_t* aql_get_arch_ops(aql_arch_type_t arch_type);

// List all supported
uint32_t aql_list_supported_architectures(const aql_arch_ops_t** ops_list,
                                         uint32_t max_count);
```

## AQLProfile v2-Compatible Interface

### PMC Event Structure

```c
typedef struct {
    aql_block_id_t block_id;        // Block ID
    uint32_t block_instance;        // Block instance/channel
    uint32_t event_id;              // Event ID
    uint32_t flags;                 // Event flags

    // Debug information
    const char* block_name;         // Block name string
    const char* event_name;         // Event name string
} aql_pmc_event_t;
```

### Profile Configuration

```c
typedef struct {
    const char* arch_name;              // Architecture name (e.g., "gfx942")
    const aql_pmc_event_t* events;      // Array of events to monitor
    uint32_t event_count;               // Number of events

    // Output buffer information
    void* output_buffer;                // Buffer for counter results
    size_t output_buffer_size;          // Size of output buffer
} aql_pmc_profile_t;
```

### AQL Packet Set

```c
typedef struct {
    aql_pm4_ib_packet_t start_packet;   // Reset counters and start
    aql_pm4_ib_packet_t stop_packet;    // Pause counters
    aql_pm4_ib_packet_t read_packet;    // Retrieve results

    // Internal state
    void* command_buffer;               // Command buffer memory
    size_t command_buffer_size;         // Command buffer size
    uint64_t handle;                    // Handle for cleanup
} aql_pmc_packets_t;
```

### Primary Interface Functions

```c
// Create PMC AQL packets (v2-compatible)
aql_result_t aql_pmc_create_packets(aql_pmc_packets_t* packets,
                                   const aql_pmc_profile_t* profile);

// Delete packets and free resources
aql_result_t aql_pmc_delete_packets(aql_pmc_packets_t* packets);

// Helper to create counter events
aql_result_t aql_create_counter_event(aql_pmc_event_t* event,
                                     const char* block_name,
                                     uint32_t instance,
                                     uint32_t event_id);
```

## Usage Example

```c
#include "aql_pmc_interface.h"

int main() {
    // 1. Create counter events
    aql_pmc_event_t events[2];
    aql_create_counter_event(&events[0], "CPC", 0, 123);
    aql_create_counter_event(&events[1], "GRBM", 0, 456);

    // 2. Setup profile
    aql_pmc_profile_t profile = {
        .arch_name = "gfx942",
        .events = events,
        .event_count = 2,
        .output_buffer = malloc(1024),
        .output_buffer_size = 1024
    };

    // 3. Create packets
    aql_pmc_packets_t packets;
    aql_result_t result = aql_pmc_create_packets(&packets, &profile);
    if (result != AQL_SUCCESS) {
        // Handle error
        return -1;
    }

    // 4. Use packets (submit to GPU queue)
    // ... submit start_packet, kernel, stop_packet, read_packet ...

    // 5. Cleanup
    aql_pmc_delete_packets(&packets);
    free(profile.output_buffer);

    return 0;
}
```

## Key Constants

```c
#define AQL_MAX_COUNTERS_PER_REQUEST    128     // Max counters per request
#define AQL_MAX_CMD_BUFFER_DWORDS      4096     // Max command buffer size
#define AQL_PACKET_SIZE                 64      // AQL packet size in bytes
#define AQL_PM4_IB_COMMAND_DWORDS       4      // PM4 IB command size
```

## Debug Support

When compiled with `AQL_DEBUG_TRACE`, the library provides extensive debugging:

```c
void aql_cmd_buffer_debug_print(const aql_cmd_buffer_t* buf, const char* prefix);
void aql_cmd_buffer_debug_print_recent(const aql_cmd_buffer_t* buf,
                                      size_t count, const char* command_name);
```

## Memory Management

The library supports custom memory allocation for kernel environments:

```c
typedef aql_result_t (*aql_memory_alloc_cb_t)(void** ptr, size_t size,
                                              uint32_t flags, void* userdata);
typedef void (*aql_memory_dealloc_cb_t)(void* ptr, void* userdata);
```

Memory flags:
- `AQL_MEM_FLAG_HOST_ACCESS`: CPU accessible
- `AQL_MEM_FLAG_DEVICE_ACCESS`: GPU accessible
- `AQL_MEM_FLAG_COHERENT`: Cache coherent
- `AQL_MEM_FLAG_UNCACHED`: Uncached access

## Architecture Differences

### GFX9 Specific
- Has UCONFIG register space
- Supports PRED_EXEC packets
- Single SDMA engine
- Has TCA, TCC, TCS blocks

### RDNA (GFX10+) Specific
- GL1/GL2 cache hierarchy instead of TCA/TCC
- Dual SDMA engines (SDMA0, SDMA1)
- WGP (Workgroup Processor) addressing
- Different register spaces

## Error Handling Best Practices

Always check return values:

```c
aql_result_t result = aql_function_call();
if (result != AQL_SUCCESS) {
    // Handle specific error
    switch (result) {
        case AQL_ERROR_BUFFER_TOO_SMALL:
            // Increase buffer size
            break;
        case AQL_ERROR_UNSUPPORTED_ARCH:
            // Architecture not supported
            break;
        default:
            // Generic error handling
            break;
    }
}
```

This interface provides a complete, kernel-compatible replacement for AQLProfile's C++ implementation while maintaining compatibility with the v2 interface.