# AQLProfile v2 Interface Documentation

## Overview

AQLProfile v2 is the second version of AMD's AQL (Asynchronous Queue Language) profiling library. It provides a C interface for creating AQL packets that enable hardware performance counter monitoring on AMD GPUs. This interface is the foundation that both ROCProfiler SDK and aql_c build upon.

## Core Architecture

AQLProfile v2 follows a three-phase workflow:
1. **Configuration**: Register agents and define counter events
2. **Packet Creation**: Generate start/stop/read AQL packets
3. **Data Collection**: Retrieve and process counter results

## Core Data Types

### Handle Types

```c
typedef struct {
    uint64_t handle;
} aqlprofile_handle_t;

typedef struct {
    uint64_t handle;
} aqlprofile_agent_handle_t;
```

### Memory Management

```c
typedef enum {
    AQLPROFILE_MEMORY_HINT_NONE = 0,
    AQLPROFILE_MEMORY_HINT_HOST = 1,
    AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED = 2,
    AQLPROFILE_MEMORY_HINT_DEVICE_COHERENT = 3,
    AQLPROFILE_MEMORY_HINT_DEVICE_NONCOHERENT = 4,
} aqlprofile_memory_hint_t;

typedef union {
    uint32_t raw;
    struct {
        uint32_t device_access : 1;
        uint32_t host_access : 1;
        uint32_t memory_hint : 6;    // aqlprofile_memory_hint_t
        uint32_t _reserved : 24;
    };
} aqlprofile_buffer_desc_flags_t;
```

### Memory Callbacks

```c
// Memory allocation callback
typedef hsa_status_t (*aqlprofile_memory_alloc_callback_t)(
    void** ptr,                         // Output: allocated memory pointer
    uint64_t size,                      // Requested size in bytes
    aqlprofile_buffer_desc_flags_t flags, // Access requirements
    void* userdata);                    // User-provided data

// Memory deallocation callback
typedef void (*aqlprofile_memory_dealloc_callback_t)(
    void* ptr,                          // Memory to free
    void* userdata);                    // User-provided data

// Memory copy callback
typedef hsa_status_t (*aqlprofile_memory_copy_t)(
    void* dst,                          // Destination
    const void* src,                    // Source
    size_t size,                        // Size to copy
    void* userdata);                    // User-provided data
```

## Agent Management

### Agent Information Structures

```c
// Version 0 (legacy)
typedef struct {
    const char* agent_gfxip;            // GPU architecture name
    uint32_t xcc_num;                   // Number of XCCs
    uint32_t se_num;                    // Number of Shader Engines
    uint32_t cu_num;                    // Number of Compute Units
    uint32_t shader_arrays_per_se;      // Shader arrays per SE
} aqlprofile_agent_info_t;

// Version 1 (extended)
typedef struct {
    const char* agent_gfxip;            // GPU architecture name
    uint32_t xcc_num;                   // Number of XCCs
    uint32_t se_num;                    // Number of Shader Engines
    uint32_t cu_num;                    // Number of Compute Units
    uint32_t shader_arrays_per_se;      // Shader arrays per SE
    uint32_t domain;                    // PCI domain
    uint32_t location_id;               // PCI BDF (Bus/Device/Function)
} aqlprofile_agent_info_v1_t;
```

### Agent Registration

```c
// Register agent (legacy interface)
hsa_status_t aqlprofile_register_agent(
    aqlprofile_agent_handle_t* agent_id,        // Output: agent handle
    const aqlprofile_agent_info_t* agent_info); // Agent information

// Register agent with versioning
hsa_status_t aqlprofile_register_agent_info(
    aqlprofile_agent_handle_t* agent_id,        // Output: agent handle
    const void* agent_info,                     // Agent info (versioned)
    aqlprofile_agent_version_t version);        // Version specifier
```

## Performance Counter Interface

### Counter Event Structure

```c
typedef enum {
    AQLPROFILE_ACCUMULATION_NONE = 0,      // No accumulation
    AQLPROFILE_ACCUMULATION_LO_RES,        // Integrate over quad-cycles
    AQLPROFILE_ACCUMULATION_HI_RES,        // Integrate every cycle
} aqlprofile_accumulation_type_t;

typedef union {
    uint32_t raw;
    struct {
        uint32_t accum : 3;                // aqlprofile_accumulation_type_t
        uint32_t _reserved : 29;
    } sq_flags;
} aqlprofile_pmc_event_flags_t;

typedef struct {
    uint32_t block_index;                           // Block channel/instance
    uint32_t event_id;                              // Event ID from XML definitions
    aqlprofile_pmc_event_flags_t flags;             // Special event flags
    hsa_ven_amd_aqlprofile_block_name_t block_name; // Block type
} aqlprofile_pmc_event_t;
```

### Counter Block Types

AQLProfile v2 supports extensive block types defined in `hsa_ven_amd_aqlprofile.h` plus extensions:

```c
typedef enum {
    // Standard blocks (from hsa_ven_amd_aqlprofile.h)
    // HSA_VEN_AMD_AQLPROFILE_BLOCKS_CPC,
    // HSA_VEN_AMD_AQLPROFILE_BLOCKS_GRBM,
    // HSA_VEN_AMD_AQLPROFILE_BLOCKS_SQ,
    // ... etc

    // Extended blocks for newer architectures
    AQLPROFILE_BLOCK_NAME_CPG = HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER,
    AQLPROFILE_BLOCK_NAME_RLC,

    // GFX12 specific blocks
    AQLPROFILE_BLOCK_NAME_CHA,
    AQLPROFILE_BLOCK_NAME_CHC,
    AQLPROFILE_BLOCK_NAME_GC_CANE,
    AQLPROFILE_BLOCK_NAME_GC_FFBM,
    AQLPROFILE_BLOCK_NAME_GC_L2TLB,
    AQLPROFILE_BLOCK_NAME_GC_UTCL1,
    AQLPROFILE_BLOCK_NAME_GC_UTCL2,
    AQLPROFILE_BLOCK_NAME_GC_VML2,
    AQLPROFILE_BLOCK_NAME_GCEA_SE,
    AQLPROFILE_BLOCK_NAME_GRBMH,
    AQLPROFILE_BLOCK_NAME_SQG,
} aqlprofile_block_name_t;
```

### Profile Configuration

```c
typedef struct {
    aqlprofile_agent_handle_t agent;        // Target agent
    const aqlprofile_pmc_event_t* events;   // Array of events to monitor
    uint32_t event_count;                   // Number of events
} aqlprofile_pmc_profile_t;
```

### Event Validation

```c
// Validate that an event is supported on the agent
hsa_status_t aqlprofile_validate_pmc_event(
    aqlprofile_agent_handle_t agent,
    const aqlprofile_pmc_event_t* event,
    bool* result);                          // Output: validation result
```

## AQL Packet Creation

### Packet Structures

```c
typedef struct {
    hsa_ext_amd_aql_pm4_packet_t start_packet;  // Reset counters and start
    hsa_ext_amd_aql_pm4_packet_t stop_packet;   // Pause counter incrementing
    hsa_ext_amd_aql_pm4_packet_t read_packet;   // Retrieve results from device
} aqlprofile_pmc_aql_packets_t;
```

### Packet Creation

```c
// Create PMC AQL packets
hsa_status_t aqlprofile_pmc_create_packets(
    aqlprofile_handle_t* handle,                    // Output: handle for data iteration
    aqlprofile_pmc_aql_packets_t* packets,          // Output: AQL packets
    aqlprofile_pmc_profile_t profile,               // Counter configuration
    aqlprofile_memory_alloc_callback_t alloc_cb,    // Memory allocation callback
    aqlprofile_memory_dealloc_callback_t dealloc_cb, // Memory deallocation callback
    aqlprofile_memory_copy_t memcpy_cb,             // Memory copy callback
    void* userdata);                                // User data for callbacks

// Delete packets and free resources
void aqlprofile_pmc_delete_packets(aqlprofile_handle_t handle);
```

## Data Collection

### Data Iteration Callback

```c
// Callback for processing counter results
typedef hsa_status_t (*aqlprofile_pmc_data_callback_t)(
    aqlprofile_pmc_event_t event,      // Event information
    uint64_t counter_id,               // Internal counter ID
    uint64_t counter_value,            // Counter value
    void* userdata);                   // User data

// Iterate through collected counter data
hsa_status_t aqlprofile_pmc_iterate_data(
    aqlprofile_handle_t handle,                    // Handle from packet creation
    aqlprofile_pmc_data_callback_t callback,      // Data processing callback
    void* userdata);                               // User data for callback
```

### Coordinate Information

```c
// Event coordinate iteration callback
typedef hsa_status_t (*aqlprofile_coordinate_callback_t)(
    int position,                      // Callback sequence number
    int id,                           // Coordinate ID
    int extent,                       // Maximum instances in dimension
    int coordinate,                   // Actual coordinate value [0, extent-1]
    const char* name,                 // Dimension name
    void* userdata);                  // User data

// Iterate event coordinates for detailed analysis
hsa_status_t aqlprofile_iterate_event_coord(
    aqlprofile_agent_handle_t agent,
    aqlprofile_pmc_event_t event,
    uint64_t sample_id,
    aqlprofile_coordinate_callback_t callback,
    void* userdata);
```

### Event Name Iteration

```c
// Event name callback
typedef hsa_status_t (*aqlprofile_eventname_callback_t)(
    int id,                           // Event dimension ID
    const char* name,                 // Dimension name
    void* data);                      // User data

// Iterate available event coordinate names
hsa_status_t aqlprofile_iterate_event_ids(
    aqlprofile_eventname_callback_t callback,
    void* user_data);
```

## Profile Information Queries

```c
typedef enum {
    AQLPROFILE_INFO_COMMAND_BUFFER_SIZE = 0,    // Returns uint32_t
    AQLPROFILE_INFO_PMC_DATA_SIZE = 1,          // Returns uint32_t
    AQLPROFILE_INFO_PMC_DATA = 2,               // Returns PMC data
    AQLPROFILE_INFO_BLOCK_COUNTERS = 4,         // Returns block counter count
    AQLPROFILE_INFO_BLOCK_ID = 5,               // Returns block ID info
    AQLPROFILE_INFO_ENABLE_CMD = 6,             // Returns enable command buffer
    AQLPROFILE_INFO_DISABLE_CMD = 7,            // Returns disable command buffer
} aqlprofile_pmc_info_type_t;

// Query profile information
hsa_status_t aqlprofile_get_pmc_info(
    const aqlprofile_pmc_profile_t* profile,
    aqlprofile_pmc_info_type_t attribute,
    void* value);
```

## Thread Trace Interface (ATT)

### Thread Trace Parameters

```c
typedef enum {
    AQLPROFILE_ATT_PARAMETER_NAME_BUFFER_SIZE_HIGH = 11,
    AQLPROFILE_ATT_PARAMETER_NAME_RT_TIMESTAMP,
} aqlprofile_att_parameter_name_ext_t;

typedef struct {
    hsa_ven_amd_aqlprofile_parameter_name_t parameter_name; // Or extended parameter
    union {
        uint32_t value;
        struct {
            uint32_t counter_id : 28;
            uint32_t simd_mask : 4;
        };
    };
} aqlprofile_att_parameter_t;

typedef struct {
    hsa_agent_t agent;                              // HSA agent
    const aqlprofile_att_parameter_t* parameters;   // ATT parameters
    uint32_t parameter_count;                       // Parameter count
} aqlprofile_att_profile_t;
```

### Thread Trace Packets

```c
typedef struct {
    hsa_ext_amd_aql_pm4_packet_t start_packet;  // Start thread trace
    hsa_ext_amd_aql_pm4_packet_t stop_packet;   // Stop and flush data
} aqlprofile_att_control_aql_packets_t;

// Create thread trace packets
hsa_status_t aqlprofile_att_create_packets(
    aqlprofile_handle_t* handle,
    aqlprofile_att_control_aql_packets_t* packets,
    aqlprofile_att_profile_t profile,
    aqlprofile_memory_alloc_callback_t alloc_cb,
    aqlprofile_memory_dealloc_callback_t dealloc_cb,
    aqlprofile_memory_copy_t memcpy_cb,
    void* userdata);

void aqlprofile_att_delete_packets(aqlprofile_handle_t handle);
```

### Thread Trace Data Collection

```c
// Thread trace data callback
typedef hsa_status_t (*aqlprofile_att_data_callback_t)(
    uint32_t shader,                   // Shader Engine ID
    void* buffer,                      // Trace data buffer
    uint64_t size,                     // Buffer size in bytes
    void* callback_data);              // User data

// Iterate thread trace data
hsa_status_t aqlprofile_att_iterate_data(
    aqlprofile_handle_t handle,
    aqlprofile_att_data_callback_t callback,
    void* userdata);
```

### Code Object Markers

```c
typedef struct {
    uint64_t id;                       // Code object ID
    uint64_t addr;                     // Load address
    uint64_t size;                     // Size in bytes
    hsa_agent_t agent;                 // Target agent
    uint32_t isUnload : 1;             // Unload marker flag
    uint32_t fromStart : 1;            // From start flag
} aqlprofile_att_codeobj_data_t;

// Create code object marker packet
hsa_status_t aqlprofile_att_codeobj_marker(
    hsa_ext_amd_aql_pm4_packet_t* packet,
    aqlprofile_handle_t* handle,
    aqlprofile_att_codeobj_data_t data,
    aqlprofile_memory_alloc_callback_t alloc_cb,
    aqlprofile_memory_dealloc_callback_t dealloc_cb,
    void* userdata);
```

## Complete Usage Example

```c
#include "aql_profile_v2.h"

// Memory allocation callback
hsa_status_t my_alloc(void** ptr, uint64_t size,
                     aqlprofile_buffer_desc_flags_t flags, void* userdata) {
    if (flags.device_access && flags.host_access) {
        // Allocate coherent memory for GPU/CPU access
        *ptr = allocate_coherent_memory(size);
    } else {
        *ptr = malloc(size);
    }
    return HSA_STATUS_SUCCESS;
}

void my_dealloc(void* ptr, void* userdata) {
    free(ptr);
}

hsa_status_t my_memcpy(void* dst, const void* src, size_t size, void* userdata) {
    memcpy(dst, src, size);
    return HSA_STATUS_SUCCESS;
}

// Data collection callback
hsa_status_t data_callback(aqlprofile_pmc_event_t event,
                          uint64_t counter_id,
                          uint64_t counter_value,
                          void* userdata) {
    printf("Block %d, Event %d: %lu\n",
           event.block_name, event.event_id, counter_value);
    return HSA_STATUS_SUCCESS;
}

int main() {
    // 1. Register agent
    aqlprofile_agent_info_v1_t agent_info = {
        .agent_gfxip = "gfx942",
        .xcc_num = 8,
        .se_num = 4,
        .cu_num = 304,
        .shader_arrays_per_se = 2,
        .domain = 0,
        .location_id = 0x1000
    };

    aqlprofile_agent_handle_t agent_handle;
    hsa_status_t status = aqlprofile_register_agent_info(
        &agent_handle, &agent_info, AQLPROFILE_AGENT_VERSION_V1);
    if (status != HSA_STATUS_SUCCESS) {
        return -1;
    }

    // 2. Define counter events
    aqlprofile_pmc_event_t events[] = {
        {
            .block_index = 0,
            .event_id = 123,
            .flags = {.raw = 0},
            .block_name = HSA_VEN_AMD_AQLPROFILE_BLOCKS_CPC
        },
        {
            .block_index = 0,
            .event_id = 456,
            .flags = {.raw = 0},
            .block_name = HSA_VEN_AMD_AQLPROFILE_BLOCKS_GRBM
        }
    };

    // 3. Validate events
    for (int i = 0; i < 2; i++) {
        bool valid = false;
        aqlprofile_validate_pmc_event(agent_handle, &events[i], &valid);
        if (!valid) {
            printf("Event %d is not valid\n", i);
            continue;
        }
    }

    // 4. Create profile
    aqlprofile_pmc_profile_t profile = {
        .agent = agent_handle,
        .events = events,
        .event_count = 2
    };

    // 5. Create AQL packets
    aqlprofile_handle_t handle;
    aqlprofile_pmc_aql_packets_t packets;
    status = aqlprofile_pmc_create_packets(
        &handle, &packets, profile,
        my_alloc, my_dealloc, my_memcpy, NULL);
    if (status != HSA_STATUS_SUCCESS) {
        return -1;
    }

    // 6. Submit packets to GPU queue
    // (Implementation depends on HSA queue management)
    // submit_aql_packet(&packets.start_packet);
    // submit_kernel_packet(...);
    // submit_aql_packet(&packets.stop_packet);
    // submit_aql_packet(&packets.read_packet);

    // 7. Collect and process results
    aqlprofile_pmc_iterate_data(handle, data_callback, NULL);

    // 8. Cleanup
    aqlprofile_pmc_delete_packets(handle);

    return 0;
}
```

## Architecture-Specific Features

### GFX9 (Vega)
- UCONFIG register space support
- TCA/TCC cache blocks
- Single SDMA engine
- Specific accumulation modes

### RDNA (GFX10+)
- GL1/GL2 cache hierarchy
- Dual SDMA engines
- WGP addressing mode
- Extended block types

### GFX12 (RDNA3)
- New performance blocks (CHA, CHC, etc.)
- Enhanced thread trace capabilities
- Extended parameter sets
- Additional memory hierarchy blocks

## Best Practices

1. **Always validate events** before packet creation
2. **Use appropriate memory allocation** based on access patterns
3. **Handle callback errors** properly in data iteration
4. **Check status codes** for all function calls
5. **Clean up resources** with delete functions
6. **Architecture awareness** when selecting counters

## Troubleshooting

### Common Issues

1. **Invalid Events**: Use `aqlprofile_validate_pmc_event()` to verify support
2. **Memory Allocation Failures**: Ensure proper allocation callback implementation
3. **Block Limits**: Some blocks have limited counters available
4. **Architecture Mismatch**: Verify agent registration matches hardware

### Debug Information

Use coordinate iteration to understand counter mapping:
```c
aqlprofile_iterate_event_coord(agent, event, sample_id,
                              coordinate_callback, userdata);
```

This provides detailed information about how counters are distributed across GPU dimensions (XCC, SE, etc.).