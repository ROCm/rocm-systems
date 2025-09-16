# ROCProfiler SDK Counter Usage with AQLProfile v2

## Overview

ROCProfiler SDK provides a high-level interface for hardware counter collection that internally uses AQLProfile v2 for packet construction and counter management. This document explains how ROCProfiler SDK integrates with AQLProfile v2 and manages counter definitions.

## Architecture Overview

```
User Application
      ↓
ROCProfiler SDK (counters.h, counter_config.h)
      ↓
CounterPacketConstruct (packet_construct.hpp)
      ↓
AQLProfile v2 (aql_profile_v2.h)
      ↓
Hardware GPU Counters
```

## Core Counter Types

### Counter Information Structure (v1)

```cpp
typedef struct rocprofiler_counter_info_v1_t {
    uint64_t size;                      // Structure size for versioning
    rocprofiler_counter_id_t id;        // Counter ID
    const char* name;                   // Counter name
    const char* description;            // Counter description
    const char* block;                  // Block name (non-derived only)
    const char* expression;             // Counter expression (derived only)
    uint8_t is_constant : 1;            // Hardware constant flag
    uint8_t is_derived : 1;             // Derived counter flag

    uint64_t dimensions_count;          // Number of dimensions
    const rocprofiler_counter_record_dimension_info_t** dimensions;
    uint64_t dimensions_instances_count; // Instance count
    const rocprofiler_counter_record_dimension_instance_info_t** dimensions_instances;
} rocprofiler_counter_info_v1_t;
```

### Counter Dimension Information

```cpp
typedef struct rocprofiler_counter_dimension_info_t {
    uint64_t size;                      // Structure size
    const char* dimension_name;         // Dimension name (e.g., "XCC", "SE")
    size_t index;                       // Position within dimension
} rocprofiler_counter_dimension_info_t;
```

### Counter Instance Information

```cpp
typedef struct rocprofiler_counter_record_dimension_instance_info_t {
    uint64_t size;                      // Structure size
    rocprofiler_counter_instance_id_t instance_id;    // Encoded instance ID
    uint64_t counter_id;                // Associated counter ID
    uint64_t dimensions_count;          // Number of dimensions
    const rocprofiler_counter_dimension_info_t** dimensions;  // Dimension array
} rocprofiler_counter_record_dimension_instance_info_t;
```

## Counter Configuration

### Creating Counter Configurations

```cpp
// Create a counter configuration for a specific agent
rocprofiler_status_t
rocprofiler_create_counter_config(rocprofiler_agent_id_t agent_id,
                                  rocprofiler_counter_id_t* counters_list,
                                  size_t counters_count,
                                  rocprofiler_counter_config_id_t* config_id);

// Destroy configuration when done
rocprofiler_status_t
rocprofiler_destroy_counter_config(rocprofiler_counter_config_id_t config_id);
```

### Counter Discovery

```cpp
// Callback for available counters
typedef rocprofiler_status_t (*rocprofiler_available_counters_cb_t)(
    rocprofiler_agent_id_t agent_id,
    rocprofiler_counter_id_t* counters,
    size_t num_counters,
    void* user_data);

// Iterate through available counters for an agent
rocprofiler_status_t
rocprofiler_iterate_agent_supported_counters(rocprofiler_agent_id_t agent_id,
                                             rocprofiler_available_counters_cb_t cb,
                                             void* user_data);
```

## Internal AQLProfile v2 Integration

### CounterPacketConstruct Class

Located in `packet_construct.hpp`, this class bridges ROCProfiler SDK and AQLProfile v2:

```cpp
class CounterPacketConstruct {
public:
    CounterPacketConstruct(rocprofiler_agent_id_t agent,
                           const std::vector<counters::Metric>& metrics);

    // Create AQL packets using AQLProfile v2
    std::unique_ptr<hsa::CounterAQLPacket> construct_packet(const CoreApiTable&,
                                                            const AmdExtTable&);

    // Event-to-metric mapping
    const counters::Metric* event_to_metric(const aqlprofile_pmc_event_t& event) const;
    std::vector<aqlprofile_pmc_event_t> get_all_events() const;

    // Validate collection capability
    rocprofiler_status_t can_collect();

private:
    struct AQLProfileMetric {
        counters::Metric metric;
        std::vector<aqlprofile_pmc_event_t> instances;
        std::vector<aqlprofile_pmc_event_t> events;
    };

    rocprofiler_agent_id_t _agent;
    std::vector<AQLProfileMetric> _metrics;
    std::vector<aqlprofile_pmc_event_t> _events;
    std::map<aqlprofile_pmc_event_t, counters::Metric> _event_to_metric;
};
```

### AQLProfile v2 Event Structure

```cpp
// AQLProfile v2 event structure used internally
typedef struct {
    uint32_t block_index;                           // Block channel
    uint32_t event_id;                              // Event ID from XML
    aqlprofile_pmc_event_flags_t flags;             // Special flags
    hsa_ven_amd_aqlprofile_block_name_t block_name; // Block name
} aqlprofile_pmc_event_t;
```

### Agent Registration

```cpp
// AQLProfile v2 agent information
typedef struct {
    const char* agent_gfxip;        // GPU architecture name
    uint32_t xcc_num;               // Number of XCCs
    uint32_t se_num;                // Number of Shader Engines
    uint32_t cu_num;                // Number of Compute Units
    uint32_t shader_arrays_per_se;  // Shader arrays per SE
    uint32_t domain;                // PCI domain
    uint32_t location_id;           // PCI BDF
} aqlprofile_agent_info_v1_t;

// Register agent with AQLProfile v2
hsa_status_t aqlprofile_register_agent_info(aqlprofile_agent_handle_t* agent_id,
                                            const void* agent_info,
                                            aqlprofile_agent_version_t version);
```

## Counter Data Flow

### 1. Counter Definition Phase

```cpp
// ROCProfiler SDK discovers available counters
rocprofiler_iterate_agent_supported_counters(agent_id, [](
    rocprofiler_agent_id_t agent,
    rocprofiler_counter_id_t* counters,
    size_t num_counters,
    void* data) {

    // Process available counters
    for (size_t i = 0; i < num_counters; ++i) {
        rocprofiler_counter_info_v1_t info;
        rocprofiler_query_counter_info(counters[i],
                                      ROCPROFILER_COUNTER_INFO_VERSION_1,
                                      &info);
        // Use counter information
    }
    return ROCPROFILER_STATUS_SUCCESS;
}, user_data);
```

### 2. Configuration Creation

```cpp
// Create configuration with selected counters
std::vector<rocprofiler_counter_id_t> selected_counters = {
    counter_id_1, counter_id_2, /* ... */
};

rocprofiler_counter_config_id_t config_id;
rocprofiler_create_counter_config(agent_id,
                                 selected_counters.data(),
                                 selected_counters.size(),
                                 &config_id);
```

### 3. Internal Packet Construction

```cpp
// Internally, ROCProfiler SDK uses CounterPacketConstruct
CounterPacketConstruct constructor(agent_id, metrics);

// Validate that counters can be collected
if (constructor.can_collect() != ROCPROFILER_STATUS_SUCCESS) {
    // Handle collection validation failure
}

// Create AQL packets via AQLProfile v2
auto packet = constructor.construct_packet(core_api, amd_ext);
```

### 4. AQLProfile v2 Packet Creation

```cpp
// CounterPacketConstruct converts ROCProfiler metrics to AQLProfile events
std::vector<aqlprofile_pmc_event_t> events = constructor.get_all_events();

aqlprofile_pmc_profile_t profile = {
    .agent = agent_handle,
    .events = events.data(),
    .event_count = static_cast<uint32_t>(events.size())
};

// Create packets using AQLProfile v2
aqlprofile_handle_t handle;
aqlprofile_pmc_aql_packets_t packets;
hsa_status_t status = aqlprofile_pmc_create_packets(
    &handle, &packets, profile,
    alloc_callback, dealloc_callback, memcpy_callback, userdata);
```

## Memory Management

### Allocation Callbacks

ROCProfiler SDK provides memory allocation callbacks to AQLProfile v2:

```cpp
// Memory allocation callback for AQLProfile v2
hsa_status_t memory_alloc_callback(void** ptr, uint64_t size,
                                  aqlprofile_buffer_desc_flags_t flags,
                                  void* userdata) {
    // Allocate memory based on flags
    if (flags.device_access && flags.host_access) {
        // Allocate coherent memory
        *ptr = allocate_coherent_memory(size);
    } else if (flags.device_access) {
        // Allocate device memory
        *ptr = allocate_device_memory(size);
    } else {
        // Allocate host memory
        *ptr = malloc(size);
    }
    return HSA_STATUS_SUCCESS;
}

// Memory deallocation callback
void memory_dealloc_callback(void* ptr, void* userdata) {
    // Free memory appropriately
    free_memory(ptr);
}
```

## Data Collection and Results

### Counter Value Retrieval

```cpp
// AQLProfile v2 provides data iteration callback
hsa_status_t data_callback(aqlprofile_pmc_event_t event,
                          uint64_t counter_id,
                          uint64_t counter_value,
                          void* userdata) {
    // Process counter results
    printf("Block: %d, Event: %d, Value: %lu\n",
           event.block_name, event.event_id, counter_value);
    return HSA_STATUS_SUCCESS;
}

// Iterate through collected data
aqlprofile_pmc_iterate_data(handle, data_callback, userdata);
```

### Coordinate Information

```cpp
// Get dimension information for counter instances
hsa_status_t coordinate_callback(int position, int id, int extent,
                                int coordinate, const char* name,
                                void* userdata) {
    printf("Dimension %s[%d/%d] at position %d\n",
           name, coordinate, extent, position);
    return HSA_STATUS_SUCCESS;
}

// Iterate event coordinates
aqlprofile_iterate_event_coord(agent_handle, event, sample_id,
                              coordinate_callback, userdata);
```

## Block and Counter Mapping

### Block Name Translation

```cpp
// ROCProfiler SDK maps block names to AQLProfile v2 enums
const char* block_name_to_string(hsa_ven_amd_aqlprofile_block_name_t block) {
    switch (block) {
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_CPC: return "CPC";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_GRBM: return "GRBM";
        case HSA_VEN_AMD_AQLPROFILE_BLOCKS_SQ: return "SQ";
        // ... more mappings
        default: return "UNKNOWN";
    }
}
```

### Event Validation

```cpp
// Validate events before packet creation
for (const auto& event : events) {
    bool valid = false;
    hsa_status_t status = aqlprofile_validate_pmc_event(agent_handle, &event, &valid);
    if (status != HSA_STATUS_SUCCESS || !valid) {
        // Handle invalid event
        continue;
    }
}
```

## Usage Example

```cpp
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/counter_config.h>

// 1. Get available counters
std::vector<rocprofiler_counter_id_t> available_counters;
rocprofiler_iterate_agent_supported_counters(agent_id,
    [](rocprofiler_agent_id_t agent, rocprofiler_counter_id_t* counters,
       size_t count, void* data) {
        auto* vec = static_cast<std::vector<rocprofiler_counter_id_t>*>(data);
        vec->assign(counters, counters + count);
        return ROCPROFILER_STATUS_SUCCESS;
    }, &available_counters);

// 2. Create configuration with subset of counters
rocprofiler_counter_config_id_t config_id;
rocprofiler_create_counter_config(agent_id,
                                 available_counters.data(),
                                 std::min(available_counters.size(), 10UL),
                                 &config_id);

// 3. Use configuration in profiling context
// (Implementation depends on specific ROCProfiler SDK usage)

// 4. Cleanup
rocprofiler_destroy_counter_config(config_id);
```

## Key Integration Points

1. **Agent Registration**: ROCProfiler SDK registers agents with AQLProfile v2
2. **Counter Translation**: SDK counters are mapped to AQLProfile v2 events
3. **Packet Construction**: AQLProfile v2 creates the actual AQL packets
4. **Memory Management**: SDK provides allocation callbacks to AQLProfile v2
5. **Data Collection**: AQLProfile v2 handles data iteration and parsing
6. **Validation**: Both layers validate counter configurations

## Architecture-Specific Considerations

### GFX9 (Vega)
- Supports UCONFIG register space
- Has different cache hierarchy (TCA/TCC vs GL1/GL2)
- Single SDMA engine

### RDNA (GFX10+)
- GL1/GL2 cache hierarchy
- Dual SDMA engines
- Different counter block organization

### GFX12 (RDNA3)
- Additional blocks: CHA, CHC, GC_CANE, etc.
- Enhanced dimension support
- Extended parameter sets

The ROCProfiler SDK provides a comprehensive abstraction over AQLProfile v2, handling the complexity of counter management, packet construction, and data collection while exposing a clean, modern C++ interface to users.