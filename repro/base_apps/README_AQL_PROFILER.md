# AQL Profiler Implementation

This directory contains a reusable AQL profiling implementation using the aqlprofile v2 interface.

## Files Created

### Core Implementation
- **`aql_profiler.h`** - Header file with C/C++ compatible interface
- **`aql_profiler.cpp`** - Implementation using aqlprofile v2 interface
- **`Makefile.aql`** - Build system for the AQL profiler

### Examples and Tests
- **`simple_aql_example.cpp`** - Simple example showing direct usage of v2 interface
- **`aql_profiler_test.cpp`** - Verification tool with multiple test scenarios
- **`app_with_aql_profiling.cpp`** - Integration example with HSA applications
- **`validate_aql_profiler.sh`** - Generated validation script

## Key Features

### 1. Simple Interface
The `aql_profiler_create_packets_simple()` function provides a straightforward way to create AQL packets:

```cpp
int aql_profiler_create_packets_simple(
    hsa_agent_t agent,
    hsa_queue_t* queue,                    // Optional - not used for packet creation
    const aqlprofile_pmc_event_t* events,  // Pre-formatted events
    uint32_t event_count,
    aqlprofile_pmc_aql_packets_t* packets, // Output: created packets
    aqlprofile_handle_t* handle            // Output: handle for cleanup
);
```

### 2. Packet-Only Approach
The implementation creates AQL packets but **does not submit them to queues**. This allows:
- External control over packet submission timing
- Integration with existing queue management
- Flexibility in workload scheduling

### 3. Specific Events Support
Uses the exact events you specified:
- `block_index=0, event_id=35, flags=0x0, block_name=6` (SQ)
- `block_index=0, event_id=37, flags=0x0, block_name=6` (SQ)
- `block_index=0, event_id=40, flags=0x0, block_name=6` (SQ)
- `block_index=0, event_id=54, flags=0x0, block_name=6` (SQ)
- `block_index=0, event_id=36, flags=0x0, block_name=6` (SQ)

## Usage Example

```cpp
#include "aql_profiler.h"

// Define events (block_name=6 is SQ)
aqlprofile_pmc_event_t events[] = {
    {.block_index = 0, .event_id = 35, .flags = {.raw = 0},
     .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)},
    {.block_index = 0, .event_id = 37, .flags = {.raw = 0},
     .block_name = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(6)},
    // ... more events
};

// Create packets
aqlprofile_pmc_aql_packets_t packets;
aqlprofile_handle_t handle;

int result = aql_profiler_create_packets_simple(
    gpu_agent,
    queue,
    events,
    event_count,
    &packets,
    &handle
);

// Use packets.start_packet, packets.read_packet, packets.stop_packet
// Submit them to your queue when ready

// Cleanup when done
aql_profiler_cleanup_simple(handle);
```

## Build Instructions

### Prerequisites
- ROCm/HIP installation with HSA runtime
- aqlprofile v2 interface headers (included in this project)
- Access to aqlprofile library (for actual runtime execution)

### Building
```bash
# Check dependencies
make -f Makefile.aql check-deps

# Build library
make -f Makefile.aql libaql_profiler.a

# Build examples (successfully links with libhsa-amd-aqlprofile64)
make -f Makefile.aql simple_aql_example

# Build test suite
make -f Makefile.aql aql_profiler_test

# Test linkage without GPU dependency
./test_aql_linkage

# Generate validation script
make -f Makefile.aql validation-script
```

### Integration with Existing Apps
The implementation can be compiled with your existing applications:

```bash
hipcc -I../../projects/aqlprofile/src/core/include \
      your_app.cpp aql_profiler.cpp \
      -lhsa-runtime64 -laqlprofile \
      -o your_app
```

## Interface Design

### High-Level API
- `aql_profiler_create()` - Create profiler context
- `aql_profiler_add_counter()` - Add events to monitor
- `aql_profiler_create_packets()` - Generate AQL packets
- `aql_profiler_get_packets()` - Retrieve packets for submission
- `aql_profiler_collect_results()` - Parse results after execution

### Simple API
- `aql_profiler_create_packets_simple()` - One-shot packet creation
- `aql_profiler_cleanup_simple()` - Cleanup

## Packet Flow

1. **Create Events**: Define `aqlprofile_pmc_event_t` structures
2. **Generate Packets**: Call `aql_profiler_create_packets_simple()`
3. **Submit Start Packet**: Submit `packets.start_packet` to queue
4. **Run Workload**: Execute your GPU work
5. **Submit Read Packet**: Submit `packets.read_packet` to queue
6. **Submit Stop Packet**: Submit `packets.stop_packet` to queue
7. **Parse Results**: Use `aqlprofile_pmc_iterate_data()` to get counter values
8. **Cleanup**: Call `aql_profiler_cleanup_simple()`

## Memory Management

The implementation provides callbacks for memory allocation:
- CPU memory pool for fine-grained access
- GPU memory pool for device-local data
- Automatic cleanup on destruction

## Error Handling

- Returns `-1` on errors with stderr output
- Validates events against agent capabilities
- Proper resource cleanup on failure paths

## Debugging Integration

Examples show integration with debugger blocking mechanisms:
- `app_debugger_block()` - Pause execution for debugging
- `app_debugger_continue()` - Resume execution
- Compatible with existing base_apps pattern

## Limitations

- Requires aqlprofile v2 library at runtime
- HSA agent and memory pool discovery required
- PM4 packet generation depends on GPU architecture support
- Event validation may vary across different GPU generations

## Future Enhancements

- Add support for ATT (Advanced Thread Trace) packets
- Extend event validation with detailed error messages
- Add helper functions for common profiling scenarios
- Support for batch processing multiple workloads