# AGENT_CODE_INFO.md - aql_c

## Directory Purpose

This directory contains a complete C implementation of the AQL (Asynchronous Queue Language) packet generation library, originally ported from C++ AQLProfile v2. The library provides architecture-abstracted GPU performance counter programming for AMD GPUs through PM4 command packet generation. It supports multiple GPU architectures (GFX9, GFX10, GFX11, GFX12) with a unified interface while handling architecture-specific differences transparently.

## Key Components

### Architecture Abstraction Layer
- **aql_arch_ops.h**: Function pointer table interface for architecture-specific operations
- **aql_arch_detect.c**: Runtime architecture detection and operations table selection
- **aql_types.h**: Core data structures, enumerations, and type definitions

### Architecture-Specific Implementations
- **aql_gfx9_ops.c**: GFX9 (Vega) implementation with PRED_EXEC and UCONFIG support
- **aql_gfx9_defs.h**: GFX9-specific constants, register definitions, and PM4 opcodes
- **aql_gfx12_ops.c**: GFX12 (RDNA3) implementation with modern cache hierarchy
- **aql_gfx12_defs.h**: GFX12-specific constants and register definitions

### Command Generation Infrastructure
- **aql_cmd_buffer.h/c**: Command buffer management for PM4 command accumulation
- **aql_packet.c**: AQL packet construction and PM4 command integration

### High-Level Interface
- **aql_pmc_interface.h/c**: AQLProfile v2-compatible PMC interface for counter operations

## Architecture Overview

The library implements a three-layer architecture:

1. **Hardware Abstraction Layer (HAL)**:
   - Architecture operations tables (`aql_arch_ops_t`)
   - Register space definitions and validation
   - PM4 command generation functions

2. **Command Buffer Layer**:
   - Memory-safe command accumulation
   - Bounds checking and validation
   - Debug tracing and packet validation

3. **High-Level Interface Layer**:
   - PMC packet creation (start/stop/read)
   - Counter event specification
   - AQLProfile v2 compatibility

## Key Data Structures

### Architecture Operations Table
```c
typedef struct aql_arch_ops {
    const char* arch_name;              /* Architecture name */
    aql_arch_type_t arch_type;          /* Architecture enumeration */

    /* Capabilities */
    bool has_pred_exec;                 /* GFX9 feature */
    bool has_uconfig_space;             /* GFX9 feature */
    bool has_dual_sdma;                 /* RDNA+ feature */
    bool has_gl_cache_hierarchy;        /* RDNA+ feature */

    /* PM4 command generation functions */
    aql_result_t (*build_write_uconfig_reg)(aql_cmd_buffer_t*, uint32_t, uint32_t);
    aql_result_t (*build_write_config_reg)(aql_cmd_buffer_t*, uint32_t, uint32_t);
    aql_result_t (*build_copy_reg_data)(aql_cmd_buffer_t*, uint32_t, void*, uint32_t, bool);
    /* ... additional PM4 builders ... */
} aql_arch_ops_t;
```

### Command Buffer Structure
```c
typedef struct {
    uint32_t* data;                     /* Command data buffer */
    size_t capacity;                    /* Total capacity in dwords */
    size_t used;                        /* Used size in dwords */
    bool is_external;                   /* External buffer flag */
} aql_cmd_buffer_t;
```

### AQL Packet Structure
```c
typedef struct {
    uint16_t header;                    /* AQL packet header */
    uint16_t pm4_ib_format;             /* PM4 IB format */
    uint32_t pm4_ib_command[4];         /* PM4 IB command */
    uint32_t dw_count_remain;           /* Remaining dword count */
    uint32_t reserved[8];               /* Reserved fields */
    uint64_t completion_signal;         /* Completion signal handle */
} aql_pm4_ib_packet_t;  /* Exactly 64 bytes */
```

## Public APIs

### Architecture Detection and Selection
```c
/* Detect architecture from GPU name string */
const aql_arch_ops_t* aql_detect_architecture(const char* gfx_name);

/* Get architecture by type enumeration */
const aql_arch_ops_t* aql_get_arch_ops(aql_arch_type_t arch_type);

/* List all supported architectures */
uint32_t aql_list_supported_architectures(const aql_arch_ops_t** ops_list, uint32_t max_count);
```

### Command Buffer Management
```c
/* Initialize command buffer with external storage */
aql_result_t aql_cmd_buffer_init(aql_cmd_buffer_t* buf, uint32_t* data, size_t capacity);

/* Append commands to buffer */
aql_result_t aql_cmd_buffer_append_dwords(aql_cmd_buffer_t* buf, const uint32_t* data, size_t count);

/* Buffer validation and utilities */
aql_result_t aql_cmd_buffer_validate(const aql_cmd_buffer_t* buf);
bool aql_cmd_buffer_is_valid(const aql_cmd_buffer_t* buf);
```

### PMC Interface (AQLProfile v2 Compatible)
```c
/* Create PMC packets for performance monitoring */
aql_result_t aql_pmc_create_packets(aql_pmc_packets_t* packets, const aql_pmc_profile_t* profile);

/* Delete PMC packets and free resources */
aql_result_t aql_pmc_delete_packets(aql_pmc_packets_t* packets);

/* Helper for creating counter events */
aql_result_t aql_create_counter_event(aql_pmc_event_t* event, const char* block_name,
                                     uint32_t instance, uint32_t event_id);
```

## Design Patterns

### Function Pointer Table Pattern
```c
/* Each architecture provides its own operations table */
const aql_arch_ops_t aql_gfx9_ops = {
    .arch_name = "GFX9",
    .build_write_uconfig_reg = gfx9_build_write_uconfig_reg,
    .build_write_config_reg = gfx9_build_write_config_reg,
    /* ... */
};
```

### Builder Pattern for Commands
```c
/* Commands built incrementally in buffer */
aql_result_t build_counter_programming_sequence(aql_cmd_buffer_t* buf,
                                               const aql_arch_ops_t* ops) {
    ops->build_write_config_reg(buf, GRBM_GFX_INDEX, grbm_value);
    ops->build_write_uconfig_reg(buf, select_reg, event_select);
    ops->build_write_uconfig_reg(buf, enable_reg, 1);
    return AQL_SUCCESS;
}
```

### Memory Safety Pattern
```c
/* All buffer operations include bounds checking */
#define AQL_CHECK_BUFFER_SPACE(buf, required_dwords) \
    do { \
        if (!aql_cmd_buffer_is_valid(buf)) return AQL_ERROR_INVALID_ARGUMENT; \
        if (aql_cmd_buffer_available(buf) < (required_dwords)) return AQL_ERROR_BUFFER_TOO_SMALL; \
    } while(0)
```

## Dependencies

### Internal Dependencies
- **aql_types.h**: Core type definitions required by all modules
- **aql_arch_ops.h**: Architecture operations interface
- **aql_cmd_buffer.h**: Command buffer utilities

### External Dependencies
- **Kernel Context**: linux/types.h, linux/string.h, linux/slab.h
- **User Context**: stdint.h, stdbool.h, string.h, stdlib.h
- **Build System**: Compiled as part of kernel module via parent Makefile

### Cross-Module Dependencies
```c
/* aql_pmc_interface.c depends on: */
#include "aql_arch_ops.h"      /* Architecture detection */
#include "aql_cmd_buffer.h"    /* Command generation */

/* Architecture implementations depend on: */
#include "aql_types.h"         /* Core types */
#include "aql_gfx9_defs.h"     /* Architecture constants */
```

## Data Flow

### Packet Generation Flow
1. **Architecture Detection**: `aql_detect_architecture(gfx_name)` → `aql_arch_ops_t*`
2. **Command Buffer Setup**: `aql_cmd_buffer_init()` → allocated buffer
3. **Command Generation**: `ops->build_*()` functions → PM4 commands in buffer
4. **Packet Creation**: `aql_populate_packet_from_buffer()` → AQL packet
5. **Resource Cleanup**: Buffer and packet cleanup

### Counter Programming Flow
1. **Event Specification**: User provides block name, instance, event ID
2. **Block Validation**: Architecture validates block and register availability
3. **Register Calculation**: `ops->get_counter_register_addr()` → register addresses
4. **Command Sequence**: Generate select → enable → read command sequence
5. **Packet Assembly**: Commands assembled into start/stop/read packets

### Architecture Abstraction Flow
1. **Pattern Matching**: GPU name matched against architecture patterns
2. **Operations Selection**: Appropriate `aql_arch_ops_t` table selected
3. **Feature Detection**: Architecture capabilities queried
4. **Command Dispatch**: Architecture-specific command generation called

## Integration Points

### Kernel Module Integration
- Compiled directly into kernel module object files
- No external library dependencies
- Kernel-safe memory allocation and string operations
- Compatible with kernel module licensing (GPL)

### Counter Block Support
Supports major AMD GPU counter blocks:
- **SQ (Shader Quad)**: Shader execution metrics
- **TCC (Texture Cache Controller)**: Cache performance (GFX9)
- **TCP (Texture Cache Per-pipe)**: Per-pipe cache metrics
- **GRBM (Graphics Register Bus Manager)**: Graphics pipeline activity
- **CPC (Command Processor Compute)**: Compute workload metrics
- **GL1C/GL2C**: Cache hierarchy (RDNA+)

### Register Space Handling
- **CONFIG Space**: Privileged registers (0x2000-0x9FFF)
- **UCONFIG Space**: User-accessible registers (0xC000-0xC0FF, GFX9 only)
- **Register Validation**: Architecture-specific address validation
- **SMN Addressing**: System Management Network support

## Critical Business Logic

### Architecture Detection Algorithm
```c
const aql_arch_ops_t* aql_detect_architecture(const char* gfx_name) {
    /* Pattern matching with fallback logic */
    for (i = 0; i < ARRAY_SIZE(architecture_patterns); i++) {
        if (strstr(gfx_name, architecture_patterns[i].pattern)) {
            return architecture_patterns[i].ops;
        }
    }
    return NULL; /* Unsupported architecture */
}
```

### PM4 Command Generation
Each architecture implements specific PM4 packet formats:
- **GFX9**: 7-dword ACQUIRE_MEM packets, PRED_EXEC support
- **GFX12**: 8-dword ACQUIRE_MEM packets, enhanced cache control
- **Register Programming**: Architecture-specific register addresses and formats
- **Cache Management**: Different coherence mechanisms per architecture

### Buffer Management Strategy
- **Static Allocation**: Preferred for kernel compatibility
- **Dynamic Allocation**: Available with callback mechanism
- **Bounds Checking**: All operations validate buffer space
- **Alignment Handling**: PM4 packet alignment requirements

### Error Handling Philosophy
- **Comprehensive Validation**: All inputs validated before processing
- **Early Error Detection**: Invalid configurations detected at creation time
- **Graceful Degradation**: Unsupported features return appropriate error codes
- **Debug Support**: Extensive debug tracing available when enabled

## Notable Implementation Details

### Kernel Compatibility Considerations
- **No C++ Features**: Pure C implementation for kernel compatibility
- **No Standard Library**: Uses kernel equivalents of string/memory functions
- **Static Linking**: All code linked into kernel module
- **GPL Licensing**: Compatible with kernel module requirements

### Memory Management
- **Stack-Friendly**: Most operations use stack-allocated structures
- **Callback-Based Allocation**: Supports custom memory allocators
- **Resource Cleanup**: Explicit cleanup functions for all resources
- **Leak Prevention**: Reference counting and cleanup validation

### Performance Optimizations
- **Function Pointer Tables**: O(1) architecture-specific dispatch
- **Batch Command Generation**: Multiple commands generated in single buffer
- **Minimal Allocations**: Reuse of command buffers across operations
- **Cache-Friendly**: Data structures designed for cache efficiency

### Debug and Validation Features
- **Compile-Time Debugging**: `AQL_DEBUG_TRACE` conditional compilation
- **Packet Validation**: Comprehensive PM4 packet format validation
- **Command Tracing**: Debug output for all command generation
- **Buffer State Tracking**: Validation of buffer consistency

This AQL C library provides a robust, kernel-compatible foundation for AMD GPU performance counter programming while maintaining compatibility with existing AQLProfile v2 interfaces and supporting future GPU architectures through its extensible design.