# C Data Structures Design Documentation

## Overview

This document details the design of C data structures that replace the C++ classes from the original AQLProfile implementation. The design prioritizes kernel module compatibility, performance, and maintainability while preserving all functionality from the original implementation.

## Design Principles

### 1. Kernel Module Compatibility
- **No Dynamic Allocation in Critical Paths**: Pre-allocated buffers and static tables
- **Fixed-Size Structures**: Avoid variable-length arrays where possible
- **Explicit Memory Management**: Clear ownership and lifetime semantics
- **Error Handling**: Return codes instead of exceptions

### 2. Performance Optimization
- **Cache-Friendly Layouts**: Structures organized for locality of reference
- **Minimal Indirection**: Direct data access without virtual function overhead
- **Static Dispatch**: Function pointer tables resolved at initialization
- **Zero-Copy Operations**: In-place buffer manipulation where possible

### 3. Type Safety and Validation
- **Strong Typing**: Enumerations for all discrete values
- **Size Assertions**: Compile-time size checks for critical structures
- **Bounds Checking**: Runtime validation for all buffer operations
- **Const Correctness**: Immutable data marked appropriately

## Core Data Structure Files

### aql_types.h - Fundamental Types

This file defines the core types used throughout the system:

#### Key Structures:

**`aql_cmd_buffer_t`** - Replaces C++ `CmdBuffer`
```c
typedef struct {
    uint32_t* data;        // Command data buffer
    size_t capacity;       // Total capacity in dwords
    size_t used;          // Used size in dwords
    bool is_external;     // External vs internal allocation
} aql_cmd_buffer_t;
```

**Benefits over C++ version:**
- Explicit capacity tracking prevents buffer overruns
- External allocation support for kernel pre-allocated buffers
- No hidden dynamic allocation
- Direct access to underlying data

**`aql_counter_request_t`** - Counter specification
```c
typedef struct {
    aql_block_id_t block_id;      // Which hardware block
    uint32_t block_instance;       // Instance within block type
    uint32_t counter_id;          // Counter index within block
    uint32_t event_select;        // Event to count
    uint32_t flags;               // Special mode flags

    // Runtime assignment
    uint32_t assigned_register;   // Hardware register assigned
    void* result_location;        // Where to store result

    // Debug information
    const char* block_name;       // Human-readable name
    const char* counter_name;     // Counter description
} aql_counter_request_t;
```

**Improvements over C++ version:**
- All information in single structure (no inheritance hierarchy)
- Explicit result location for zero-copy result retrieval
- Debug information embedded for troubleshooting
- Clear separation of configuration vs runtime state

**`aql_pm4_ib_packet_t`** - AQL packet structure
```c
typedef struct {
    uint16_t header;
    uint16_t pm4_ib_format;
    uint32_t pm4_ib_command[4];
    uint32_t dw_count_remain;
    uint32_t reserved[8];
    uint64_t completion_signal;
} aql_pm4_ib_packet_t;

_Static_assert(sizeof(aql_pm4_ib_packet_t) == 64, "AQL packet must be 64 bytes");
```

**Key features:**
- Exact binary compatibility with HSA AQL format
- Compile-time size validation
- Simplified signal handling (uint64_t vs complex HSA type)
- Clear field documentation

### aql_arch_ops.h - Architecture Abstraction

This file replaces the C++ virtual function interface with function pointer tables:

#### Architecture Operations Table:

**`aql_arch_ops_t`** - Complete architecture interface
```c
typedef struct aql_arch_ops {
    // Identification
    const char* arch_name;
    aql_arch_type_t arch_type;
    uint32_t gfx_version;

    // Capabilities
    bool has_pred_exec;
    bool has_uconfig_space;
    bool has_dual_sdma;
    // ... more capability flags

    // Register space definitions
    aql_register_spaces_t register_spaces;

    // Function pointers for all operations
    aql_result_t (*build_write_uconfig_reg)(aql_cmd_buffer_t*, uint32_t, uint32_t);
    aql_result_t (*build_write_config_reg)(aql_cmd_buffer_t*, uint32_t, uint32_t);
    // ... all other operations
} aql_arch_ops_t;
```

**Advantages over C++ virtual functions:**
- **Static Resolution**: Function pointers resolved once at initialization
- **No Vtable Overhead**: Direct function calls
- **Explicit Capabilities**: Boolean flags for feature detection
- **Compile-Time Validation**: Missing functions cause link errors
- **Debug Visibility**: Function pointers visible in debugger

### aql_cmd_buffer.h - Command Buffer Management

This file provides the command buffer interface that replaces C++ `CmdBuffer`:

#### Buffer Management Functions:
- **Initialization**: Support for both external and allocated buffers
- **Bounds Checking**: All operations validate buffer space
- **Debug Tracing**: Optional command tracing for development
- **Alignment**: Automatic padding and alignment support

#### Key Improvements:
1. **Safety**: Comprehensive bounds checking prevents buffer overruns
2. **Flexibility**: Works with pre-allocated or dynamically allocated buffers
3. **Debug Support**: Built-in tracing and validation
4. **Performance**: Zero-copy operations where possible

## Architecture-Specific Implementations

### Static Function Tables

Each architecture provides a static function table:

```c
// gfx9_ops.c
const aql_arch_ops_t aql_gfx9_ops = {
    .arch_name = "gfx9",
    .arch_type = AQL_ARCH_GFX9,
    .has_pred_exec = true,
    .has_uconfig_space = true,
    .build_write_uconfig_reg = aql_gfx9_build_write_uconfig_reg,
    .build_write_config_reg = aql_gfx9_build_write_config_reg,
    // ... all other functions
};
```

### Architecture Detection

```c
const aql_arch_ops_t* aql_detect_architecture(const char* gfx_name) {
    if (strstr(gfx_name, "gfx94")) return &aql_gfx9_ops;
    if (strstr(gfx_name, "gfx110")) return &aql_gfx11_ops;
    // ... etc
    return NULL;
}
```

## Memory Management Strategy

### Allocation Patterns

1. **Static Tables**: Architecture information stored in read-only static tables
2. **Pre-Allocated Buffers**: Command buffers use pre-allocated storage when possible
3. **Callback-Based Allocation**: Flexible allocation via callbacks for special needs
4. **Zero-Copy Results**: Counter results written directly to target memory

### Kernel Module Considerations

```c
// Kernel-compatible allocation callback
static aql_result_t kernel_alloc_callback(void** ptr, size_t size,
                                         uint32_t flags, void* userdata) {
    if (flags & AQL_MEM_FLAG_COHERENT) {
        *ptr = dma_alloc_coherent(device, size, &dma_handle, GFP_KERNEL);
    } else {
        *ptr = kmalloc(size, GFP_KERNEL);
    }
    return *ptr ? AQL_SUCCESS : AQL_ERROR_NO_MEMORY;
}
```

## Error Handling Design

### Consistent Error Codes

All functions return `aql_result_t` enumeration:
- **AQL_SUCCESS**: Operation completed successfully
- **AQL_ERROR_INVALID_ARGUMENT**: Invalid input parameters
- **AQL_ERROR_BUFFER_TOO_SMALL**: Insufficient buffer space
- **AQL_ERROR_UNSUPPORTED_ARCH**: Architecture not supported
- etc.

### Error Context

The `aql_context_t` structure maintains error state:
```c
typedef struct {
    // ... other fields
    aql_result_t last_error;        // Last operation error
    char error_msg[256];            // Detailed error message
} aql_context_t;
```

### Validation Strategy

1. **Input Validation**: All public functions validate parameters
2. **State Validation**: Context state checked before operations
3. **Hardware Validation**: Register addresses validated against architecture tables
4. **Buffer Validation**: Buffer operations check bounds and alignment

## Debugging and Tracing

### Compile-Time Debug Control

```c
#ifdef AQL_DEBUG_TRACE
void aql_cmd_buffer_debug_print(const aql_cmd_buffer_t* buf, const char* prefix);
#else
#define aql_cmd_buffer_debug_print(buf, prefix) do {} while(0)
#endif
```

### Runtime Debug Flags

```c
#define AQL_DEBUG_COMMANDS     (1U << 0)  // Debug command generation
#define AQL_DEBUG_PACKETS      (1U << 1)  // Debug packet contents
#define AQL_DEBUG_REGISTERS    (1U << 2)  // Debug register programming
#define AQL_DEBUG_COUNTERS     (1U << 3)  // Debug counter assignment
```

### Debug Utilities

- **Command Tracing**: Every command logged with hex dump
- **Packet Validation**: PM4 packet structure validation
- **Register Validation**: Register address and value validation
- **Performance Tracking**: Operation timing and statistics

## Performance Characteristics

### Memory Layout Optimization

1. **Cache Line Alignment**: Critical structures aligned to cache boundaries
2. **Hot Data Grouping**: Frequently accessed fields grouped together
3. **Minimal Padding**: Structures packed efficiently
4. **Sequential Access**: Data layouts optimized for sequential access patterns

### Function Call Overhead

- **Static Dispatch**: Architecture functions resolved once
- **Inline Utilities**: Simple operations inlined
- **Batch Operations**: Multiple operations combined when possible
- **Zero Validation**: Release builds can skip validation

## Testing and Validation

### Unit Test Support

Each structure and function designed for independent testing:
```c
// Example unit test
void test_cmd_buffer_append() {
    uint32_t buffer_data[16];
    aql_cmd_buffer_t buf;

    aql_cmd_buffer_init(&buf, buffer_data, 16);
    assert(aql_cmd_buffer_append_dword(&buf, 0x12345678) == AQL_SUCCESS);
    assert(buf.used == 1);
    assert(buf.data[0] == 0x12345678);
}
```

### Integration Testing

Architecture operations tables can be tested independently:
```c
void test_gfx9_operations() {
    const aql_arch_ops_t* ops = &aql_gfx9_ops;
    uint32_t buffer_data[64];
    aql_cmd_buffer_t buf;

    aql_cmd_buffer_init(&buf, buffer_data, 64);
    assert(ops->build_write_config_reg(&buf, 0x30800, 0x12345678) == AQL_SUCCESS);
    // Validate generated packet format...
}
```

## Migration Benefits

### From C++ to C Conversion Benefits:

1. **Kernel Compatibility**: Can be used directly in Linux kernel modules
2. **Performance**: Eliminates virtual function overhead and C++ runtime
3. **Debugging**: Simpler debugging without C++ complexity
4. **Memory Safety**: Explicit memory management prevents leaks
5. **Portability**: Pure C code more portable across environments
6. **Predictability**: No hidden allocations or complex constructors

### Maintained Capabilities:

- ✅ All PM4 command generation functionality
- ✅ Support for all GPU architectures (GFX9-12)
- ✅ Complete counter management
- ✅ AQL packet population
- ✅ Debug tracing and validation
- ✅ Architecture abstraction
- ✅ Extensibility for new architectures

The C data structure design successfully maintains all functionality from the original C++ implementation while gaining the benefits of kernel module compatibility and improved performance characteristics.