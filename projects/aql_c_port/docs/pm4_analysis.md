# PM4 Packet Generation Analysis

## Overview

This document analyzes the PM4 (Processor for Multimedia 4) packet generation logic from the C++ AQLProfile library to guide the C port implementation.

## Core PM4 Command Types

Based on analysis of `src/pm4/cmd_builder.h`, the following PM4 commands are essential:

### 1. Register Programming Commands
- **WriteUConfigRegPacket**: Program GPU configuration registers
- **WriteShRegPacket**: Write to shader registers
- **WriteConfigRegPacket**: Write to configuration registers

### 2. Data Movement Commands
- **CopyRegDataPacket**: Copy data from register to memory
- **IndirectBufferCmd**: Execute command buffer indirectly

### 3. Synchronization Commands
- **WaitIdle**: Wait for GPU to become idle
- **WaitRegMem**: Wait for register/memory condition
- **MutexAcquire/Release**: Hardware mutex operations

### 4. Utility Commands
- **NopPacket**: No-operation padding
- **ThreadTraceEventFinish**: Thread trace completion
- **PrimeL2**: Prime L2 cache pages

## CmdBuffer Data Structure

The C++ implementation uses a vector-based buffer:

```cpp
class CmdBuffer {
private:
    std::vector<uint32_t> data_;
public:
    void Append(T&& packet);
    size_t Size() const;
    size_t DwSize() const;
    const void* Data() const;
    void Clear();
};
```

### C Port Design:
```c
typedef struct {
    uint32_t* data;
    size_t capacity;    // Total allocated size in dwords
    size_t used;        // Used size in dwords
} aql_cmd_buffer_t;
```

## Architecture Abstraction

The C++ design uses virtual functions for architecture-specific behavior:

```cpp
class CmdBuilder {
public:
    virtual void BuildWriteUConfigRegPacket(CmdBuffer* cmdbuf, uint32_t addr, uint32_t value) = 0;
    virtual void BuildWriteShRegPacket(CmdBuffer* cmdbuf, uint32_t addr, uint32_t value) = 0;
    // ... other virtual functions
};
```

### C Port Design - Function Pointer Tables:
```c
typedef struct {
    void (*build_write_uconfig_reg)(aql_cmd_buffer_t* buf, uint32_t addr, uint32_t value);
    void (*build_write_sh_reg)(aql_cmd_buffer_t* buf, uint32_t addr, uint32_t value);
    void (*build_copy_reg_data)(aql_cmd_buffer_t* buf, uint32_t src_addr, void* dst_addr, uint32_t size, bool wait);
    void (*build_indirect_buffer)(aql_cmd_buffer_t* buf, const void* cmd_addr, size_t cmd_size);
    void (*build_wait_idle)(aql_cmd_buffer_t* buf);
    void (*build_wait_reg_mem)(aql_cmd_buffer_t* buf, bool mem_space, uint64_t wait_addr, bool func_eq, uint32_t mask, uint32_t wait_val);
    void (*build_nop)(aql_cmd_buffer_t* buf, uint32_t num_dwords);
} aql_cmd_builder_ops_t;
```

## Register Offset Management

The C++ code uses a register offset table:

```cpp
const reg_base_offset_table* const ip_offset_table;
uint32_t get_addr(const Register& reg) {
    return (*ip_offset_table)[reg];
}
```

### C Port Design:
```c
typedef struct {
    uint32_t base_addr;
    uint32_t offset;
} aql_register_addr_t;

typedef struct {
    const aql_register_addr_t* registers;
    size_t count;
} aql_register_table_t;
```

## Debug and Tracing

The C++ code includes extensive debug tracing:

```cpp
#if defined(DEBUG_TRACE)
template <typename... Ts>
void PrintCommand(const char* function_name, Ts&&... packets) {
    // Print packet contents in hex format
}
#endif
```

### C Port Design:
```c
#ifdef AQL_DEBUG_TRACE
void aql_debug_print_command(const char* func_name, const uint32_t* data, size_t dwords);
#define AQL_DEBUG_CMD(buf, func) aql_debug_print_command(func, (buf)->data, (buf)->used)
#else
#define AQL_DEBUG_CMD(buf, func) do {} while(0)
#endif
```

## Utility Functions

Important utility functions to port:

```cpp
constexpr uint32_t Low32(uint64_t u) { return static_cast<uint32_t>(u); }
constexpr uint32_t High32(uint64_t u) { return static_cast<uint32_t>(u >> 32); }
inline uint32_t PtrLow32(const void* p) { return Low32(reinterpret_cast<uintptr_t>(p)); }
```

### C Port:
```c
static inline uint32_t aql_low32(uint64_t u) { return (uint32_t)(u); }
static inline uint32_t aql_high32(uint64_t u) { return (uint32_t)(u >> 32); }
static inline uint32_t aql_ptr_low32(const void* p) { return aql_low32((uintptr_t)p); }
static inline uint32_t aql_ptr_high32(const void* p) { return aql_high32((uintptr_t)p); }
```

## Key Insights for C Port

1. **Buffer Management**: Replace std::vector with pre-allocated buffers for kernel compatibility
2. **Architecture Abstraction**: Use function pointer tables instead of virtual functions
3. **Error Handling**: Replace exceptions with return codes
4. **Memory Safety**: Add bounds checking for all buffer operations
5. **Thread Safety**: Use kernel synchronization primitives where needed
6. **Debug Support**: Maintain debug tracing capability for development

## Next Steps

1. Analyze architecture-specific builders (GFX9, GFX10, GFX11, GFX12)
2. Document packet formats and register programming sequences
3. Design C data structures for counter management
4. Implement core command generation functions