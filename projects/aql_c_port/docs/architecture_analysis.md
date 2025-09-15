# Architecture-Specific Builder Analysis

## Overview

Analysis of GPU architecture-specific PM4 command builders (GFX9, GFX10, GFX11, GFX12) to understand differences and commonalities for the C port.

## Architecture Classes Identified

- `Gfx9CmdBuilder` - GFX9 (Vega) architecture
- `Gfx10CmdBuilder` - GFX10 (RDNA1) architecture
- `Gfx11CmdBuilder` - GFX11 (RDNA2) architecture
- `Gfx12CmdBuilder` - GFX12 (RDNA3) architecture

## Common Patterns

### 1. Packet Header Generation
All architectures use the same pattern for Type 3 PM4 packet headers:

```cpp
static uint32_t MakePacket3Header(uint32_t opcode, size_t packet_size) {
    uint32_t count = packet_size / sizeof(uint32_t) - 2;
    uint32_t header = PACKET3(opcode, count);
    return header;
}
```

**C Port Design:**
```c
static inline uint32_t aql_make_packet3_header(uint32_t opcode, size_t packet_size) {
    uint32_t count = packet_size / sizeof(uint32_t) - 2;
    return PACKET3(opcode, count);
}
```

### 2. Register Space Classification
Both GFX9 and GFX11 implement similar register space validation:

```cpp
static constexpr bool IsPrivilegedConfigReg(uint32_t addr) {
    return ((addr >= CONFIG_SPACE_START) && (addr <= CONFIG_SPACE_END));
}

static constexpr bool IsUConfigReg(uint32_t addr) {  // GFX9 only
    return (addr >= UCONFIG_SPACE_START) && (addr <= UCONFIG_SPACE_END);
}
```

**C Port Design:**
```c
typedef struct {
    uint32_t config_space_start;
    uint32_t config_space_end;
    uint32_t uconfig_space_start;  // Only for GFX9
    uint32_t uconfig_space_end;    // Only for GFX9
    bool has_uconfig_space;
} aql_register_spaces_t;

static inline bool aql_is_privileged_config_reg(const aql_register_spaces_t* spaces, uint32_t addr) {
    return (addr >= spaces->config_space_start) && (addr <= spaces->config_space_end);
}
```

## Architecture-Specific Differences

### 1. PRED_EXEC Support
**GFX9 has PRED_EXEC packet support:**
```cpp
void BuildPredExecPacket(CmdBuffer* cmdbuf, uint32_t xcc_select = 0, uint32_t exec_count = 0) {
    uint32_t header = MakePacket3Header(PACKET3_PRED_EXEC, 2 * sizeof(uint32_t));
    uint32_t virtualxccid_select = 1 << xcc_select;
    uint32_t dword2 = PACKET3_PRED_EXEC__EXEC_COUNT(exec_count) |
                      PACKET3_PRED_EXEC__VIRTUAL_XCC_ID_SELECT(virtualxccid_select);
    uint32_t pm4_mec_pred_exec_cmd[2] = {header, dword2};
    APPEND_COMMAND_WRAPPER(cmdbuf, pm4_mec_pred_exec_cmd);
}
```

**GFX11+ does not have PRED_EXEC** - this is a key architectural difference.

### 2. Cache Flush Differences
**GFX9 Cache Flush (7 DWords):**
```cpp
uint32_t header = MakePacket3Header(PACKET3_ACQUIRE_MEM, 7 * sizeof(uint32_t));
// ... 7 dword packet
```

**GFX11 Cache Flush (8 DWords):**
```cpp
uint32_t header = MakePacket3Header(PACKET3_ACQUIRE_MEM, 8 * sizeof(uint32_t));
// ... 8 dword packet with additional GCR_CNTL field
uint32_t dword8 = PACKET3_ACQUIRE_MEM__GCR_CNTL(
    ((GCR_CNTL__SEQ_FORWARD & GCR_CNTL__SEQ_MASK) | GCR_CNTL__GL2_WB_MASK));
```

### 3. Barrier Commands
All architectures implement similar barrier/wait idle commands using EVENT_WRITE:

```cpp
void BuildBarrierCommand(CmdBuffer* cmdBuf) {
    uint32_t header = MakePacket3Header(PACKET3_EVENT_WRITE, 2 * sizeof(uint32_t));
    uint32_t dword2 = PACKET3_EVENT_WRITE__EVENT_TYPE(CS_PARTIAL_FLUSH) |
                      PACKET3_EVENT_WRITE__EVENT_INDEX(PACKET3_EVENT_WRITE__EVENT_INDEX__CS_PARTIAL_FLUSH);
    uint32_t pm4mec_event_write_cmd[2] = {header, dword2};
    APPEND_COMMAND_WRAPPER(cmdBuf, pm4mec_event_write_cmd);
}
```

## C Port Architecture Function Table Design

```c
typedef struct {
    const char* arch_name;
    uint32_t gfx_version;

    // Architecture capabilities
    bool has_pred_exec;
    bool has_uconfig_space;
    bool has_extended_cache_flush;

    // Register space definitions
    aql_register_spaces_t register_spaces;

    // Function pointers for architecture-specific operations
    void (*build_pred_exec)(aql_cmd_buffer_t* buf, uint32_t xcc_select, uint32_t exec_count);
    void (*build_cache_flush)(aql_cmd_buffer_t* buf, size_t addr, size_t size);
    void (*build_barrier)(aql_cmd_buffer_t* buf);
    void (*build_write_uconfig_reg)(aql_cmd_buffer_t* buf, uint32_t addr, uint32_t value);
    void (*build_write_sh_reg)(aql_cmd_buffer_t* buf, uint32_t addr, uint32_t value);
    void (*build_copy_reg_data)(aql_cmd_buffer_t* buf, uint32_t src_addr, void* dst_addr, uint32_t size, bool wait);
    void (*build_indirect_buffer)(aql_cmd_buffer_t* buf, const void* cmd_addr, size_t cmd_size);
    void (*build_wait_reg_mem)(aql_cmd_buffer_t* buf, bool mem_space, uint64_t wait_addr, bool func_eq, uint32_t mask, uint32_t wait_val);
    void (*build_nop)(aql_cmd_buffer_t* buf, uint32_t num_dwords);

    // Architecture-specific register tables
    const aql_register_table_t* register_table;
} aql_arch_ops_t;
```

## Implementation Strategy

### 1. Common Command Functions
Implement shared functions that work across all architectures:
- Packet header generation
- Basic register writes
- NOP packets
- Basic barrier commands

### 2. Architecture-Specific Functions
Implement separate functions for architecture differences:
- Cache flush variants (7 vs 8 dword)
- PRED_EXEC (GFX9 only)
- Extended register spaces

### 3. Runtime Architecture Detection
```c
const aql_arch_ops_t* aql_detect_architecture(const char* gfx_name) {
    if (strstr(gfx_name, "gfx942") || strstr(gfx_name, "gfx94")) return &gfx9_ops;
    if (strstr(gfx_name, "gfx103") || strstr(gfx_name, "gfx10")) return &gfx10_ops;
    if (strstr(gfx_name, "gfx110") || strstr(gfx_name, "gfx11")) return &gfx11_ops;
    if (strstr(gfx_name, "gfx120") || strstr(gfx_name, "gfx12")) return &gfx12_ops;
    return NULL;
}
```

## Key Insights

1. **High Commonality**: Most packet structures are identical across architectures
2. **Version-Specific Features**: PRED_EXEC is GFX9-specific, newer architectures have extended cache control
3. **Register Space Evolution**: UConfig space is GFX9-specific
4. **Packet Size Differences**: Cache flush packets grew from 7 to 8 dwords in newer architectures
5. **Function Table Approach**: C function pointers can effectively replace C++ virtual functions

## Next Steps

1. Analyze packet formats in detail
2. Document register programming sequences
3. Design counter management structures
4. Implement architecture-specific function tables