# Register Programming Sequences Analysis

## Overview

This document analyzes the register programming sequences used across different GPU architectures in the AQLProfile system, mapping counter blocks, register spaces, and programming patterns.

## Register Space Classification

### Common Register Spaces Across Architectures

1. **CONFIG_SPACE**: Privileged configuration registers
2. **UCONFIG_SPACE**: User-accessible configuration registers (GFX9 only)
3. **MMREG_SPACE**: Memory-mapped registers for data movement

### Register Space Ranges by Architecture

#### GFX9 (Vega)
- **CONFIG_SPACE**: `CONFIG_SPACE_START` to `CONFIG_SPACE_END`
- **UCONFIG_SPACE**: `UCONFIG_SPACE_START` to `UCONFIG_SPACE_END` (unique to GFX9)
- Supports both privileged and user-accessible configuration registers

#### GFX11+ (RDNA2/3)
- **CONFIG_SPACE**: `CONFIG_SPACE_START` to `CONFIG_SPACE_END`
- **No UCONFIG_SPACE**: User configuration registers removed in RDNA architectures
- Simplified register access model

## Counter Block Evolution

### GFX9 Counter Blocks (Vega)
Complete set of graphics and compute counter blocks:

```c
enum aql_gfx9_counter_blocks {
    AQL_CB_BLOCK = 0,           // Color Buffer
    AQL_CPC_BLOCK,              // Command Processor Compute
    AQL_CPF_BLOCK,              // Command Processor Fetch
    AQL_CPG_BLOCK,              // Command Processor Graphics
    AQL_DB_BLOCK,               // Depth Buffer
    AQL_GDS_BLOCK,              // Global Data Share
    AQL_GRBM_BLOCK,             // Graphics Register Bus Manager
    AQL_GRBM_SE_BLOCK,          // GRBM Shader Engine
    AQL_IA_BLOCK,               // Input Assembler
    AQL_PA_SC_BLOCK,            // Primitive Assembly Setup/Clipping
    AQL_PA_SU_BLOCK,            // Primitive Assembly Setup Unit
    AQL_SPI_BLOCK,              // Shader Processor Input
    AQL_SQ_BLOCK,               // Shader Quad
    AQL_SQ_GS_BLOCK,            // Shader Quad Geometry Shader
    AQL_SQ_VS_BLOCK,            // Shader Quad Vertex Shader
    AQL_SQ_PS_BLOCK,            // Shader Quad Pixel Shader
    AQL_SQ_HS_BLOCK,            // Shader Quad Hull Shader
    AQL_SQ_CS_BLOCK,            // Shader Quad Compute Shader
    AQL_SX_BLOCK,               // Shader Export
    AQL_TA_BLOCK,               // Texture Addresser
    AQL_TCA_BLOCK,              // Texture Cache Arbiter
    AQL_TCC_BLOCK,              // Texture Cache Controller
    AQL_TCP_BLOCK,              // Texture Cache Per-pipe
    AQL_TCS_BLOCK,              // Texture Cache System
    AQL_TD_BLOCK,               // Texture Data
    AQL_VGT_BLOCK,              // Vertex Grouper/Tessellator
    AQL_WD_BLOCK,               // Workgroup Distributor

    // Memory Controller blocks
    AQL_GCEA_BLOCK,             // Graphics Command Engine Arbiter
    AQL_ATC_BLOCK,              // Address Translation Cache
    AQL_ATC_L2_BLOCK,           // ATC L2 Cache
    AQL_MC_VM_L2_BLOCK,         // Memory Controller VM L2
    AQL_RPB_BLOCK,              // Render Backend
    AQL_RMI_BLOCK,              // Read/Write Memory Interface

    // System blocks
    AQL_SDMA_BLOCK,             // System DMA
    AQL_UMC_BLOCK,              // Unified Memory Controller
    AQL_IOMMU_V2_BLOCK,         // IOMMU Version 2
};
```

### GFX11 Counter Blocks (RDNA2)
Evolved architecture with new cache hierarchy:

```c
enum aql_gfx11_counter_blocks {
    // Core graphics blocks (inherited from GFX9)
    AQL_CB_BLOCK = 0,
    AQL_CPC_BLOCK,
    AQL_CPF_BLOCK,
    AQL_CPG_BLOCK,
    AQL_DB_BLOCK,
    AQL_GDS_BLOCK,
    AQL_GRBM_BLOCK,
    AQL_GRBM_SE_BLOCK,
    AQL_SPI_BLOCK,
    AQL_SQ_BLOCK,
    AQL_SQ_GS_BLOCK,
    AQL_SQ_PS_BLOCK,
    AQL_SQ_HS_BLOCK,
    AQL_SQ_CS_BLOCK,
    AQL_SX_BLOCK,
    AQL_TA_BLOCK,
    AQL_TD_BLOCK,

    // Removed blocks (commented out in GFX11):
    // AQL_IA_BLOCK         - Input Assembler removed
    // AQL_PA_SC_BLOCK      - PA blocks restructured
    // AQL_PA_SU_BLOCK
    // AQL_SQ_VS_BLOCK      - Vertex shader block removed
    // AQL_TCA_BLOCK        - Texture cache arbiter removed
    // AQL_TCC_BLOCK        - Texture cache controller removed
    // AQL_TCS_BLOCK        - Texture cache system removed
    // AQL_VGT_BLOCK        - Vertex grouper/tessellator removed
    // AQL_WD_BLOCK         - Workgroup distributor removed

    // New RDNA2 cache blocks
    AQL_GL1A_BLOCK,             // GL1 Texture Cache Arbiter
    AQL_GL1C_BLOCK,             // GL1 Texture Cache Controller
    AQL_GL2A_BLOCK,             // GL2 Cache Arbiter
    AQL_GL2C_BLOCK,             // GL2 Cache Controller
    AQL_GCR_BLOCK,              // Graphics Command Ring
    AQL_GUS_BLOCK,              // Graphics Unified Scheduler

    // Enhanced system blocks
    AQL_SDMA0_BLOCK,            // SDMA Engine 0
    AQL_SDMA1_BLOCK,            // SDMA Engine 1 (dual SDMA)
    AQL_TCP_BLOCK,              // Texture Cache Per-pipe (re-added)
};
```

## Register Programming Patterns

### PM4 Packet Types for Register Access

#### PACKET3_WRITE_DATA
Used for writing data to registers or memory:

```c
// Packet structure for register writes
typedef struct {
    uint32_t header;            // PACKET3(PACKET3_WRITE_DATA, count)
    uint32_t control;           // Destination selection and control
    uint32_t dst_addr_lo;       // Lower 32 bits of destination
    uint32_t dst_addr_hi;       // Upper 32 bits of destination (if needed)
    uint32_t data[];            // Data to write
} aql_write_data_packet_t;

// Destination selection values
#define AQL_DST_SEL_MMREG      0    // Memory-mapped register
#define AQL_DST_SEL_TC_L2      2    // Texture cache L2
#define AQL_DST_SEL_GDS        3    // Global Data Share
```

#### Control Fields for Register Programming

```c
// Write data control field construction
static inline uint32_t aql_make_write_data_control(uint32_t dst_sel, bool wait_confirm) {
    uint32_t control = 0;
    control |= (dst_sel & 0x7) << 0;                    // Destination selection
    control |= (wait_confirm ? 1 : 0) << 20;           // Wait for write confirmation
    return control;
}
```

### Architecture-Specific Register Programming

#### GFX9 Register Programming Sequence
```c
// Example: Program performance counter register on GFX9
static void aql_gfx9_program_counter_reg(aql_cmd_buffer_t* buf, uint32_t block_reg,
                                        uint32_t counter_id, uint32_t event_select) {
    uint32_t reg_addr = block_reg + (counter_id * 4);  // Counter registers are 4 bytes apart

    // Check if it's a privileged register
    if (aql_is_privileged_config_reg(&gfx9_register_spaces, reg_addr)) {
        // Use CONFIG register write
        aql_gfx9_build_write_config_reg(buf, reg_addr, event_select);
    } else if (aql_gfx9_is_uconfig_reg(reg_addr)) {
        // Use UCONFIG register write (GFX9 specific)
        aql_gfx9_build_write_uconfig_reg(buf, reg_addr, event_select);
    } else {
        // Use SH register write
        aql_gfx9_build_write_sh_reg(buf, reg_addr, event_select);
    }
}
```

#### GFX11+ Register Programming Sequence
```c
// Example: Program performance counter register on GFX11+
static void aql_gfx11_program_counter_reg(aql_cmd_buffer_t* buf, uint32_t block_reg,
                                         uint32_t counter_id, uint32_t event_select) {
    uint32_t reg_addr = block_reg + (counter_id * 4);

    // GFX11+ simplified: no UCONFIG space
    if (aql_is_privileged_config_reg(&gfx11_register_spaces, reg_addr)) {
        // Use CONFIG register write
        aql_gfx11_build_write_config_reg(buf, reg_addr, event_select);
    } else {
        // Use SH register write
        aql_gfx11_build_write_sh_reg(buf, reg_addr, event_select);
    }
}
```

## Counter Block Register Layouts

### Standard Counter Block Register Pattern
Most counter blocks follow this register layout:

```c
typedef struct {
    uint32_t base_addr;         // Base register address for the block
    uint32_t counter_count;     // Number of counters in this block
    uint32_t counter_stride;    // Bytes between counter registers (usually 4)
    uint32_t select_reg;        // Counter select/event register
    uint32_t enable_reg;        // Counter enable register
    uint32_t result_reg;        // Counter result register
} aql_counter_block_regs_t;
```

### Block-Specific Register Variations

#### SQ (Shader Quad) Block
```c
// SQ block has special handling for accumulation counters
typedef struct {
    aql_counter_block_regs_t common;
    uint32_t accum_select_reg;      // Accumulation mode select
    uint32_t accum_enable_reg;      // Accumulation enable
    bool supports_accumulation;     // Architecture capability
} aql_sq_block_regs_t;
```

#### Memory Controller Blocks
```c
// Memory controller blocks may have different addressing
typedef struct {
    aql_counter_block_regs_t common;
    uint32_t channel_count;         // Number of memory channels
    uint32_t channel_stride;        // Address stride between channels
    bool uses_smn_addressing;       // Uses System Management Network addressing
} aql_mc_block_regs_t;
```

## C Port Data Structures

### Architecture Register Table
```c
typedef struct {
    const char* arch_name;
    uint32_t gfx_version;

    // Register space definitions
    uint32_t config_space_start;
    uint32_t config_space_end;
    uint32_t uconfig_space_start;   // GFX9 only, 0 for others
    uint32_t uconfig_space_end;     // GFX9 only, 0 for others

    // Counter block information
    const aql_counter_block_regs_t* counter_blocks;
    uint32_t counter_block_count;

    // Architecture capabilities
    bool has_uconfig_space;
    bool has_dual_sdma;
    bool has_gl_cache_hierarchy;

} aql_arch_register_info_t;
```

### Register Programming Function Table
```c
typedef struct {
    void (*write_config_reg)(aql_cmd_buffer_t* buf, uint32_t addr, uint32_t value);
    void (*write_uconfig_reg)(aql_cmd_buffer_t* buf, uint32_t addr, uint32_t value);
    void (*write_sh_reg)(aql_cmd_buffer_t* buf, uint32_t addr, uint32_t value);
    void (*copy_reg_data)(aql_cmd_buffer_t* buf, uint32_t src_addr, void* dst, uint32_t size);
    bool (*is_valid_counter_reg)(uint32_t block_id, uint32_t reg_addr);
} aql_register_ops_t;
```

## Key Insights for C Implementation

1. **Architecture Evolution**: Counter blocks were removed/added between generations
2. **Register Space Simplification**: UCONFIG space removed in RDNA architectures
3. **Cache Hierarchy Changes**: New GL1/GL2 cache blocks in RDNA
4. **Dual SDMA**: RDNA architectures have multiple SDMA engines
5. **Programming Consistency**: Register programming patterns are consistent within architectures
6. **Address Calculation**: Simple arithmetic for counter register addressing

## Implementation Strategy

1. **Static Tables**: Define register information as static const tables per architecture
2. **Runtime Detection**: Select appropriate table based on detected GPU architecture
3. **Unified Interface**: Provide common programming interface that handles architecture differences
4. **Validation**: Implement register address validation for security
5. **Error Handling**: Return error codes for invalid register accesses

## Next Steps

1. Design comprehensive C data structures
2. Create architecture function pointer tables
3. Implement core PM4 command generation
4. Add register validation and security checks