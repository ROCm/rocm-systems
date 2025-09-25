# GFX12 Block Implementation TODO

## Analysis Summary

Based on comparison between perf-pmu-stub and aqlprofile implementations:

### Currently Implemented in perf-pmu-stub (gfx12_creator.c):
1. **CPC** (Command Processor C) - 2 counters, Global block
2. **SQ** (Shader Quotient) - 8 counters, SE/SA/WGP dimensions

### Missing Blocks Available in aqlprofile:

#### Global Blocks (Instance Count = 1, accessed globally):
- **CPF** - Command Processor F (2 counters, max event 4)
  - Registers: `regCPF_PERFCOUNTER0_SELECT` (0x3807), `regCPF_PERFCOUNTER0_LO` (0x300a)
- **CPG** - Command Processor G (2 counters, max event 30)
- **GCR** - Graphics Command Ring (2 counters, max event 151)
- **GRBM** - Graphics Register Bus Manager (2 counters, max event 51)
  - Registers: `regGRBM_PERFCOUNTER0_SELECT`, `regGRBM_PERFCOUNTER0_LO` (0x3040)
- **RLC** - RunList Controller (2 counters, max event 6)
- **SDMA** - System DMA (2 counters, max event 125, 2 instances)
- **CHA** - Cache Hierarchy A (4 counters, max event 25)
  - Registers: `regCHA_PERFCOUNTER0_LO` (0x3600), `regCHA_PERFCOUNTER0_HI` (0x3601)
- **CHC** - Cache Hierarchy C (4 counters, max event 94, 2 instances)
  - Registers: `regCHC_PERFCOUNTER0_LO` (0x33c0), `regCHC_PERFCOUNTER0_HI` (0x33c1)
- **GL2A** - Graphics L2 Cache A (4 counters, max event 114, 4 instances)
  - Registers: `regGL2A_PERFCOUNTER0_SELECT` (0x3b90), `regGL2A_PERFCOUNTER0_LO` (0x3390)
- **GL2C** - Graphics L2 Cache C (4 counters, max event 249, 16 instances)
  - Registers: `regGL2C_PERFCOUNTER0_SELECT` (0x3b80), `regGL2C_PERFCOUNTER0_LO` (0x3380)

#### SE (Shader Engine) Blocks:
- **SPI** - Shader Processor Input (6 counters, max event 318)
- **SQG** - Shader Quotient Graphics (8 counters, max event 45) - Different from current SQ
- **GRBMH** - GRBM per shader engine (2 counters, max event 25)
- **GCEA_SE** - GC Engine A per SE (2 counters, max event 32, 8 instances)
- **GC_UTCL1** - GC Unified TLB Cache L1 (4 counters, max event 71, 2 instances)

#### SA (Shader Array) Blocks:
- **GL1A** - Graphics L1 Cache A (4 counters, max event 21)
  - Registers: `regGL1A_PERFCOUNTER0_SELECT` (0x3dc0)
- **GL1C** - Graphics L1 Cache C (4 counters, max event 121, 4 instances)
  - Registers: `regGL1C_PERFCOUNTER0_SELECT` (0x3ba0), `regGL1C_PERFCOUNTER0_LO` (0x33a0)

#### WGP (Workgroup Processor) Blocks:
- **SQC/SQ** - Shader Quotient Compute (16 counters, max event 511, 32-bit only)
- **TA** - Texture Addressing (2 counters, max event 254, 2 instances)
  - Registers: `regTA_PERFCOUNTER0_SELECT` (0x3ac0), `regTA_PERFCOUNTER0_LO` (0x32c0)
- **TCP** - Texture Cache per Pipe (4 counters, max event 99, 2 instances)
  - Registers: `regTCP_PERFCOUNTER0_SELECT` (0x3b40), `regTCP_PERFCOUNTER0_LO` (0x3340)
- **TD** - Texture Data (2 counters, max event 271, 2 instances)
  - Registers: `regTD_PERFCOUNTER0_SELECT` (0x3b00), `regTD_PERFCOUNTER0_LO` (0x3300)

## Implementation Tasks Required:

### 1. Extend hardware_ip_block_t enum
Add missing block IDs to `aql_structures.h`:
```c
typedef enum {
    // Existing blocks
    HW_IP_BLOCK_CPC,
    HW_IP_BLOCK_SQ,

    // Missing blocks to add
    HW_IP_BLOCK_CPF,
    HW_IP_BLOCK_CPG,
    HW_IP_BLOCK_GCR,
    HW_IP_BLOCK_GRBM,
    HW_IP_BLOCK_RLC,
    HW_IP_BLOCK_SDMA,
    HW_IP_BLOCK_CHA,
    HW_IP_BLOCK_CHC,
    HW_IP_BLOCK_GL2A,
    HW_IP_BLOCK_GL2C,
    HW_IP_BLOCK_SPI,
    HW_IP_BLOCK_SQG,
    HW_IP_BLOCK_GRBMH,
    HW_IP_BLOCK_GCEA_SE,
    HW_IP_BLOCK_GC_UTCL1,
    HW_IP_BLOCK_GL1A,
    HW_IP_BLOCK_GL1C,
    HW_IP_BLOCK_SQC,
    HW_IP_BLOCK_TA,
    HW_IP_BLOCK_TCP,
    HW_IP_BLOCK_TD,
    HW_IP_BLOCK_LAST
} hardware_ip_block_t;
```

### 2. Add register offset definitions
Add all missing register offsets to `gfx12_creator.c`:
```c
// Example for CPF block
#define mmCPF_PERFCOUNTER0_SELECT    55623  // 0x3807 from offset.h
#define mmCPF_PERFCOUNTER0_LO        12298  // 0x300a from offset.h
#define mmCPF_PERFCOUNTER0_HI        12299  // 0x300b from offset.h
// ... and so on for all blocks
```

### 3. Implement block creation functions
Create functions similar to `create_gfx12_cpc_block()` for each missing block:
```c
static block_info_t* create_gfx12_cpf_block(void);
static block_info_t* create_gfx12_gl2c_block(void);
static block_info_t* create_gfx12_ta_block(void);
// ... etc
```

### 4. Update create_gfx12_arch()
Add instantiation and mapping of all new blocks to the architecture.

### 5. Add proper dimension configurations
Configure SE, SA, and WGP blocks based on GFX12 hardware layout:
- SE blocks: 4 shader engines
- SA blocks: 2 shader arrays per SE
- WGP blocks: 4 workgroup processors per SA

## Priority Blocks for Implementation:
1. **GL2C** - Critical L2 cache performance monitoring (16 instances)
2. **TA/TCP/TD** - Texture unit performance (essential for graphics workloads)
3. **SPI** - Shader processor input coordination
4. **GL1A/GL1C** - L1 cache monitoring for memory hierarchy analysis
5. **GRBM** - Overall graphics command processing
6. **SDMA** - System DMA for data movement analysis

## Files to Modify:
- `perf-pmu-stub/src/aql_c/aql_structures.h` - Add enum values
- `perf-pmu-stub/src/aql_c/gfx12_creator.c` - Add register definitions and block creators
- Test files - Verify new blocks work correctly

## Reference Files:
- `projects/aqlprofile/gfxip/gfx12/gfx12_block_info.h` - Block parameters
- `projects/aqlprofile/gfxip/gfx12/gfx12_block_table.h` - Register definitions
- `projects/aqlprofile/linux/registers/gc/gc_12_0_0_offset.h` - Register offsets