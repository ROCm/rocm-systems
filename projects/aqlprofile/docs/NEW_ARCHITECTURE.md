# AQLProfile New Architecture Documentation

## Overview

This document describes the new architecture abstraction system for AQLProfile, designed to eliminate hardcoded GPU architecture dependencies and make it easy to add new GFX architectures.

## Problems Solved

### Before Refactoring

The original AQLProfile implementation had several architectural issues:

1. **Hardcoded Architecture Logic**: MI300-specific code scattered in builders
2. **Factory Explosion**: 9 separate factory files with duplicated patterns
3. **Template Coupling**: Compile-time templates preventing runtime flexibility
4. **Block Table Duplication**: Static tables copy-pasted per architecture
5. **Register Fragmentation**: Hardware addresses spread across def files

**Adding a new GFX architecture required modifying 15+ files!**

### After Refactoring

The new architecture uses clean abstractions:

1. **Single Responsibility**: Each class has one job
2. **Strategy Pattern**: Architecture-specific behavior encapsulated
3. **Runtime Polymorphism**: Flexible architecture selection
4. **Centralized Configuration**: Hardware config in data structures
5. **Register Schema**: Type-safe register definitions

**Adding a new GFX architecture now requires 2 files (hpp + cpp)!**

## Core Abstractions

### 1. HardwareConfig

Data class holding GPU topology and capabilities:

```cpp
struct HardwareConfig {
  std::string gfxip;           // "gfx90a", "gfx1100", etc.
  std::string name;            // "MI200", "Navi31", etc.
  
  // Topology
  uint32_t se_count;           // Shader Engines
  uint32_t sa_per_se_count;    // Shader Arrays per SE
  uint32_t cu_count;           // Compute Units
  uint32_t wgp_count;          // Work Group Processors
  
  // Multi-die
  uint32_t xcc_count;          // XCC count
  uint32_t aid_count;          // AID count (MI300+)
  
  // Capabilities
  bool supports_pmc;           // Performance counters
  bool supports_spm;           // Streaming PM
  bool supports_sqtt;          // Thread trace
  bool supports_concurrent;    // Concurrent mode
  
  // Architecture flags
  bool has_aid_aware_counters; // MI300-specific
  bool has_spm_core1;          // MI100/MI200
  // ... more flags
};
```

**Benefits:**
- No more `if (xcc_number_ > 1)` conditionals
- All config in one place
- Easy to serialize/deserialize

### 2. RegisterSchema

Type-safe register definitions:

```cpp
enum class RegisterId {
  GRBM_GFX_INDEX,
  CP_PERFMON_CNTL,
  SQ_PERFCOUNTER_CTRL,
  SQTT_BUF_BASE,
  // ... more registers
};

class RegisterSchema {
  void DefineRegister(RegisterId id, uint32_t offset);
  uint32_t GetOffset(RegisterId id) const;
  bool HasRegister(RegisterId id) const;
  // ... more methods
};
```

**Benefits:**
- No hardcoded register addresses in code
- Per-architecture overrides supported
- Compile-time type safety

### 3. HardwareArchitecture

Abstract interface for all GPU architectures:

```cpp
class HardwareArchitecture {
  virtual const HardwareConfig& GetConfig() const = 0;
  virtual const RegisterSchema& GetRegisterSchema() const = 0;
  virtual const GpuBlockInfo* GetBlockInfo(uint32_t block_id) const = 0;
  virtual pm4_builder::CmdBuilder* CreateCmdBuilder() const = 0;
  
  // Architecture queries
  virtual bool IsGFX9() const { return false; }
  virtual bool IsGFX10() const { return false; }
  // ... more queries
};
```

**Benefits:**
- Clean interface for all architectures
- Easy to mock for testing
- Runtime polymorphism

### 4. ArchitectureRegistry

Singleton registry for architecture lookup:

```cpp
auto& registry = ArchitectureRegistry::Instance();

// Register
registry.Register("gfx90a", std::make_unique<Gfx9Architecture>(info));

// Lookup (with prefix matching)
const auto* arch = registry.Lookup("gfx90a:sramecc+:xnack-");
```

**Benefits:**
- Centralized architecture management
- Prefix matching for gfxip variations
- Thread-safe

## Architecture Implementations

### Implemented Architectures

1. **Gfx9Architecture** - Vega series baseline
   - gfx900, gfx902, gfx906
   - Base class for MI series

2. **Mi100Architecture** - CDNA 1 (gfx908)
   - Accumulator regs: low=1, hi=158; SPM delay max=0x34
   - Extends Gfx9Architecture

3. **Mi200Architecture** - CDNA 2 (gfx90a)
   - Accumulator regs: low=1, hi=185; SPM delay max=0x3e
   - Extends Gfx9Architecture

4. **Mi300Architecture** - CDNA 3 (gfx940/941/942)
   - Accumulator regs: low=1, hi=184; SPM delay max=0x27
   - 4 AIDs, per-agent XCC count, AID-aware counter routing
   - Extends Gfx9Architecture

5. **Mi350Architecture** - CDNA 3.5 (gfx950)
   - Accumulator regs: low=1, hi=200; SPM delay max=0x33
   - Inherits multi-XCC/AID topology from Mi300Architecture
   - Extends Mi300Architecture

6. **Gfx10Architecture** - RDNA 1
   - gfx1010, gfx1011, gfx1012, gfx1030+
   - Introduces WGP concept

7. **Gfx11Architecture** - RDNA 3
   - gfx1100, gfx1101, gfx1102, gfx1103
   - No SPM support

8. **Gfx12Architecture** - RDNA 4 (gfx1200/gfx1201)
   - Generic GFX12 baseline; compiled with GFX12_VARIANT_1200

9. **Mi450Architecture** - CDNA 4 (gfx1250)
   - Compiled in a separate translation unit with `GFX12_VARIANT=0x1250`
   - gfx1250-specific block table (GCEA_SE, GL2A/GL2C, GC_CANE, ATCL2, etc.)
   - Extends Gfx12Architecture

### Architecture Hierarchy

```
HardwareArchitecture (abstract)
├── Gfx9Architecture
│   ├── Mi100Architecture (extends Gfx9)
│   ├── Mi200Architecture (extends Gfx9)
│   └── Mi300Architecture (extends Gfx9)
│       └── Mi350Architecture (extends Mi300)
├── Gfx10Architecture
├── Gfx11Architecture
│   └── Gfx115xArchitecture (extends Gfx11)
└── Gfx12Architecture
    └── Mi450Architecture (extends Gfx12, gfx1250)
```

## Usage Examples

### Query Architecture Capabilities

```cpp
#include "core/architecture_registry.hpp"

auto& registry = ArchitectureRegistry::Instance();
const auto* arch = registry.Lookup("gfx90a");

if (arch) {
  const auto& config = arch->GetConfig();
  std::cout << "SEs: " << config.se_count << "\n";
  std::cout << "PMC: " << (config.supports_pmc ? "Yes" : "No") << "\n";
  std::cout << "Multi-XCC: " << (config.IsMultiXCC() ? "Yes" : "No") << "\n";
}
```

### Use Register Schema

```cpp
const auto& schema = arch->GetRegisterSchema();

if (schema.HasRegister(RegisterId::GRBM_GFX_INDEX)) {
  uint32_t offset = schema.GetOffset(RegisterId::GRBM_GFX_INDEX);
  // Write to register at offset
}
```

### Query Block Information

```cpp
uint32_t sq_block_id = arch->FindBlockByName("SQ");
if (sq_block_id != UINT32_MAX) {
  const GpuBlockInfo* block = arch->GetBlockInfo(sq_block_id);
  size_t buffer_size = arch->GetBytesNeededForBlock(sq_block_id);
  
  std::cout << "SQ counters: " << block->counter_count << "\n";
  std::cout << "Buffer needed: " << buffer_size << " bytes\n";
}
```

### Create Command Builder

```cpp
pm4_builder::CmdBuilder* cmd_builder = arch->CreateCmdBuilder();
// Use cmd_builder for PM4 packet generation
delete cmd_builder;
```

## Adding a New Architecture

To add support for a new GFX architecture (e.g., GFX13):

### Step 1: Create Header File

`src/core/architectures/gfx13_architecture.hpp`:

```cpp
class Gfx13Architecture : public HardwareArchitecture {
 public:
  explicit Gfx13Architecture(const AgentInfo* agent_info);

  bool IsGFX13() const override { return true; }
  uint32_t GetAccumLowID() const override { return 1; }
  uint32_t GetAccumHiID() const override { return 1; }

 protected:
  void InitializeConfig(const AgentInfo* agent_info) override;
  void InitializeRegisterSchema() override;
  void InitializeBlockTable() override;
};
```

### Step 2: Create Implementation File

`src/core/architectures/gfx13_architecture.cpp`:

```cpp
#include "def/gfx13_def.h"  // provides gfx13_cntx_prim and block info objects

Gfx13Architecture::Gfx13Architecture(const AgentInfo* agent_info) {
  InitializeConfig(agent_info);
  InitializeRegisterSchema();
  InitializeBlockTable();
}

void Gfx13Architecture::InitializeConfig(const AgentInfo* agent_info) {
  config_.gfxip = agent_info->gfxip;
  config_.name  = agent_info->name;
  config_.se_count         = agent_info->se_num;
  config_.sa_per_se_count  = agent_info->shader_arrays_per_se;
  config_.cu_count         = agent_info->cu_num;
  config_.xcc_count        = 1;
}

void Gfx13Architecture::InitializeRegisterSchema() {
  // Populate from gfx13_cntx_prim addresses
  using P = gfx13_cntx_prim;
  register_schema_.DefineRegister(RegisterId::GRBM_GFX_INDEX,     P::GRBM_GFX_INDEX_offset);
  register_schema_.DefineRegister(RegisterId::CP_PERFMON_CNTL,     P::CP_PERFMON_CNTL_offset);
  // ... more registers
}

void Gfx13Architecture::InitializeBlockTable() {
  // Build table from gfx13 block info objects declared in gfx13_def.h
  static const GpuBlockInfo* table[AQLPROFILE_BLOCKS_NUMBER] = {};
  // table[AQLPROFILE_BLOCK_NAME_FOO] = &FooCounterBlockInfo;
  block_table_ = table;
  block_count_ = AQLPROFILE_BLOCKS_NUMBER;
}
```

If gfx13 requires a separate `GFX13_VARIANT` compile-time macro (like gfx1250), split the
block-table initialisation into a separate `.cpp` compiled with that define — see
`mi450_architecture.cpp` as the pattern.

### Step 3: Wire into the dispatch table

In `src/core/architecture_init.cpp`, add a prefix entry and a creation branch:

```cpp
// In GetGfxIpPrefix() — more-specific prefixes must precede generic ones:
{"gfx13", "gfx13"},

// In CreateArchitectureForAgent():
if (prefix == "gfx13") return new Gfx13Architecture(agent_info);
```

### Step 4: Add to CMakeLists.txt

```cmake
${LIB_DIR}/core/architectures/gfx13_architecture.cpp
```

### Step 5: Add builders to Pm4FactoryAdapter

In `src/core/pm4_factory_adapter.cpp`, add a branch in `InitializeBuilders()`:

```cpp
} else if (architecture_->IsGFX13()) {
  cmd_builder_ = new pm4_builder::Gfx13CmdBuilder(nullptr);
  pmc_builder_ = new pm4_builder::GpuPmcBuilder<
      pm4_builder::Gfx13CmdBuilder, gfx13_cntx_prim, false>(agent_info);
  spm_builder_ = new pm4_builder::GpuSpmBuilder<
      pm4_builder::Gfx13CmdBuilder, gfx13_cntx_prim>(agent_info);
  sqtt_builder_ = new pm4_builder::GpuSqttBuilder<
      pm4_builder::Gfx13CmdBuilder, gfx13_cntx_prim>(agent_info);
}
```

### Step 6: Test

```bash
ninja -C build hardware-architecture-test
./build/hardware-architecture-test --gtest_filter=*Gfx13*
```

## Testing

### Unit Tests

All abstractions have comprehensive unit tests:

- `hardware_config_tests.cpp` - 12 tests
- `register_schema_tests.cpp` - 13 tests  
- `architecture_registry_tests.cpp` - 13 tests
- `hardware_architecture_tests.cpp` - 15 tests

**Total: 53 unit tests**

### Running Tests

```bash
# All abstraction tests
ninja -C build hardware-config-test register-schema-test \
                architecture-registry-test hardware-architecture-test

# Run specific test
./build/hardware-config-test
./build/architecture-registry-test
```

### Integration Example

See `examples/pmc_counter_example.cpp` for a complete end-to-end example demonstrating:

1. Architecture registration
2. Capability queries
3. Register schema usage
4. Block information lookup
5. PMC profiling workflow

## Migration from Old System

### Old Code (Factory Pattern)

```cpp
Pm4Factory* factory = Pm4Factory::Create(agent);
if (factory->GetGpuId() == MI300_GPU_ID) {
  // MI300-specific logic
  if (xcc_number_ > 1) {
    // Multi-XCC handling
  }
}
```

### New Code (Architecture Pattern)

```cpp
const auto* arch = ArchitectureRegistry::Instance().Lookup(agent.gfxip);
const auto& config = arch->GetConfig();

if (arch->IsMI300()) {
  // MI300-specific logic
  if (config.IsMultiXCC()) {
    // Multi-XCC handling
  }
}
```

## Benefits Summary

### For Developers

- **Cleaner Code**: No hardcoded conditionals
- **Easier Testing**: Mock architectures for unit tests
- **Type Safety**: Register IDs instead of raw offsets
- **Discoverability**: Architecture capabilities explicit

### For Maintainers

- **Fewer Files**: 2 files per architecture vs 15+
- **Consistent Pattern**: All architectures follow same structure
- **Easier Review**: Architecture code isolated
- **Less Duplication**: Shared abstractions reused

### For Users

- **Faster Builds**: Less template instantiation
- **Better Errors**: Clear architecture not found messages
- **Runtime Flexibility**: Architecture detection at runtime
- **Forward Compatible**: New architectures work automatically

## Future Work

### Completed Enhancements

1. **MI Series Implementations** ✅
   - Mi100Architecture (SPM core1, accum regs 1/158)
   - Mi200Architecture (accum regs 1/185, SPM delay 0x3e)
   - Mi300Architecture (AID-aware counters, 4 AIDs, per-agent XCC)
   - Mi350Architecture (accum hi=200, SPM delay 0x33, extends Mi300)

2. **GFX12 / gfx1250 (Mi450) support** ✅
   - Gfx12Architecture for gfx1200/gfx1201
   - Mi450Architecture compiled with `GFX12_VARIANT=0x1250` in a separate TU

3. **Factory Integration** ✅
   - `Pm4FactoryAdapter` bridges `HardwareArchitecture` → `Pm4Factory` interface
   - `Pm4Factory::Create()` dispatches to new path when `AQLPROFILE_USE_NEW_ARCH` is set
   - Circular-include resolved by moving the two `Create` overloads from the header
     into `pm4_factory.cpp`

### Planned Enhancements

1. **Builder Refactoring**:
   - Extract CounterAllocator
   - Create CommandSequencer
   - Implement RegisterProgrammer
   - Simplify PmcBuilder/SpmBuilder/SqttBuilder

2. **Remove legacy factory subclasses**:
   - Once the new path is validated, Gfx9Factory/Mi100Factory/etc. can be removed
   - Requires full integration testing across all supported GPU families

3. **Block Schema Files**: JSON/YAML block definitions
   - External block configuration
   - Easier to audit and modify
   - Version control friendly

4. **Performance Optimization**:
   - Cache frequently accessed data
   - Optimize register lookups
   - Profile hot paths

## References

- Original design: `docs/development/build_system.md`
- API reference: `src/core/include/aqlprofile-sdk/aql_profile_v2.h`
- Examples: `examples/pmc_counter_example.cpp`
- Tests: `src/core/tests/*_tests.cpp`

## Questions?

For questions or issues with the new architecture:
1. Check this documentation
2. Review example code in `examples/`
3. Look at test code in `src/core/tests/`
4. File an issue with architecture details

---

**Last Updated**: 2026-04-29  
**Authors**: Claude (Architecture Design), AMD ROCm Team (Original Implementation)
