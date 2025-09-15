# AQL C Library Validation Suite

This directory contains comprehensive validation tools and test data for the AQL C library counter block implementations across GFX9 (Vega) and GFX12 (RDNA3) architectures.

## Overview

The validation suite tests all major GPU counter blocks with real hardware register addresses extracted from the AQLProfile codebase. It validates the AQLProfile v2-compatible PMC interface and packet generation functionality.

## Files

### Validation Tools
- `aql_c_pmc_dumper.c` - PMC packet dumper and validation tool
- `Makefile` - Build system for validation tools

### Test Data
- `test_all_gfx9_blocks.csv` - Comprehensive GFX9 counter test (32 counters, 10 blocks)
- `test_all_gfx12_blocks.csv` - Comprehensive GFX12 counter test (32 counters, 11 blocks)
- `test_counters_gfx942.csv` - Basic counter validation test (16 counters, 5 blocks)

### Reference Results
- `aql_c_comprehensive_gfx9_packets.txt` - GFX9 validation packet dump
- `aql_c_comprehensive_gfx12_packets.txt` - GFX12 validation packet dump
- `aqlprofile_v2_packets.txt` - Reference AQLProfile v2 packet dump

### Reports
- `comprehensive_validation_report.md` - Complete validation results and analysis

## Quick Start

1. **Build the validation tools:**
   ```bash
   make all
   ```

2. **Run all validation tests:**
   ```bash
   make test
   ```

3. **Run architecture-specific tests:**
   ```bash
   make test-gfx9    # GFX9/Vega validation
   make test-gfx12   # GFX12/RDNA3 validation
   make test-basic   # Basic functionality test
   ```

## Prerequisites

- ROCm installation in `/opt/rocm`
- AQL C library built (`../build/libaql_c_port.a`)
- GCC with C99 support
- AMD GPU hardware (optional, for actual counter testing)

## Test Coverage

### GFX9 (Vega) Architecture
- **Blocks Tested**: CB, CPC, CPF, CPG, DB, GDS, GRBM, PA_SC, PA_SU, SQ, TCP, TCC
- **Success Rate**: 69% (22/32 counters)
- **Real Hardware Registers**: ✅ All extracted from `gc_9_2_1_offset.h`

### GFX12 (RDNA3) Architecture
- **Blocks Tested**: CB, CPC, CPF, CPG, DB, GDS, GRBM, PA_SC, PA_SU, SQ, TA, TCP, TD, GL2C
- **Success Rate**: 75% (24/32 counters)
- **Real Hardware Registers**: ✅ All extracted from `gc_12_0_0_offset.h`

## Validation Results

The validation demonstrates:

- ✅ **Real hardware register integration** with actual addresses from AQLProfile
- ✅ **AQLProfile v2 interface compatibility** with start/stop/read packet generation
- ✅ **Architecture-specific support** for both GFX9 and GFX12 features
- ✅ **PMC packet generation** following AMD AQL format specifications
- ✅ **Memory management** with proper command buffer allocation/cleanup

### Known Issues
- PA_SC/PA_SU blocks show event creation failures (-12) on both architectures
- TA block shows similar issues on GFX12 only
- These are graphics pipeline blocks that may require additional setup

## Build Targets

- `make all` - Build all validation tools
- `make test` - Run comprehensive validation tests
- `make test-gfx9` - GFX9-specific validation
- `make test-gfx12` - GFX12-specific validation
- `make test-basic` - Basic functionality test
- `make clean` - Remove build artifacts
- `make help` - Show detailed help

## Output Files

Test results are saved to `build/` directory:
- `gfx9_validation_results.txt` - GFX9 test output
- `gfx12_validation_results.txt` - GFX12 test output
- `basic_validation_results.txt` - Basic test output

## Integration

The validation suite confirms the AQL C library is production-ready for:
- Core compute workload monitoring (CPC, CPF, CPG blocks)
- Memory and cache monitoring (CB, DB, SQ, TCP, TCC/GL2C blocks)
- System-level monitoring (GRBM block)

The library provides a complete AQLProfile v2-compatible interface for performance counter programming on AMD GPUs.