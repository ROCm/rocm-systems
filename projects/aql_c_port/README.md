# AQL Packet Generation C Port

This project ports the AQLProfile packet generation logic from C++ to pure C for Linux kernel module integration.

## Project Structure

- `src/` - Core C implementation files
- `include/` - Public and private header files
- `tests/` - Unit tests and validation
- `docs/` - Documentation and analysis
- `examples/` - Usage examples and demos

## Goal

Port the PM4 packet generation and AQL packet population logic from the C++ AQLProfile library to pure C, enabling integration into a Linux kernel module for GPU performance monitoring via the perf subsystem.

## Architecture Support

Target architectures:
- GFX9 (Vega)
- GFX10 (RDNA1)
- GFX11 (RDNA2)
- GFX12 (RDNA3)

## Status

🚧 **In Development** - Currently analyzing C++ source and designing C port architecture.