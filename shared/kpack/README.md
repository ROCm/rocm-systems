# rocm-kpack

Kernel packing tool and runtime library for managing ROCm device code distribution.

## Overview

The rocm-kpack project provides tooling and runtime libraries to cleanly separate and distribute ROCm device code assets per GPU architecture. This enables efficient packaging and delivery of ROCm applications and libraries by avoiding fat binaries and providing structured access to per-architecture kernel bundles.

## Problem Statement

Current ROCm libraries handle device code distribution in two primary ways:

1. **Fat binaries**: Embed hsaco files for all GPU architectures directly into shared objects, executables, and libraries. This results in bloated binaries containing code for architectures that may never be used.

1. **Ad-hoc database directories**: Some libraries manage exploded directories of kernels and metadata in custom formats. These are difficult to package cleanly and lack standardization.

Neither approach provides a clean mechanism for distributing device code separately from host code, making it challenging to optimize download sizes and installation footprints for specific GPU architectures.

## Solution

rocm-kpack introduces a structured format and tooling ecosystem to:

- Separate device code from host code at packaging time
- Bundle device code per GPU architecture (gfx target)
- Enable runtime discovery and loading of appropriate device code
- Support clean distribution of architecture-specific packages

## Architecture

The project consists of three core components:

### 1. Python Tooling

Tools for transforming ROCm install trees:

- Extract device code (hsaco files) from binaries and database directories
- Organize device code by GPU architecture (gfx target)
- Generate kpack archives for packaging and distribution
- Post-process Python wheels to create per-arch sidecar wheels (future)

### 2. C++ Runtime Library

Runtime library for querying and loading device code:

- Query kpack archives for supported hsaco files and metadata
- Integrate with ROCm runtime for device code discovery
- Provide host-side APIs for kernel library database management
- Support efficient lookup by GPU architecture and kernel identifiers

### 3. Comprehensive Testing

Validation suite covering:

- Python tooling correctness (extraction, bundling, unbundling)
- C++ library API and functionality
- Integration testing with real ROCm assets
- Round-trip testing (pack → unpack → verify)

## Future Extensions

- **Advanced compression**: Optimize compression by treating collections of kernels across architectures as a unified compression domain
- **WheelNext integration**: Post-process Python wheel files to generate per-architecture sidecar wheels with dynamic dependency management
- **Format versioning**: Support evolution of the kpack format with backward compatibility

## Use Cases

- **ROCm distribution**: Ship base ROCm installation with on-demand architecture packages
- **Application deployment**: Distribute applications with separate device code bundles per supported GPU
- **Cloud services**: Download only the device code needed for available GPU hardware
- **Development workflows**: Test and validate device code for specific architectures without full fat binaries

## Project Status

This is an active development project. The initial implementation focuses on:

- Python binutils for working with hsaco files
- Bulk unbundling tools for extracting device code
- Test infrastructure and asset management

## License

See LICENSE file for details.
