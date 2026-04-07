# Test Bundle Generation

This directory contains infrastructure for generating test assets used to validate bundled binary unbundling functionality.

## Overview

The test assets are versioned by:

- **Platform**: `linux` (ELF) or `windows` (COFF)
- **Code Object Version**: `cov5`, `cov6`, etc.

Test bundles cover different scenarios:

- Single vs multiple GPU architectures
- Uncompressed vs compressed code objects
- Various architecture combinations (CDNA, RDNA, etc.)

## Files

- `simple_kernel.hip`: Minimal HIP kernel source for test generation
- `build_test_bundles.py`: Cross-platform Python script to generate test bundles

## Usage

### Linux

```bash
# Activate venv
source ../venv/bin/activate

# Auto-detect ROCm installation
python build_test_bundles.py

# Or specify ROCm path explicitly
python build_test_bundles.py --rocm-path /opt/rocm
```

### Windows

```powershell
# Activate venv
..\venv\Scripts\activate

# Auto-detect ROCm installation
python build_test_bundles.py

# Or specify ROCm path explicitly
python build_test_bundles.py --rocm-path "C:\Program Files\AMD\ROCm"
```

## Output Structure

Generated test assets are placed in:

```
test_assets/bundled_binaries/
├── linux/
│   ├── cov5/
│   │   ├── test_kernel_single                # Executable (single arch)
│   │   ├── test_kernel_multi                 # Executable (multi arch)
│   │   ├── test_kernel_compressed            # Executable (compressed)
│   │   ├── test_kernel_wide                  # Executable (wide arch coverage)
│   │   ├── libtest_kernel_single.so          # Shared library (single arch)
│   │   ├── libtest_kernel_multi.so           # Shared library (multi arch)
│   │   └── MANIFEST.txt
│   └── cov6/
│       └── ...
└── windows/
    ├── cov5/
    │   ├── test_kernel_single.exe            # Executable (single arch)
    │   ├── test_kernel_multi.exe             # Executable (multi arch)
    │   ├── test_kernel_compressed.exe        # Executable (compressed)
    │   ├── test_kernel_wide.exe              # Executable (wide arch coverage)
    │   ├── test_kernel_single.dll            # Shared library (single arch)
    │   ├── test_kernel_multi.dll             # Shared library (multi arch)
    │   └── MANIFEST.txt
    └── cov6/
        └── ...
```

**Note**: All test assets are fully linked executables or shared libraries. This ensures consistent cross-platform testing (both Linux ELF and Windows PE/COFF formats).

## Test Bundle Types

All test bundles are fully linked executables or shared libraries (not intermediate object files). This ensures:

- Cross-platform compatibility (Linux ELF and Windows PE/COFF)
- Realistic unbundling scenarios matching actual deployment
- Consistent behavior across different toolchains

### Executables

#### test_kernel_single / test_kernel_single.exe

- Single architecture: gfx1100
- Minimal test case for basic unbundling

#### test_kernel_multi / test_kernel_multi.exe

- Multiple architectures: gfx1100, gfx1101
- Tests multi-target handling

#### test_kernel_compressed / test_kernel_compressed.exe

- Multiple architectures: gfx1100, gfx1101
- Compressed code objects (CCOB format) if compiler supports it
- Tests compression handling
- Falls back to uncompressed if compression not supported

#### test_kernel_wide / test_kernel_wide.exe

- Wide architecture coverage: gfx900, gfx906, gfx908, gfx90a, gfx1100
- Tests handling of diverse architecture sets (CDNA + RDNA)

### Shared Libraries

#### libtest_kernel_single.so / test_kernel_single.dll

- Single architecture: gfx1100
- Shared library format
- Tests unbundling from libraries

#### libtest_kernel_multi.so / test_kernel_multi.dll

- Multiple architectures: gfx1100, gfx1101
- Shared library format
- Tests multi-arch unbundling from libraries

## Regenerating Test Assets

Test assets should be regenerated when:

1. **Toolchain updates**: New ROCm/HIP versions may produce different bundle formats
1. **New architectures**: Adding support for new GPU architectures
1. **Format changes**: Code object format changes (e.g., COV5 → COV6)
1. **Bug fixes**: Issues found with specific bundle types

### Process

1. Update `simple_kernel.hip` if needed (keep it minimal!)
1. Run `build_test_bundles.py` on target platform
1. Commit generated binaries to `test_assets/bundled_binaries/`
1. Update test cases to cover new variants

### Best Practices

- **Keep kernels simple**: Minimal code to reduce bundle size
- **Version everything**: Platform + code object version in directory structure
- **Document changes**: Update MANIFEST.txt to explain what changed
- **Test both formats**: Regenerate on both Linux and Windows when possible

## Windows COFF Bundles

Windows uses COFF (Common Object File Format) instead of ELF. To generate Windows test assets:

1. Set up Windows environment with HIP/ROCm SDK
1. Run the same `build_test_bundles.py` script
1. Script automatically detects platform and generates COFF bundles
1. Commit to `test_assets/bundled_binaries/windows/`

**Note**: Windows COFF bundles use the same clang-offload-bundler tool but produce different binary formats that must be handled separately during unbundling.

## Corner Cases to Test

The test bundles help validate handling of:

- **Empty bundles**: Bundles with no device code
- **Host-only bundles**: Bundles with only host code
- **Malformed bundles**: Corrupted or invalid bundle structures
- **Version mismatches**: Old vs new code object formats
- **Platform differences**: ELF vs COFF format variations
- **Compression edge cases**: Partially compressed, unsupported compression

## Integration with Tests

Test bundles are consumed by:

- `tests/test_binutils.py`: Basic unbundling tests
- `tests/test_artifact_scanner.py`: Tree scanning with bundled binaries
- Future: `tests/test_unbundling_comprehensive.py`: Exhaustive bundle testing

Each test should:

1. Load test bundle from `test_assets/bundled_binaries/`
1. Unbundle and validate contents
1. Check for expected architectures
1. Verify code object integrity
