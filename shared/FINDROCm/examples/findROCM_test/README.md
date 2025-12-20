# FINDROCM CMAKE Config Test

This directory contains example cmake demonstrating how to use `find_package(FINDROCm)` in your CMake projects.

## Overview

These examples show how to:
- Use `find_package()` to locate FINDROCm
- Print All Configurations, Definitions Found

## Prerequisites

### 1. Configure FINDROCm cmake
First, Configure FINDROCm:

```bash
cd rocm-systems/shared/FINDROCm
chmod +x CMAKE_Config_gen_FINDROCm.sh
./CMAKE_Config_gen_FINDROCm.sh
cd build
cmake --install .
```

### 2. Required Tools

- CMake 3.18 or higher
- ROCm-compatible hardware (optional, for runtime tests)

## Building the Examples

### Basic Build

```bash
cd examples/findROCM_test
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
```

## Understanding the CMakeLists.txt

The example `CMakeLists.txt` demonstrates several important patterns:

```cmake
find_package(FINDROCm REQUIRED)
```

This ensures all FINDROCm configs are available, or build fails.

### Component Status Variables

After `find_package()`, check individual components:

```cmake
FINDROCm_<component>_FOUND  # TRUE if component was found
```

### Using Provided Variables

```cmake
FINDROCM_VERSION           # Version string
FINDROCM_PREFIX            # Install prefix
FINDROCM_LIB_DIR           # Library directory
FINDROCM_INCLUDE_DIR       # Include directory
FINDROCM_INCLUDE_DIRS      # Include directories (list)
FINDROCM_LIBRARIES         # Libraries to link
```

## Troubleshooting

### Problem: Package not found

```
Could not find a package configuration file provided by "FINDROCm"
```

**Solution:**
```bash
# Specify the location explicitly
cmake .. -DFINDROCm_DIR=/opt/rocm/lib/cmake/FINDROCm

# Or use CMAKE_PREFIX_PATH
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.0
```

### Request Different Components

Modify the `find_package()` call in `CMakeLists.txt`:

```cmake
find_package(FINDROCm REQUIRED COMPONENTS
    rocm-core
    rocprofiler-sdk
)
```

## Next Steps

3. **Modify for your needs** - Adapt the patterns to your application
4. **Read the documentation** - See `FIND_PACKAGE_GUIDE.md` in repo root

## Additional Resources

- [FINDROCM_FIND_PACKAGE_GUIDE.md](../../FINDROCM_FIND_PACKAGE_GUIDE.md) - Comprehensive guide
- [ROCm Documentation](https://rocm.docs.amd.com/)



