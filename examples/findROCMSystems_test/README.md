# ROCM Systems CMAKE Config Test

This directory contains example cmake demonstrating how to use `find_package(rocm-systems)` in your CMake projects.

## Overview

These examples show how to:
- Use `find_package()` to locate ROCm Systems
- Print All Configurations, Definitions Found

## Prerequisites

### 1. Configure rocm-systems cmake
First, Configure rocm-systems:

```bash
cd rocm-systems
chmod +x CMAKE_Config_gen_rocm-systems.sh
./CMAKE_Config_gen_rocm-systems.sh
cd build
cmake --install .
```

### 2. Required Tools

- CMake 3.18 or higher
- ROCm-compatible hardware (optional, for runtime tests)

## Building the Examples

### Basic Build

```bash
cd examples/findROCMSystems_test
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
```

## Understanding the CMakeLists.txt

The example `CMakeLists.txt` demonstrates several important patterns:

```cmake
find_package(rocm-systems REQUIRED)
```

This ensures all rocm system component configs are available, or build fails.

### Component Status Variables

After `find_package()`, check individual components:

```cmake
rocm-systems_<component>_FOUND  # TRUE if component was found
```

### Using Provided Variables

```cmake
ROCM_SYSTEMS_VERSION           # Version string
ROCM_SYSTEMS_PREFIX            # Install prefix
ROCM_SYSTEMS_LIB_DIR           # Library directory
ROCM_SYSTEMS_INCLUDE_DIR       # Include directory
ROCM_SYSTEMS_INCLUDE_DIRS      # Include directories (list)
ROCM_SYSTEMS_LIBRARIES         # Libraries to link
```

## Troubleshooting

### Problem: Package not found

```
Could not find a package configuration file provided by "rocm-systems"
```

**Solution:**
```bash
# Specify the location explicitly
cmake .. -Drocm-systems_DIR=/opt/rocm-systems/lib/cmake/rocm-systems

# Or use CMAKE_PREFIX_PATH
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.0
```

### Request Different Components

Modify the `find_package()` call in `CMakeLists.txt`:

```cmake
find_package(rocm-systems REQUIRED COMPONENTS
    rocm-core
    rocprofiler-sdk
)
```

## Next Steps

3. **Modify for your needs** - Adapt the patterns to your application
4. **Read the documentation** - See `FIND_PACKAGE_GUIDE.md` in repo root

## Additional Resources

- [ROCM-SYSTEMS_FIND_PACKAGE_GUIDE.md](../../ROCM-SYSTEMS_FIND_PACKAGE_GUIDE.md) - Comprehensive guide
- [ROCm Documentation](https://rocm.docs.amd.com/)



