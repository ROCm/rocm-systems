# ROCM Systems CMAKE Config Test Application Examples

This directory contains example test applications demonstrating how to use `find_package(rocm-systems)` in your CMake projects.

## Overview

These examples show how to:
- Use `find_package()` to locate ROCm Systems
- Request specific components
- Handle optional components
- Link against ROCm libraries
- Build test applications that depend on ROCm

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
- C++17 compatible compiler (GCC 9+, Clang 10+)
- ROCm-compatible hardware (optional, for runtime tests)

## Building the Examples

### Basic Build

```bash
cd examples/test-application
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build .
```

### Specify Custom Installation Path

If you installed ROCm Systems to a custom location:

```bash
cmake .. -Drocm-systems_DIR=/path/to/rocm-systems/lib/cmake/rocm-systems
cmake --build .
```

### Debug Build

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=/opt/rocm-systems
cmake --build .
```

## Example Applications

### 1. rocm_info_test

**Purpose:** Demonstrates accessing ROCm version and path information.

**Components used:** `rocm-core`

**Run:**
```bash
./rocm_info_test
```

**What it does:**
- Queries ROCm version information
- Displays installation paths
- Shows environment variables
- Demonstrates basic find_package usage

### 2. hsa_runtime_test

**Purpose:** Demonstrates using HSA Runtime through find_package.

**Components used:** `rocr-runtime`

**Run:**
```bash
./hsa_runtime_test
```

**What it does:**
- Initializes HSA Runtime
- Enumerates available HSA agents (CPUs, GPUs)
- Displays agent information
- Shows how to link against HSA libraries

**Note:** Requires ROCm-compatible hardware or will gracefully handle missing devices.

**What it does:**
- Shows conditional compilation based on component availability
- Demonstrates optional component pattern
- Explains how to enable optional components
- Adapts behavior based on what's available

## Understanding the CMakeLists.txt

The example `CMakeLists.txt` demonstrates several important patterns:

### Required Components

```cmake
find_package(rocm-systems REQUIRED COMPONENTS
    rocm-core
    rocr-runtime
)
```

This ensures `rocm-core` and `rocr-runtime` are available, or build fails.

### Optional Components

```cmake
find_package(rocm-systems COMPONENTS
    rocprofiler-sdk
)
```

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

## Building Individual Examples

You can build only specific targets:

```bash
# Build only rocm_info_test
cmake --build . --target rocm_info_test

# Build only hsa_runtime_test
cmake --build . --target hsa_runtime_test
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

### Problem: Missing headers

```
fatal error: hsa/hsa.h: No such file or directory
```

**Solution:**
```cmake
target_include_directories(my_app PRIVATE ${ROCM_SYSTEMS_INCLUDE_DIRS})
```

### Problem: Linking errors

```
undefined reference to `hsa_init'
```

**Solution:** Make sure to link against the libraries:
```cmake
target_link_libraries(my_app PRIVATE ${ROCM_SYSTEMS_LIBRARIES})
# Or specific library:
target_link_libraries(my_app PRIVATE hsa-runtime64::hsa-runtime64)
```

## Expected Output

### rocm_info_test

```
========================================
ROCm Systems Information Test
========================================
...
ROCm Version Major: 7
ROCm Version Minor: 1
ROCm Version Patch: 0
...
✓ Application built successfully using find_package(rocm-systems)
```

### hsa_runtime_test

```
========================================
HSA Runtime Test
========================================
✓ HSA Runtime initialized successfully
...
Available HSA Agents
========================================
  Agent: AMD Ryzen ... (CPU)
  Agent: gfx90a (GPU) - Compute Units: 110
...
Total agents found: 2
```

## Customization

### Add Your Own Test

1. Create a new `.cpp` file
2. Add to `CMakeLists.txt`:
   ```cmake
   add_executable(my_test my_test.cpp)
   target_link_libraries(my_test PRIVATE ${ROCM_SYSTEMS_LIBRARIES})
   ```
3. Build: `cmake --build . --target my_test`

### Request Different Components

Modify the `find_package()` call in `CMakeLists.txt`:

```cmake
find_package(rocm-systems REQUIRED COMPONENTS
    rocm-core
    rocprofiler-sdk
)
```

## Next Steps

1. **Study the CMakeLists.txt** - See how components are detected and used
2. **Run the examples** - Understand what information is available
3. **Modify for your needs** - Adapt the patterns to your application
4. **Read the documentation** - See `FIND_PACKAGE_GUIDE.md` in repo root

## Additional Resources

- [ROCM-SYSTEMS_FIND_PACKAGE_GUIDE.md](../../ROCM-SYSTEMS_FIND_PACKAGE_GUIDE.md) - Comprehensive guide
- [ROCm Documentation](https://rocm.docs.amd.com/)



