# ROCm Systems find_package() Support Guide

This guide explains how to use `find_package()` to integrate ROCm Systems components into your test applications.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Component Selection](#component-selection)
5. [Example Applications](#example-applications)
6. [Troubleshooting](#troubleshooting)

## Overview

After building and installing ROCm Systems, you can use CMake's `find_package()` command to:
- Locate the installed ROCm components
- Link against ROCm libraries
- Access ROCm headers and utilities
- Build test applications that depend on ROCm

### What Gets Installed

When you install ROCm Systems, the following CMake files are created:

```
<install-prefix>/lib/cmake/rocm-systems/
├── rocm-systems-config.cmake         # Main configuration file
└── rocm-systems-config-version.cmake # Version compatibility file
```

These files enable `find_package(rocm-systems)` to work.

## Installation

### 1. Configure ROCm Systems CMAKE Config

```bash
cd rocm-systems
chmod +x CMAKE_Config_gen_rocm-systems.sh
./CMAKE_Config_gen_rocm-systems.sh
```

### 2. Install

```bash
cd build
sudo cmake --install .
```

## Basic Usage

### Simple Example

In your test application's `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyROCmTest)

# Find ROCm Systems
find_package(rocm-systems REQUIRED)

# Your executable
add_executable(my_test main.cpp)

# Link against ROCm libraries
target_link_libraries(my_test PRIVATE ${ROCM_SYSTEMS_LIBRARIES})
target_include_directories(my_test PRIVATE ${ROCM_SYSTEMS_INCLUDE_DIRS})
```

### Specify Installation Path

If ROCm Systems is not in a standard location:

```bash
cmake .. -Drocm-systems_DIR=/path/to/rocm-systems/lib/cmake/rocm-systems
```

Or set CMAKE_PREFIX_PATH:

```bash
cmake .. -DCMAKE_PREFIX_PATH=/path/to/rocm-systems
```

## Component Selection

You can request specific components:

### Request Specific Components

```cmake
find_package(rocm-systems REQUIRED COMPONENTS
    rocm-core
    rocr-runtime
    rocminfo
)
```

### Available Components

The following components can be requested:

| Component | Description |
|-----------|-------------|
| `rocm-core` | Core ROCm versioning library |
| `rocr-runtime` | HSA Runtime |
| `rocm_smi` | System Management Interface |
| `clr` | Compute Language Runtime (HIP) |
| `rocprofiler-sdk` | ROCProfiler SDK |
| `rocprofiler-systems` | Omnitrace profiler |
| `rocprofiler-register` | Profiler registration |
| `rdc` | Data Center tool |

### Check Available Components

```cmake
find_package(rocm-systems REQUIRED)

message(STATUS "Available ROCm components: ${ROCM_SYSTEMS_AVAILABLE_COMPONENTS}")
```

## Example Applications

### Example 1: Basic ROCm Info Tool

**Directory structure:**
```
my-rocm-test/
├── CMakeLists.txt
└── main.cpp
```

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(ROCmInfoTest VERSION 1.0.0 LANGUAGES CXX)

# Find ROCm Systems with rocminfo component
find_package(rocm-systems REQUIRED COMPONENTS rocm-core rocr-runtime)

# Create test executable
add_executable(rocm_info_test main.cpp)

# Link against ROCm libraries
target_include_directories(rocm_info_test PRIVATE ${ROCM_SYSTEMS_INCLUDE_DIRS})
target_link_libraries(rocm_info_test PRIVATE ${ROCM_SYSTEMS_LIBRARIES})

# Set C++ standard
set_target_properties(rocm_info_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
```

**main.cpp:**
```cpp
#include <iostream>
#include <rocm-core/rocm_version.h>

int main() {
    std::cout << "ROCm Version: " << ROCM_VERSION_STRING << std::endl;
    return 0;
}
```

**Build:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm-systems
cmake --build .
./rocm_info_test
```

### Example 2: Multiple Components

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(ROCmMultiTest VERSION 1.0.0 LANGUAGES CXX)

# Find multiple components
find_package(rocm-systems REQUIRED COMPONENTS
    rocm-core
    rocr-runtime
)

# Optional component
find_package(rocm-systems COMPONENTS rocprofiler-sdk)

add_executable(multi_test main.cpp)

target_include_directories(multi_test PRIVATE ${ROCM_SYSTEMS_INCLUDE_DIRS})

# Link required components
target_link_libraries(multi_test PRIVATE 
    rocm-core
    hsa-runtime64::hsa-runtime64
)

# Conditionally link optional component
if(rocm-systems_rocprofiler-sdk_FOUND)
    target_link_libraries(multi_test PRIVATE rocprofiler-sdk)
    target_compile_definitions(multi_test PRIVATE HAVE_ROCPROFILER_SDK)
endif()
```

### Example 4: Using ROCm Variables

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(ROCmPathTest VERSION 1.0.0 LANGUAGES CXX)

# Find rocm-systems
find_package(rocm-systems REQUIRED)

# Use provided variables
message(STATUS "ROCm Systems Version: ${ROCM_SYSTEMS_VERSION}")
message(STATUS "ROCm Path: ${ROCM_SYSTEMS_ROCM_PATH}")
message(STATUS "Install Prefix: ${ROCM_SYSTEMS_PREFIX}")
message(STATUS "Library Dir: ${ROCM_SYSTEMS_LIB_DIR}")
message(STATUS "Include Dir: ${ROCM_SYSTEMS_INCLUDE_DIR}")
message(STATUS "Binary Dir: ${ROCM_SYSTEMS_BIN_DIR}")

add_executable(path_test main.cpp)

# Use the paths
target_include_directories(path_test PRIVATE ${ROCM_SYSTEMS_INCLUDE_DIR})
```

## Provided Variables

After `find_package(rocm-systems)` succeeds, the following variables are available:

### Version Information
```cmake
ROCM_SYSTEMS_VERSION         # Full version string (e.g., "7.1.0")
ROCM_SYSTEMS_VERSION_MAJOR   # Major version number
ROCM_SYSTEMS_VERSION_MINOR   # Minor version number
ROCM_SYSTEMS_VERSION_PATCH   # Patch version number
```

### Installation Paths
```cmake
ROCM_SYSTEMS_PREFIX          # Installation prefix
ROCM_SYSTEMS_LIB_DIR         # Library directory
ROCM_SYSTEMS_INCLUDE_DIR     # Include directory
ROCM_SYSTEMS_BIN_DIR         # Binary directory
ROCM_SYSTEMS_ROCM_PATH       # ROCm installation path
```

### Components
```cmake
ROCM_SYSTEMS_AVAILABLE_COMPONENTS  # List of available components
rocm-systems_FOUND_COMPONENTS      # List of found components
rocm-systems_<component>_FOUND     # TRUE if component was found
```

### Libraries and Includes
```cmake
ROCM_SYSTEMS_LIBRARIES       # List of libraries to link
ROCM_SYSTEMS_INCLUDE_DIRS    # List of include directories
```

## Advanced Usage

### Verbose Output

Enable verbose output to see detailed information:

```cmake
set(rocm-systems_FIND_VERBOSE TRUE)
find_package(rocm-systems REQUIRED)
```

Or:

```cmake
set(ROCM_SYSTEMS_VERBOSE TRUE)
find_package(rocm-systems REQUIRED)
```

### Version Requirements

Require a specific version:

```cmake
find_package(rocm-systems 7.1.0 EXACT REQUIRED)
```

Or a minimum version:

```cmake
find_package(rocm-systems 7.0.0 REQUIRED)
```

### Quiet Mode

Suppress non-error messages:

```cmake
find_package(rocm-systems QUIET COMPONENTS rocm-core)
```

## Troubleshooting

### Problem: Package Not Found

**Error:**
```
Could not find a package configuration file provided by "rocm-systems"
```

**Solutions:**

1. **Specify the installation path:**
   ```bash
   cmake .. -Drocm-systems_DIR=/path/to/lib/cmake/rocm-systems
   ```

2. **Use CMAKE_PREFIX_PATH:**
   ```bash
   cmake .. -DCMAKE_PREFIX_PATH=/path/to/rocm-systems
   ```

3. **Check installation:**
   ```bash
   ls /opt/rocm-systems/lib/cmake/rocm-systems/
   # Should show rocm-systems-config.cmake
   ```

### Problem: Component Not Found

**Error:**
```
ROCm Systems: Required component 'xyz' not found
```

**Solutions:**

1. **Check available components:**
   ```cmake
   find_package(rocm-systems REQUIRED)
   message(STATUS "Available: ${ROCM_SYSTEMS_AVAILABLE_COMPONENTS}")
   ```

2. **Make component optional:**
   ```cmake
   find_package(rocm-systems COMPONENTS xyz)
   if(NOT rocm-systems_xyz_FOUND)
       message(STATUS "Component xyz not available, skipping")
   endif()
   ```

### Problem: Library Linking Errors

**Error:**
```
undefined reference to ...
```

**Solutions:**

1. **Use correct library names:**
   ```cmake
   # Instead of: target_link_libraries(test rocm-core)
   # Use: target_link_libraries(test ${ROCM_SYSTEMS_LIBRARIES})
   ```

2. **Check for individual component packages:**
   ```cmake
   find_package(hsa-runtime64 REQUIRED)
   target_link_libraries(test hsa-runtime64::hsa-runtime64)
   ```

### Problem: Headers Not Found

**Error:**
```
fatal error: rocm-core/rocm_version.h: No such file or directory
```

**Solutions:**

1. **Add include directories:**
   ```cmake
   target_include_directories(test PRIVATE ${ROCM_SYSTEMS_INCLUDE_DIRS})
   ```

2. **Check specific component paths:**
   ```cmake
   target_include_directories(test PRIVATE
       ${ROCM_SYSTEMS_INCLUDE_DIR}/rocm-core
       ${ROCM_SYSTEMS_INCLUDE_DIR}/roctracer
   )
   ```

## Best Practices

### 1. Always Specify Required Components

```cmake
# Good
find_package(rocm-systems REQUIRED COMPONENTS rocm-core rocr-runtime)

# Not recommended
find_package(rocm-systems REQUIRED)
```

### 2. Check Component Availability

```cmake
find_package(rocm-systems COMPONENTS optional-component)
if(rocm-systems_optional-component_FOUND)
    # Use the component
endif()
```

### 3. Use Imported Targets When Available

```cmake
# Preferred (if available)
target_link_libraries(test hsa-runtime64::hsa-runtime64)

# Fallback
target_link_libraries(test roctracer64)
```

### 4. Set Appropriate CMAKE_PREFIX_PATH

```bash
export CMAKE_PREFIX_PATH=/opt/rocm-systems:$CMAKE_PREFIX_PATH
cmake ..
```

### 5. Handle Missing Components Gracefully

```cmake
find_package(rocm-systems REQUIRED COMPONENTS rocm-core)
find_package(rocm-systems COMPONENTS roctracer)

if(rocm-systems_roctracer_FOUND)
    target_compile_definitions(my_test PRIVATE HAVE_ROCTRACER)
endif()
```

## Complete Example Project

See the `examples/test-application/` directory for a complete working example.

## Additional Resources

- [CMake find_package Documentation](https://cmake.org/cmake/help/latest/command/find_package.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)

## Support

For issues with find_package support:

1. Check this guide's troubleshooting section
2. Verify installation: `ls $PREFIX/lib/cmake/rocm-systems`
3. Enable verbose output: `set(rocm-systems_FIND_VERBOSE TRUE)`

---

**Version:** 1.0  
**Last Updated:** November 2025


