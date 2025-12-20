# FINDROCm find_package() Support Guide

This guide explains how to use the `find_package()` of FINDROCm config.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Component Selection](#component-selection)
5. [Example Applications](#example-applications)
6. [Troubleshooting](#troubleshooting)

## Overview

After building and installing FINDROCm, you can use CMake's `find_package()` command to:
- Locate the installed ROCm components
- Link against ROCm libraries
- Access ROCm headers and utilities
- Build test applications that depend on ROCm

### What Gets Installed

When you install FINDROCm, the following CMake files are created:

```
<install-prefix>/lib/cmake/FINDROCm/
├── FINDROCmConfig.cmake         # Main configuration file
└── FINDROCmConfig-version.cmake # Version compatibility file
```

These files enable `find_package(FINDROCm)` to work.

## Installation

### 1. Configure FINDROCm CMAKE Config

```bash
cd rocm-systems/shared/FINDROCm
chmod +x CMAKE_Config_gen_FINDROCm.sh
./CMAKE_Config_gen_FINDROCm.sh
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

# Find FINDROCm
find_package(FINDROCm REQUIRED)

# Your executable
add_executable(my_test main.cpp)

# Link against ROCm libraries
target_link_libraries(my_test PRIVATE ${FINDROCM_LIBRARIES})
target_include_directories(my_test PRIVATE ${FINDROCM_INCLUDE_DIRS})
```

### Specify Installation Path

If FINDROCm is not in a standard location:

```bash
cmake .. -DFINDROCm_DIR=/path/to/FINDROCm/lib/cmake/FINDROCm
```

Or set CMAKE_PREFIX_PATH:

```bash
cmake .. -DCMAKE_PREFIX_PATH=/path/to/FINDROCm
```

## Component Selection

You can request specific components:

### Request Specific Components

```cmake
find_package(FINDROCm REQUIRED COMPONENTS
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
find_package(FINDROCm REQUIRED)

message(STATUS "Available ROCm components: ${FINDROCM_AVAILABLE_COMPONENTS}")
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

# Find FINDROCm with rocminfo component
find_package(FINDROCm REQUIRED COMPONENTS rocm-core rocr-runtime)

# Create test executable
add_executable(rocm_info_test main.cpp)

# Link against ROCm libraries
target_include_directories(rocm_info_test PRIVATE ${FINDROCM_INCLUDE_DIRS})
target_link_libraries(rocm_info_test PRIVATE ${FINDROCM_LIBRARIES})

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
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build .
./rocm_info_test
```

### Example 2: Multiple Components

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(ROCmMultiTest VERSION 1.0.0 LANGUAGES CXX)

# Find multiple components
find_package(FINDROCm REQUIRED COMPONENTS
    rocm-core
    rocr-runtime
)

# Optional component
find_package(FINDROCm COMPONENTS rocprofiler-sdk)

add_executable(multi_test main.cpp)

target_include_directories(multi_test PRIVATE ${FINDROCM_INCLUDE_DIRS})

# Link required components
target_link_libraries(multi_test PRIVATE 
    rocm-core
    hsa-runtime64::hsa-runtime64
)

# Conditionally link optional component
if(FINDROCm_rocprofiler-sdk_FOUND)
    target_link_libraries(multi_test PRIVATE rocprofiler-sdk)
    target_compile_definitions(multi_test PRIVATE HAVE_ROCPROFILER_SDK)
endif()
```

### Example 4: Using ROCm Variables

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(ROCmPathTest VERSION 1.0.0 LANGUAGES CXX)

# Find FINDROCm
find_package(FINDROCm REQUIRED)

# Use provided variables
message(STATUS "FIND ROCm Version: ${FINDROCM_VERSION}")
message(STATUS "FIND ROCm Path: ${FINDROCM_ROCM_PATH}")
message(STATUS "Install Prefix: ${FINDROCM_PREFIX}")
message(STATUS "Library Dir: ${FINDROCM_LIB_DIR}")
message(STATUS "Include Dir: ${FINDROCM_INCLUDE_DIR}")
message(STATUS "Binary Dir: ${FINDROCM_BIN_DIR}")

add_executable(path_test main.cpp)

# Use the paths
target_include_directories(path_test PRIVATE ${FINDROCM_INCLUDE_DIR})
```

## Provided Variables

After `find_package(FINDROCm)` succeeds, the following variables are available:

### Version Information
```cmake
FINDROCM_VERSION         # Full version string (e.g., "7.1.0")
FINDROCM_VERSION_MAJOR   # Major version number
FINDROCM_VERSION_MINOR   # Minor version number
FINDROCM_VERSION_PATCH   # Patch version number
```

### Installation Paths
```cmake
FINDROCM_PREFIX          # Installation prefix
FINDROCM_LIB_DIR         # Library directory
FINDROCM_INCLUDE_DIR     # Include directory
FINDROCM_BIN_DIR         # Binary directory
FINDROCM_ROCM_PATH       # ROCm installation path
```

### Components
```cmake
FINDROCM_AVAILABLE_COMPONENTS  # List of available components
FINDROCm_FOUND_COMPONENTS      # List of found components
FINDROCm_<component>_FOUND     # TRUE if component was found
```

### Libraries and Includes
```cmake
FINDROCm_LIBRARIES       # List of libraries to link
FINDROCm_INCLUDE_DIRS    # List of include directories
```

## Advanced Usage

### Verbose Output

Enable verbose output to see detailed information:

```cmake
set(FINDROCm_FIND_VERBOSE TRUE)
find_package(FINDROCm REQUIRED)
```

Or:

```cmake
set(FINDROCM_VERBOSE TRUE)
find_package(FINDROCm REQUIRED)
```

### Version Requirements

Require a specific version:

```cmake
find_package(FINDROCm 7.1.0 EXACT REQUIRED)
```

Or a minimum version:

```cmake
find_package(FINDROCm 7.0.0 REQUIRED)
```

### Quiet Mode

Suppress non-error messages:

```cmake
find_package(FINDROCm QUIET COMPONENTS rocm-core)
```

## Troubleshooting

### Problem: Package Not Found

**Error:**
```
Could not find a package configuration file provided by "FINDROCm"
```

**Solutions:**

1. **Specify the installation path:**
   ```bash
   cmake .. -DFINDROCm_DIR=/path/to/lib/cmake/FINDROCm
   ```

2. **Use CMAKE_PREFIX_PATH:**
   ```bash
   cmake .. -DCMAKE_PREFIX_PATH=/path/to/FINDROCm
   ```

3. **Check installation:**
   ```bash
   ls /opt/AMD/lib/cmake/FINDROCm/
   # Should show FINDROCmConfig.cmake
   ```

### Problem: Component Not Found

**Error:**
```
FINDROCm: Required component 'xyz' not found
```

**Solutions:**

1. **Check available components:**
   ```cmake
   find_package(FINDROCm REQUIRED)
   message(STATUS "Available: ${FINDROCM_AVAILABLE_COMPONENTS}")
   ```

2. **Make component optional:**
   ```cmake
   find_package(FINDROCm COMPONENTS xyz)
   if(NOT FINDROCm_xyz_FOUND)
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
   # Use: target_link_libraries(test ${FINDROCM_LIBRARIES})
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
   target_include_directories(test PRIVATE ${FINDROCM_INCLUDE_DIRS})
   ```

2. **Check specific component paths:**
   ```cmake
   target_include_directories(test PRIVATE
       ${FINDROCM_INCLUDE_DIR}/rocm-core
       ${FINDROCM_INCLUDE_DIR}/roctracer
   )
   ```

## Best Practices

### 1. Always Specify Required Components

```cmake
# Good
find_package(FINDROCm REQUIRED COMPONENTS rocm-core rocr-runtime)

# Not recommended
find_package(FINDROCm REQUIRED)
```

### 2. Check Component Availability

```cmake
find_package(FINDROCm COMPONENTS optional-component)
if(FINDROCm_optional-component_FOUND)
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
export CMAKE_PREFIX_PATH=/opt/rocm:$CMAKE_PREFIX_PATH
cmake ..
```

### 5. Handle Missing Components Gracefully

```cmake
find_package(FINDROCm REQUIRED COMPONENTS rocm-core)
find_package(FINDROCm COMPONENTS roctracer)

if(FINDROCm_roctracer_FOUND)
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
2. Verify installation: `ls $PREFIX/lib/cmake/FINDROCm`
3. Enable verbose output: `set(FINDROCm_FIND_VERBOSE TRUE)`



