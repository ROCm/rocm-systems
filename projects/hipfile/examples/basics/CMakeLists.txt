# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# These examples use the hipFile C API directly from C++ (matching the
# style of the standalone aiscp/offsetcp/read-write-check programs).
# They deliberately bypass ais_add_executable and wire the dependencies
# in by hand.

set(BASICS_EXAMPLES
    bufregister-write
    no-bufregister-write
    no-odirect-write
    iterative-read
    iterative-devmem-offset-read
    various-mem-rw
    subregion-write
    roundtrip-verify
)

# Shared helpers (open/close/hash/etc.) used by every example. Pulled out of
# the per-example sources to remove verbatim duplication; each example still
# drives the hipFile API directly in main() so the example flow stays readable.
add_library(basics_common STATIC basics_common.cpp)
target_compile_features(basics_common PRIVATE cxx_std_${AIS_CXX_STANDARD})
set_target_properties(basics_common PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(basics_common PRIVATE -Wall -Wextra -fno-sanitize=all)
target_include_directories(basics_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(CMAKE_HIP_PLATFORM STREQUAL "amd")
    target_compile_definitions(basics_common PRIVATE __HIP_PLATFORM_AMD__)
elseif(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    target_compile_definitions(basics_common PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(basics_common SYSTEM PRIVATE ${HIP_INCLUDE_DIRS})
    target_include_directories(
        basics_common
        SYSTEM
        BEFORE
        PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
    )
endif()

target_include_directories(basics_common SYSTEM PRIVATE ${HIPFILE_INCLUDE_PATH})
target_link_libraries(basics_common PUBLIC hip::host hipfile)

foreach(example ${BASICS_EXAMPLES})
    add_executable(${example} "${example}.cpp")
    target_compile_features(${example} PRIVATE cxx_std_17)
    set_target_properties(${example} PROPERTIES CXX_EXTENSIONS OFF)
    target_compile_options(${example} PRIVATE -Wall -Wextra -fno-sanitize=all)

    if(CMAKE_HIP_PLATFORM STREQUAL "amd")
        target_compile_definitions(${example} PRIVATE __HIP_PLATFORM_AMD__)
    elseif(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
        target_compile_definitions(${example} PRIVATE __HIP_PLATFORM_NVIDIA__)
        target_include_directories(${example} SYSTEM PRIVATE ${HIP_INCLUDE_DIRS})
        target_include_directories(
            ${example}
            SYSTEM
            BEFORE
            PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(${example} PRIVATE CUDA::cuda_driver CUDA::cudart)
    endif()

    target_include_directories(${example} SYSTEM PRIVATE ${HIPFILE_INCLUDE_PATH})
    target_link_libraries(${example} PRIVATE basics_common)
endforeach()
