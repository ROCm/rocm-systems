# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Shared helpers (open/close/hash/etc.) used by the basics/ and async/
# examples. Pulled out of the per-example sources to remove verbatim
# duplication; each example still drives the hipFile API directly in main()
# so the example flow stays readable.
add_library(examples_common STATIC examples_common.cpp)
target_compile_features(examples_common PRIVATE cxx_std_${AIS_CXX_STANDARD})
set_target_properties(examples_common PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(examples_common PRIVATE -Wall -Wextra)
# hipfile is instrumented when the sanitizers are on; this static lib is linked
# into the example executables, so instrument it to match and avoid an
# instrumentation mismatch.
if(AIS_USE_SANITIZERS OR AIS_USE_THREAD_SANITIZER)
    ais_add_sanitizers(examples_common)
endif()
target_include_directories(examples_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(CMAKE_HIP_PLATFORM STREQUAL "amd")
    target_compile_definitions(examples_common PRIVATE __HIP_PLATFORM_AMD__)
elseif(CMAKE_HIP_PLATFORM STREQUAL "nvidia")
    target_compile_definitions(examples_common PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(examples_common SYSTEM PRIVATE ${HIP_INCLUDE_DIRS})
    target_include_directories(
        examples_common
        SYSTEM
        BEFORE
        PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
    )
endif()

target_include_directories(examples_common SYSTEM PRIVATE ${HIPFILE_INCLUDE_PATH})
target_link_libraries(examples_common PUBLIC hip::host hipfile)
