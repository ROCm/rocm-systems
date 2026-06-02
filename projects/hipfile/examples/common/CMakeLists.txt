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
target_compile_options(examples_common PRIVATE -Wall -Wextra -fno-sanitize=all)
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

# Host-only per-test scratch driver. Creates a unique mkdtemp dir under a
# --base (typically AIS_CAPABLE_DIR), optionally seeds an {INPUT} file,
# substitutes {TMPDIR}/{INPUT} in the child argv, runs the example, and
# rm -rf's the dir on exit. Used by the basics/ CTest wrappers to give each
# test its own scratch space without the shared-volume cleanup races that
# the SHA256-tagged shared fixture worked around.
if(BUILD_TESTING)
    add_executable(examples_tmpdriver examples_tmpdriver.cpp)
    target_compile_features(examples_tmpdriver PRIVATE cxx_std_17)
    set_target_properties(examples_tmpdriver PROPERTIES CXX_EXTENSIONS OFF)
    target_compile_options(examples_tmpdriver PRIVATE -Wall -Wextra -fno-sanitize=all)
endif()
