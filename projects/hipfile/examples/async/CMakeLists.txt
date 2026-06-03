# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Async examples for the hipFile C API. Same style as examples/basics:
# bypass ais_add_executable and wire dependencies in by hand. Shared helpers
# come from the examples_common target defined in examples/common; the parent
# CMakeLists adds examples/common before examples/async, so the target is in
# scope by the time we reach this file.

set(ASYNC_EXAMPLES
    roundtrip-async
    roundtrip-async-nonblocking-stream
    roundtrip-async-multi-stream
    roundtrip-async-multi-stream-registered
)

foreach(example ${ASYNC_EXAMPLES})
    add_executable(${example} "${example}.cpp")
    target_compile_features(${example} PRIVATE cxx_std_${AIS_CXX_STANDARD})
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
    target_link_libraries(${example} PRIVATE examples_common)
endforeach()

# CTest wrappers: run each async example as a hipFile system test.
# Gated on BUILD_TESTING; AIS_INSTALL_EXAMPLES is already enforced by the
# parent CMakeLists which only adds this subdirectory when examples are built.
if(BUILD_TESTING)
    # Each test runs in its own mktemp directory so that parallel ctest
    # invocations and concurrent CI runners cannot interfere with one another.
    # The binary is passed as $0 so "$0" "$@" runs it with the file args;
    # trap ensures the directory is removed even on failure.
    #
    # Each async example calls seed_read_file() to truncate+seed its READ_FILE,
    # so no upfront dd seeding is needed.
    foreach(example ${ASYNC_EXAMPLES})
        add_test(
            NAME async_${example}
            COMMAND sh -c
                "dir=$(mktemp -d '${AIS_CAPABLE_DIR}/hipfile_async_XXXXXX') && trap 'rm -rf \"$dir\"' EXIT && \"$0\" \"$dir/in_${example}.bin\" \"$dir/out_${example}.bin\""
                $<TARGET_FILE:${example}>
        )
        set_tests_properties(async_${example} PROPERTIES
            LABELS "system;hipfile;async"
        )
    endforeach()
endif()
