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
    target_link_libraries(${example} PRIVATE examples_common)
endforeach()

# CTest wrappers: run each example as a hipFile system test.
# Gated on BUILD_TESTING; AIS_INSTALL_EXAMPLES is already enforced by the
# parent CMakeLists which only adds this subdirectory when examples are built.
if(BUILD_TESTING)
    set(BASICS_TEST_DIR "${AIS_CAPABLE_DIR}/hipfile_basics_tests")

    add_test(
        NAME basics_setup_dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BASICS_TEST_DIR}
    )
    add_test(
        NAME basics_cleanup
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${BASICS_TEST_DIR}
    )
    set_tests_properties(basics_setup_dir PROPERTIES
        FIXTURES_SETUP basics_scratch
    )
    set_tests_properties(basics_cleanup PROPERTIES
        FIXTURES_CLEANUP basics_scratch
    )

    # Every basics example calls seed_read_file() (or O_TRUNC's its input),
    # so each test gets its own READ_FILE path to avoid races under
    # `ctest --parallel`. The binaries create/truncate/seed the file
    # themselves; no upfront dd seed is needed.
    function(_basics_add_test name)
        add_test(
            NAME basics_${name}
            COMMAND $<TARGET_FILE:${name}> ${ARGN}
            WORKING_DIRECTORY ${BASICS_TEST_DIR}
        )
        set_tests_properties(basics_${name} PROPERTIES
            FIXTURES_REQUIRED basics_scratch
            LABELS "system;hipfile;basics"
        )
    endfunction()

    _basics_add_test(bufregister-write            out_bufregister.bin)
    _basics_add_test(no-bufregister-write         out_no_bufregister.bin)
    _basics_add_test(no-odirect-write             out_no_odirect.bin)
    _basics_add_test(iterative-read               in_iter.bin out_iter.bin)
    _basics_add_test(iterative-devmem-offset-read in_iter_off.bin out_iter_off.bin)
    _basics_add_test(subregion-write              in_subregion.bin out_subregion.bin)
    _basics_add_test(roundtrip-verify             rtv_created.bin rtv_copied.bin)

    # various-mem-rw exercises MODE = 1 (device), 2 (managed), 3 (pinned-host).
    foreach(mode_pair "1;device" "2;managed" "3;pinned")
        list(GET mode_pair 0 mode_num)
        list(GET mode_pair 1 mode_name)
        add_test(
            NAME basics_various-mem-rw_${mode_name}
            COMMAND
                $<TARGET_FILE:various-mem-rw> in_vmrw_${mode_name}.bin
                out_vmrw_${mode_name}.bin ${mode_num}
            WORKING_DIRECTORY ${BASICS_TEST_DIR}
        )
        set_tests_properties(basics_various-mem-rw_${mode_name} PROPERTIES
            FIXTURES_REQUIRED basics_scratch
            LABELS "system;hipfile;basics"
        )
        if(mode_name STREQUAL "managed" OR mode_name STREQUAL "pinned")
            set_tests_properties(basics_various-mem-rw_${mode_name} PROPERTIES
                DISABLED TRUE
            )
        endif()
    endforeach()
endif()
