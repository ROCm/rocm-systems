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

# CTest wrappers: each example runs under examples_tmpdriver, which mkdtemp's
# a unique scratch dir under AIS_CAPABLE_DIR per test, optionally seeds an
# {INPUT} file with deterministic bytes, substitutes {TMPDIR}/{INPUT} in argv,
# and rm -rf's the dir on exit. This removes the prior shared-fixture model
# (and its SHA256 runner-tag namespacing hack for shared CI volumes).
if(BUILD_TESTING)
    # Read-style tests need a seeded input. 256 KiB matches the prior
    # `dd bs=4096 count=64` fixture, so behavior is unchanged for the
    # examples that consumed BASICS_INPUT.
    set(_basics_input_bytes 262144)

    function(_basics_add_test name)
        add_test(
            NAME basics_${name}
            COMMAND
                $<TARGET_FILE:examples_tmpdriver> --base ${AIS_CAPABLE_DIR}
                                                  ${_basics_driver_extra}
                -- $<TARGET_FILE:${name}> ${ARGN}
        )
        set_tests_properties(basics_${name} PROPERTIES
            LABELS "system;hipfile;basics"
        )
    endfunction()

    # Write-only examples — no input needed.
    set(_basics_driver_extra "")
    _basics_add_test(bufregister-write    {TMPDIR}/out_bufregister.bin)
    _basics_add_test(no-bufregister-write {TMPDIR}/out_no_bufregister.bin)
    _basics_add_test(no-odirect-write     {TMPDIR}/out_no_odirect.bin)
    _basics_add_test(roundtrip-verify     {TMPDIR}/rtv_created.bin
                                          {TMPDIR}/rtv_copied.bin)

    # Read-style examples — driver seeds {INPUT}.
    set(_basics_driver_extra --seed-input ${_basics_input_bytes})
    _basics_add_test(iterative-read               {INPUT} {TMPDIR}/out_iter.bin)
    _basics_add_test(iterative-devmem-offset-read {INPUT} {TMPDIR}/out_iter_off.bin)
    _basics_add_test(subregion-write              {INPUT} {TMPDIR}/out_subregion.bin)

    # various-mem-rw exercises MODE = 1 (device), 2 (managed), 3 (pinned-host).
    foreach(mode_pair "1;device" "2;managed" "3;pinned")
        list(GET mode_pair 0 mode_num)
        list(GET mode_pair 1 mode_name)
        add_test(
            NAME basics_various-mem-rw_${mode_name}
            COMMAND
                $<TARGET_FILE:examples_tmpdriver> --base ${AIS_CAPABLE_DIR}
                                                  --seed-input ${_basics_input_bytes}
                -- $<TARGET_FILE:various-mem-rw>
                   {INPUT} {TMPDIR}/out_vmrw_${mode_name}.bin ${mode_num}
        )
        set_tests_properties(basics_various-mem-rw_${mode_name} PROPERTIES
            LABELS "system;hipfile;basics"
        )
        if(mode_name STREQUAL "managed" OR mode_name STREQUAL "pinned")
            set_tests_properties(basics_various-mem-rw_${mode_name} PROPERTIES
                DISABLED TRUE
            )
        endif()
    endforeach()
endif()
