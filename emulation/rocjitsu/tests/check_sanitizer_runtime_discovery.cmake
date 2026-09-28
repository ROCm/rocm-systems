# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED ROCJITSU_SOURCE_DIR)
    message(FATAL_ERROR "ROCJITSU_SOURCE_DIR is required")
endif()
if(NOT DEFINED ROCJITSU_TEST_BINARY_DIR)
    message(FATAL_ERROR "ROCJITSU_TEST_BINARY_DIR is required")
endif()

include("${ROCJITSU_SOURCE_DIR}/cmake/rj_sanitizers.cmake")

function(check_clang_runtime_layout layout use_arch_suffix)
    set(test_root "${ROCJITSU_TEST_BINARY_DIR}/sanitizer-runtime-${layout}")
    set(resource_dir "${test_root}/resource")
    set(runtime_dir "${resource_dir}/lib/test-target")
    set(mock_compiler "${test_root}/mock-clang")
    file(REMOVE_RECURSE "${test_root}")
    file(MAKE_DIRECTORY "${runtime_dir}")

    if(use_arch_suffix)
        set(asan_name "libclang_rt.asan-x86_64.so")
        set(ubsan_name "libclang_rt.ubsan_standalone-x86_64.so")
        set(tsan_name "libclang_rt.tsan-x86_64.so")
    else()
        set(asan_name "libclang_rt.asan.so")
        set(ubsan_name "libclang_rt.ubsan_standalone.so")
        set(tsan_name "libclang_rt.tsan.so")
    endif()

    set(expected_libraries)
    foreach(runtime_name IN ITEMS "${asan_name}" "${ubsan_name}" "${tsan_name}")
        set(runtime_path "${runtime_dir}/${runtime_name}")
        file(WRITE "${runtime_path}" "")
        list(APPEND expected_libraries "${runtime_path}")
    endforeach()

    set(mock_compiler_template
        [=[#!/bin/sh
case "$1" in
  --print-resource-dir) printf '%s\n' '@resource_dir@' ;;
  --print-file-name=*) printf '%s\n' "${1#*=}" ;;
  *) exit 1 ;;
esac
]=]
    )
    string(CONFIGURE "${mock_compiler_template}" mock_compiler_contents @ONLY)
    file(WRITE "${mock_compiler}" "${mock_compiler_contents}")
    file(
        CHMOD
        "${mock_compiler}"
        PERMISSIONS
            OWNER_READ
            OWNER_WRITE
            OWNER_EXECUTE
            GROUP_READ
            GROUP_EXECUTE
            WORLD_READ
            WORLD_EXECUTE
    )

    rj_find_sanitizer_shared_libraries(
        discovered_libraries
        discovered_asan
        discovered_tsan
        COMPILER "${mock_compiler}"
        COMPILER_ID Clang
        SYSTEM_PROCESSOR x86_64
        SANITIZERS address undefined thread
    )

    if(NOT "${discovered_libraries}" STREQUAL "${expected_libraries}")
        message(
            FATAL_ERROR
            "${layout}: expected '${expected_libraries}', got "
            "'${discovered_libraries}'"
        )
    endif()
    if(NOT "${discovered_asan}" STREQUAL "${runtime_dir}/${asan_name}")
        message(FATAL_ERROR "${layout}: incorrect ASan runtime")
    endif()
    if(NOT "${discovered_tsan}" STREQUAL "${runtime_dir}/${tsan_name}")
        message(FATAL_ERROR "${layout}: incorrect TSan runtime")
    endif()
endfunction()

check_clang_runtime_layout(legacy TRUE)
check_clang_runtime_layout(target-directory FALSE)
