# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Standalone Linux regression: cmake -P check_sanitizer_runtime_lookup.cmake
cmake_minimum_required(VERSION 3.20)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/rj_sanitizers.cmake")
if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "This mock compiler fixture requires Linux")
endif()
if(NOT TEST_ROOT)
    set(TEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/sanitizer-runtime-lookup-test")
endif()
file(MAKE_DIRECTORY "${TEST_ROOT}")

function(make_compiler name runtime_dir resource_dir direct_dir)
    set(_compiler "${TEST_ROOT}/${name}-compiler")
    file(WRITE "${_compiler}" "#!/bin/sh\n"
        "# Require the compiler argument so target selection is tested.\n"
        "test \"$1\" = '--target=x86_64-unknown-linux-gnu' || exit 2\nshift\n"
        "case \"$1\" in\n"
        "--print-runtime-dir) printf '%s\\n' '${runtime_dir}' ;;\n"
        "--print-resource-dir) printf '%s\\n' '${resource_dir}' ;;\n"
        "--print-file-name=*) name=\"\${1#*=}\"; "
        "if test -f '${direct_dir}'/\"$name\"; then "
        "printf '%s/%s\\n' '${direct_dir}' \"$name\"; "
        "else printf '%s\\n' \"$name\"; fi ;;\n"
        "*) exit 1 ;;\nesac\n")
    file(CHMOD "${_compiler}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
    set(COMPILER "${_compiler}" PARENT_SCOPE)
endfunction()

function(check_runtime compiler_id expected_asan expected_tsan expected_ubsan)
    rj_find_sanitizer_shared_libraries(libraries asan tsan
        COMPILER "${COMPILER}" COMPILER_ID "${compiler_id}"
        COMPILER_ARG1 --target=x86_64-unknown-linux-gnu
        SYSTEM_PROCESSOR x86_64 SANITIZERS address thread undefined)
    if(NOT asan STREQUAL expected_asan OR NOT tsan STREQUAL expected_tsan)
        message(FATAL_ERROR "Wrong runtime selection: ${asan};${tsan}")
    endif()
    if(NOT libraries STREQUAL "${expected_asan};${expected_tsan};${expected_ubsan}")
        message(FATAL_ERROR "Wrong runtime list: ${libraries}")
    endif()
    rj_sanitizer_runtime_link_options(options COMPILER_ID "${compiler_id}"
        SHARED ON SHARED_LIBRARIES ${libraries} SANITIZERS address thread undefined)
    get_filename_component(runtime_dir "${expected_asan}" DIRECTORY)
    if(NOT "-Wl,-rpath,${runtime_dir}" IN_LIST options)
        message(FATAL_ERROR "Runtime directory missing from rpath: ${options}")
    endif()
endfunction()

# Clang 24 per-target layout, including a foreign target directory that must
# never be selected just because it sorts first.
set(resource "${TEST_ROOT}/clang24")
set(selected "${resource}/lib/x86_64-unknown-linux-gnu")
file(MAKE_DIRECTORY "${selected}" "${resource}/lib/aarch64-unknown-linux-gnu")
foreach(runtime asan tsan ubsan_standalone)
    file(WRITE "${selected}/libclang_rt.${runtime}.so" "")
    file(WRITE "${resource}/lib/aarch64-unknown-linux-gnu/libclang_rt.${runtime}.so" "")
endforeach()
make_compiler(clang24 "${selected}" "${resource}" "${TEST_ROOT}/missing")
check_runtime(Clang "${selected}/libclang_rt.asan.so"
    "${selected}/libclang_rt.tsan.so" "${selected}/libclang_rt.ubsan_standalone.so")

# Traditional Clang layout and resource-dir fallback remain supported.
set(legacy "${TEST_ROOT}/clang18/lib/linux")
file(MAKE_DIRECTORY "${legacy}")
foreach(runtime asan tsan ubsan_standalone)
    file(WRITE "${legacy}/libclang_rt.${runtime}-x86_64.so" "")
endforeach()
foreach(mode direct fallback)
    if(mode STREQUAL "direct")
        set(direct_dir "${legacy}")
    else()
        set(direct_dir "${TEST_ROOT}/missing")
    endif()
    make_compiler("clang18-${mode}" "" "${TEST_ROOT}/clang18" "${direct_dir}")
    check_runtime(Clang "${legacy}/libclang_rt.asan-x86_64.so"
        "${legacy}/libclang_rt.tsan-x86_64.so" "${legacy}/libclang_rt.ubsan_standalone-x86_64.so")
endforeach()

set(gcc "${TEST_ROOT}/gcc/lib")
file(MAKE_DIRECTORY "${gcc}")
foreach(runtime asan tsan ubsan)
    file(WRITE "${gcc}/lib${runtime}.so" "")
endforeach()
make_compiler(gcc "" "" "${gcc}")
check_runtime(GNU "${gcc}/libasan.so" "${gcc}/libtsan.so" "${gcc}/libubsan.so")
message(STATUS "Sanitizer runtime lookup tests passed")
