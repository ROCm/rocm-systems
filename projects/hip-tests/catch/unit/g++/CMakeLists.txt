# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# AMD specific test
if(HIP_PLATFORM MATCHES "amd" AND UNIX AND GXX_EXEC)
    set(TEST_SRC
        hipMalloc.cc
        half_types.cc
    )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/malloc.o
        COMMAND ${GXX_EXEC} -c
        ${CMAKE_CURRENT_SOURCE_DIR}/hipMalloc.cpp -I${HIP_INCLUDE_DIR}
        -fPIC -D__HIP_PLATFORM_AMD__ -o ${CMAKE_CURRENT_BINARY_DIR}/malloc.o
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hipMalloc.cpp
        COMMENT "Compiling malloc.o with g++")
    add_custom_target(malloc_custom DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/malloc.o)
    add_library(malloc_gpp OBJECT IMPORTED)
    set_property(TARGET malloc_gpp PROPERTY IMPORTED_OBJECTS
        "${CMAKE_CURRENT_BINARY_DIR}/malloc.o")

    add_custom_target(half_custom COMMAND ${GXX_EXEC} -c
        ${CMAKE_CURRENT_SOURCE_DIR}/half_types.cpp -I${HIP_INCLUDE_DIR}
        -fPIC -D__HIP_PLATFORM_AMD__ -o half_types.o BYPRODUCTS half_types.o)
    add_library(half_types OBJECT IMPORTED)
    set_property(TARGET half_types PROPERTY IMPORTED_OBJECTS
        "${CMAKE_CURRENT_BINARY_DIR}/half_types.o")

    set(GPP_LIBS malloc_gpp half_types)
    set(GPP_DEPS malloc_custom half_custom)

    # Low precision headers need __bf16/_Float16. g++ provides these from v13
    # on architectures that implement them (e.g. x86_64, aarch64), but NOT on
    # others such as ppc64le, where g++ 13+ still rejects them with
    # "_Float16 is not supported on this target". Probe the host compiler for
    # the types instead of assuming the version implies support.
    execute_process(COMMAND ${GXX_EXEC} -dumpversion
                    OUTPUT_VARIABLE GXX_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(GXX_HAS_LOW_PRECISION_FP OFF)
    if(GXX_VERSION VERSION_GREATER_EQUAL 13)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/gpp_fp_probe.cpp
             "int main() { __bf16 b{}; _Float16 h{}; return (int)(sizeof(b) + sizeof(h)); }\n")
        execute_process(
            COMMAND ${GXX_EXEC} -std=c++17 -fsyntax-only
                    ${CMAKE_CURRENT_BINARY_DIR}/gpp_fp_probe.cpp
            RESULT_VARIABLE GXX_FP_PROBE_RESULT OUTPUT_QUIET ERROR_QUIET)
        if(GXX_FP_PROBE_RESULT EQUAL 0)
            set(GXX_HAS_LOW_PRECISION_FP ON)
        endif()
    endif()
    if(GXX_HAS_LOW_PRECISION_FP)
        list(APPEND TEST_SRC fp_headers.cc)
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fp_headers.o
            COMMAND ${GXX_EXEC} -c -std=c++17 -Wall -Wextra -Werror
            ${CMAKE_CURRENT_SOURCE_DIR}/fp_headers.cpp -I${HIP_INCLUDE_DIR}
            -fPIC -D__HIP_PLATFORM_AMD__ -o ${CMAKE_CURRENT_BINARY_DIR}/fp_headers.o
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fp_headers.cpp
            COMMENT "Compiling fp_headers.o with g++")
        add_custom_target(fp_headers_custom DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fp_headers.o)
        add_library(fp_headers_gpp OBJECT IMPORTED)
        set_property(TARGET fp_headers_gpp PROPERTY IMPORTED_OBJECTS
            "${CMAKE_CURRENT_BINARY_DIR}/fp_headers.o")
        list(APPEND GPP_LIBS fp_headers_gpp)
        list(APPEND GPP_DEPS fp_headers_custom)

        # Same TU through clang in plain C++ mode, compile only.
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fp_headers_clang.stamp
                COMMAND ${CMAKE_CXX_COMPILER} -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror
                ${CMAKE_CURRENT_SOURCE_DIR}/fp_headers.cpp -I${HIP_INCLUDE_DIR}
                -D__HIP_PLATFORM_AMD__
                COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/fp_headers_clang.stamp
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fp_headers.cpp
                COMMENT "Checking fp_headers.cpp with clang -x c++")
            add_custom_target(fp_headers_clang_check
                DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/fp_headers_clang.stamp)
            list(APPEND GPP_DEPS fp_headers_clang_check)
        endif()
    endif()

    hip_add_exe_to_target(NAME gppTests
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests
                        LINKER_LIBS ${GPP_LIBS})

    add_dependencies(gppTests ${GPP_DEPS})
endif()
