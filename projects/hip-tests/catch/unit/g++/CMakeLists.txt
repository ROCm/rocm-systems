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

    hip_add_exe_to_target(NAME gppTests
                        TEST_SRC ${TEST_SRC}
                        TEST_TARGET_NAME build_tests
                        LINKER_LIBS malloc_gpp half_types)

    add_dependencies(gppTests malloc_custom half_custom)
endif()
