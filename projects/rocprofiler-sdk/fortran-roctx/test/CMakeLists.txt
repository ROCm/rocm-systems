# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if(NOT ROCTX_FORTRAN_NEEDS_C_PACKAGE)
    message(
        STATUS "rocTX: rocprofiler-sdk-roctx was not found, skipping the Fortran binding "
               "tests (they link it). Point CMAKE_PREFIX_PATH at an installed ROCm to "
               "build them standalone.")
    return()
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    string(REPLACE "<SOURCE>" "-Tf <SOURCE>" CMAKE_Fortran_COMPILE_OBJECT
                   "${CMAKE_Fortran_COMPILE_OBJECT}")
endif()

set_source_files_properties(roctx.f03 PROPERTIES LANGUAGE Fortran)
add_executable(roctx_fortran_test_roctx roctx.f03)
target_link_libraries(roctx_fortran_test_roctx PRIVATE roctx_fortran)
target_compile_options(roctx_fortran_test_roctx
                       PRIVATE $<$<COMPILE_LANG_AND_ID:Fortran,GNU>:-std=gnu>)
add_test(NAME roctx_fortran_roctx COMMAND roctx_fortran_test_roctx)
