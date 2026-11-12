# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# The ROCTx binding test: one test, no GPU needed.

# The Intel compilers skip .f03 and .f08 sources unless -Tf precedes them.
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    string(
        REPLACE
        "<SOURCE>"
        "-Tf <SOURCE>"
        CMAKE_Fortran_COMPILE_OBJECT
        "${CMAKE_Fortran_COMPILE_OBJECT}"
    )
endif()

# Keep only the binding wrappers a test calls (the archive has one section per
# procedure), so a test links even when another wrapper names a C symbol the
# installed library does not export. Spelled -Wl, rather than LINKER:, which
# CMake 3.27 passes to LLVMFlang as a bare --gc-sections.
if(NOT APPLE AND NOT WIN32)
    add_link_options(-Wl,--gc-sections)
endif()

# The parent found rocprofiler-sdk-roctx; the test links it.
if(NOT TARGET rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
    message(
        STATUS
        "ROCTx: rocprofiler-sdk-roctx was not found, skipping the Fortran binding "
        "test (it links it). Point CMAKE_PREFIX_PATH at an installed ROCm to "
        "build it."
    )
    return()
endif()

set_source_files_properties(roctx.f03 PROPERTIES LANGUAGE Fortran)
add_executable(roctx_fortran_test_roctx roctx.f03)
set_target_properties(
    roctx_fortran_test_roctx
    PROPERTIES LINKER_LANGUAGE Fortran
)
target_link_libraries(roctx_fortran_test_roctx PRIVATE roctx_fortran)
add_test(NAME roctx_fortran_roctx COMMAND roctx_fortran_test_roctx)

# Installed test: the executable goes to bin/, and a CTestTestfile.cmake with
# the same test to bin/roctx-fortran/, for ctest --test-dir.
include(GNUInstallDirs)
install(
    TARGETS roctx_fortran_test_roctx
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT tests
)
set(_ctest "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-CTestTestfile.cmake")
file(
    WRITE "${_ctest}"
    "add_test(roctx_fortran_roctx \"../roctx_fortran_test_roctx\")\n"
)
install(
    FILES "${_ctest}"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/roctx-fortran"
    COMPONENT tests
    RENAME CTestTestfile.cmake
)
