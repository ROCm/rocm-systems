# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# The ROCTx binding: the `roctx` module, roc::roctx_fortran and the
# roctx-fortran package. Included by CMakeLists.txt, which ran the guard and
# set _fc_name. ROCTx has no array overloads, so FORTRAN_ARRAY_INTERFACES does
# not apply.

add_library(roctx_fortran STATIC roctx.F90)
add_library(roc::roctx_fortran ALIAS roctx_fortran)

# Each module compiles to one object, and a linker extracts whole objects from
# an archive, so a program that calls one wrapper links all of them. One section
# per procedure lets a program linked with -Wl,--gc-sections keep only the
# wrappers it calls, and still link when another wrapper names a C function the
# installed library does not export. Only these compilers split the sections
# (the Intel ones accept the flags but do not).
target_compile_options(
    roctx_fortran
    PRIVATE
        "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,LLVMFlang,Flang,NVHPC>:-ffunction-sections;-fdata-sections>"
)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(ROCTX_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
endif()

# Not version-specific: a site installing two versions of one compiler sets it.
set(ROCTX_FORTRAN_COMPILER_DIR
    "${_fc_name}"
    CACHE STRING
    "Per-compiler subdirectory the ROCTx .mod and archive install under"
)

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include/roctx")
set_target_properties(
    roctx_fortran
    PROPERTIES
        Fortran_MODULE_DIRECTORY "${_moddir}"
        Fortran_PREPROCESS ON
        Fortran_FORMAT FREE
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
    roctx_fortran
    PUBLIC
        "$<BUILD_INTERFACE:${_moddir}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCTX_FORTRAN_COMPILER_DIR}>"
)

if(NOT TARGET rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
    if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
        enable_language(C)
    endif()
    find_package(rocprofiler-sdk-roctx QUIET)
endif()

# ROCTX_FORTRAN_NEEDS_C_PACKAGE tells the config package whether to find
# rocprofiler-sdk-roctx.
if(TARGET rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
    target_link_libraries(
        roctx_fortran
        INTERFACE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
    )
    set(ROCTX_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
    set(ROCTX_FORTRAN_NEEDS_C_PACKAGE FALSE)
    message(
        STATUS
        "ROCTx: rocprofiler-sdk-roctx was not found, so the Fortran bindings carry "
        "no link interface; a consumer will have to link it "
        "(-lrocprofiler-sdk-roctx) itself."
    )
endif()

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/roctx-fortran")

install(
    TARGETS roctx_fortran
    EXPORT roctx-fortran-targets
    ARCHIVE
        DESTINATION
            "${CMAKE_INSTALL_LIBDIR}/fortran/${ROCTX_FORTRAN_COMPILER_DIR}"
        COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)
install(
    DIRECTORY "${_moddir}/"
    DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCTX_FORTRAN_COMPILER_DIR}"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
    FILES_MATCHING
    PATTERN "*.mod"
)
install(
    EXPORT roctx-fortran-targets
    FILE roctx-fortran-targets.cmake
    NAMESPACE roc::
    DESTINATION "${_fortran_pkgdir}"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)

set(ROCTX_FORTRAN_LIBDIR "${CMAKE_INSTALL_LIBDIR}/fortran")
set(ROCTX_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/roctx-fortran-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config.cmake"
    INSTALL_DESTINATION "${_fortran_pkgdir}"
    # Not ROCTX_FORTRAN_COMPILER_DIR: it is a bare name, not a path under the prefix.
    PATH_VARS ROCTX_FORTRAN_LIBDIR ROCTX_FORTRAN_INCLUDEDIR
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config.cmake"
    DESTINATION "${_fortran_pkgdir}"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)

# From rocprofiler-sdk's source VERSION file, which versions rocprofiler-sdk-roctx.
set(_roctx_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../rocprofiler-sdk/VERSION")
    file(
        STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../rocprofiler-sdk/VERSION"
        _roctx_version
        LIMIT_COUNT 1
    )
    string(
        REGEX MATCH
        "^[0-9]+\\.[0-9]+\\.[0-9]+"
        _roctx_version
        "${_roctx_version}"
    )
endif()

if(_roctx_version)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config-version.cmake"
        VERSION "${_roctx_version}"
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/roctx-fortran-config-version.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
    )
else()
    message(
        STATUS
        "ROCTx: could not determine the version from rocprofiler-sdk's VERSION "
        "file, so no roctx-fortran-config-version.cmake is installed and a "
        "versioned find_package(roctx-fortran) will not match."
    )
endif()

# Under share/, not include/fortran/, whose entries the config lists as compilers.
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/roctx.F90"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/roctx/fortran"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)
