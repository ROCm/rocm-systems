# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# The HIP binding: the `hip` module, hip::hip_fortran and the hip-fortran
# package. Included by CMakeLists.txt, which ran the guard and set _fc_name.

add_library(hip_fortran STATIC hip.F90)
add_library(hip::hip_fortran ALIAS hip_fortran)

# Each module compiles to one object, and a linker extracts whole objects from
# an archive, so a program that calls one wrapper links all of them. One section
# per procedure lets a program linked with -Wl,--gc-sections keep only the
# wrappers it calls, and still link when another wrapper names a C function the
# installed library does not export. Only these compilers split the sections
# (the Intel ones accept the flags but do not).
target_compile_options(
    hip_fortran
    PRIVATE
        "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,LLVMFlang,Flang,NVHPC>:-ffunction-sections;-fdata-sections>"
)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HIP_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
endif()

# Not version-specific: a site installing two versions of one compiler sets it.
set(HIP_FORTRAN_COMPILER_DIR
    "${_fc_name}"
    CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under"
)

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include/hip")
set_target_properties(
    hip_fortran
    PROPERTIES
        Fortran_MODULE_DIRECTORY "${_moddir}"
        Fortran_PREPROCESS ON
        Fortran_FORMAT FREE
        POSITION_INDEPENDENT_CODE ON
)

if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
    target_compile_definitions(hip_fortran PRIVATE USE_ASSUMED_SHAPE=1)
elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
    target_compile_definitions(hip_fortran PRIVATE USE_ASSUMED_RANK=1)
endif()

# No line-length flag: the generated source stays within 131 columns.

target_include_directories(
    hip_fortran
    PUBLIC
        "$<BUILD_INTERFACE:${_moddir}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}>"
)

# hip-config.cmake needs C or CXX enabled for FindThreads.
if(NOT TARGET hip::host)
    if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
        enable_language(C)
    endif()
    find_package(hip QUIET)
endif()

# HIP_FORTRAN_NEEDS_C_PACKAGE tells the config package whether to find hip.
if(TARGET hip::host)
    target_link_libraries(hip_fortran INTERFACE hip::host)
    set(HIP_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
    set(HIP_FORTRAN_NEEDS_C_PACKAGE FALSE)
    message(
        STATUS
        "HIP: the runtime was not found, so the Fortran bindings carry no link "
        "interface; a consumer will have to link it (-lamdhip64) itself."
    )
endif()

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/hip-fortran")

install(
    TARGETS hip_fortran
    EXPORT hip-fortran-targets
    ARCHIVE
        DESTINATION
            "${CMAKE_INSTALL_LIBDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}"
        COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)
install(
    DIRECTORY "${_moddir}/"
    DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIP_FORTRAN_COMPILER_DIR}"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
    FILES_MATCHING
    PATTERN "*.mod"
)
install(
    EXPORT hip-fortran-targets
    FILE hip-fortran-targets.cmake
    NAMESPACE hip::
    DESTINATION "${_fortran_pkgdir}"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)

set(HIP_FORTRAN_LIBDIR "${CMAKE_INSTALL_LIBDIR}/fortran")
set(HIP_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hip-fortran-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config.cmake"
    INSTALL_DESTINATION "${_fortran_pkgdir}"
    # Not HIP_FORTRAN_COMPILER_DIR: it is a bare name, not a path under the prefix.
    PATH_VARS HIP_FORTRAN_LIBDIR HIP_FORTRAN_INCLUDEDIR
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config.cmake"
    DESTINATION "${_fortran_pkgdir}"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)

# From HIP's source VERSION file, as hipamd does; hip_VERSION is the INSTALLED HIP.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../hip/VERSION")
    file(
        STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../hip/VERSION"
        _hip_vlist
        REGEX "^[0-9]+"
    )
    list(LENGTH _hip_vlist _hip_vn)
    if(_hip_vn GREATER_EQUAL 3)
        list(GET _hip_vlist 0 1 2 _hip_v)
        list(JOIN _hip_v "." _fortran_version)
    endif()
    unset(_hip_vlist)
endif()

if(_fortran_version)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config-version.cmake"
        VERSION "${_fortran_version}"
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/hip-fortran-config-version.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
    )
else()
    message(
        STATUS
        "HIP: could not determine the version from the VERSION file, so no "
        "hip-fortran-config-version.cmake is installed and a versioned "
        "find_package(hip-fortran) will not match."
    )
endif()

# Under share/, not include/fortran/, whose entries the config lists as compilers.
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/hip.F90"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/hip/fortran"
    COMPONENT ${HIP_FORTRAN_INSTALL_COMPONENT}
)
