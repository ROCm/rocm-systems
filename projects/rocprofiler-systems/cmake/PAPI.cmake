# ======================================================================================
# PAPI.cmake
#
# Configure papi for omnitrace
#
# ======================================================================================

include_guard(GLOBAL)

omnitrace_checkout_git_submodule(
    RELATIVE_PATH external/papi
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://bitbucket.org/icl/papi.git
    REPO_BRANCH effd1ef4e0fd4b80e36546791277215a2d6b9eba
    TEST_FILE src/configure)

set(PAPI_LIBPFM_SOVERSION
    "4.11.1"
    CACHE STRING "libpfm.so version")

set(OMNITRACE_PAPI_SOURCE_DIR ${PROJECT_BINARY_DIR}/external/papi/source)
set(OMNITRACE_PAPI_INSTALL_DIR ${PROJECT_BINARY_DIR}/external/papi/install)

if(NOT EXISTS "${OMNITRACE_PAPI_SOURCE_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/external/papi
                ${OMNITRACE_PAPI_SOURCE_DIR})
endif()

if(NOT EXISTS "${OMNITRACE_PAPI_INSTALL_DIR}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
                            ${OMNITRACE_PAPI_INSTALL_DIR})
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
                            ${OMNITRACE_PAPI_INSTALL_DIR}/include)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
                            ${OMNITRACE_PAPI_INSTALL_DIR}/lib)
    execute_process(
        COMMAND
            ${CMAKE_COMMAND} -E touch ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpapi.a
            ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpfm.a
            ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpfm.so)
    set(_OMNITRACE_PAPI_BUILD_BYPRODUCTS
        ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpapi.a
        ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpfm.a
        ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpfm.so)
endif()

omnitrace_add_option(OMNITRACE_PAPI_AUTO_COMPONENTS "Automatically enable components" OFF)

# -------------- PACKAGES -----------------------------------------------------

set(_OMNITRACE_VALID_PAPI_COMPONENTS
    appio
    bgpm
    coretemp
    coretemp_freebsd
    cuda
    emon
    example
    host_micpower
    infiniband
    intel_gpu
    io
    libmsr
    lmsensors
    lustre
    micpower
    mx
    net
    nvml
    pcp
    perfctr
    perfctr_ppc
    perf_event
    perf_event_uncore
    perfmon2
    perfmon_ia64
    perfnec
    powercap
    powercap_ppc
    rapl
    rocm
    rocm_smi
    sde
    sensors_ppc
    stealtime
    sysdetect
    vmware)
set(OMNITRACE_VALID_PAPI_COMPONENTS
    "${_OMNITRACE_VALID_PAPI_COMPONENTS}"
    CACHE STRING "Valid PAPI components")
mark_as_advanced(OMNITRACE_VALID_PAPI_COMPONENTS)

# default components which do not require 3rd-party headers or libraries
set(_OMNITRACE_PAPI_COMPONENTS
    appio
    coretemp
    io
    infiniband
    # lustre micpower mx
    net
    perf_event
    perf_event_uncore
    # rapl stealtime
    )

if(OMNITRACE_PAPI_AUTO_COMPONENTS)
    # rocm
    if(OMNITRACE_USE_HIP
       OR OMNITRACE_USE_ROCTRACER
       OR OMNITRACE_USE_ROCM_SMI)
        list(APPEND _OMNITRACE_PAPI_COMPONENTS rocm)
    endif()

    # lmsensors
    find_path(OMNITRACE_PAPI_LMSENSORS_ROOT_DIR NAMES include/sensors/sensors.h
                                                      include/sensors.h)

    if(OMNITRACE_PAPI_LMSENSORS_ROOT_DIR)
        list(APPEND _OMNITRACE_PAPI_COMPONENTS lmsensors)
    endif()

    # pcp
    find_path(OMNITRACE_PAPI_PCP_ROOT_DIR NAMES include/pcp/impl.h)
    find_library(
        OMNITRACE_PAPI_PCP_LIBRARY
        NAMES pcp
        PATH_SUFFIXES lib lib64)

    if(OMNITRACE_PAPI_PCP_ROOT_DIR AND OMNITRACE_PAPI_PCP_LIBRARY)
        list(APPEND _OMNITRACE_PAPI_COMPONENTS pcp)
    endif()
endif()

# set the OMNITRACE_PAPI_COMPONENTS cache variable
set(OMNITRACE_PAPI_COMPONENTS
    "${_OMNITRACE_PAPI_COMPONENTS}"
    CACHE STRING "PAPI components")
omnitrace_add_feature(OMNITRACE_PAPI_COMPONENTS "PAPI components")
string(REPLACE ";" "\ " _OMNITRACE_PAPI_COMPONENTS "${OMNITRACE_PAPI_COMPONENTS}")
set(OMNITRACE_PAPI_EXTRA_ENV)

foreach(_COMP ${OMNITRACE_PAPI_COMPONENTS})
    string(REPLACE ";" ", " _OMNITRACE_VALID_PAPI_COMPONENTS_MSG
                   "${OMNITRACE_VALID_PAPI_COMPONENTS}")
    if(NOT "${_COMP}" IN_LIST OMNITRACE_VALID_PAPI_COMPONENTS)
        omnitrace_message(
            AUTHOR_WARNING
            "OMNITRACE_PAPI_COMPONENTS contains an unknown component '${_COMP}'. Known components: ${_OMNITRACE_VALID_PAPI_COMPONENTS_MSG}"
            )
    endif()
    unset(_OMNITRACE_VALID_PAPI_COMPONENTS_MSG)
endforeach()

if("rocm" IN_LIST OMNITRACE_PAPI_COMPONENTS)
    find_package(ROCmVersion REQUIRED)
    list(APPEND OMNITRACE_PAPI_EXTRA_ENV PAPI_ROCM_ROOT=${ROCmVersion_DIR})
endif()

if("lmsensors" IN_LIST OMNITRACE_PAPI_COMPONENTS AND OMNITRACE_PAPI_LMSENSORS_ROOT_DIR)
    list(APPEND OMNITRACE_PAPI_EXTRA_ENV
         PAPI_LMSENSORS_ROOT=${OMNITRACE_PAPI_LMSENSORS_ROOT_DIR})
endif()

if("pcp" IN_LIST OMNITRACE_PAPI_COMPONENTS AND OMNITRACE_PAPI_PCP_ROOT_DIR)
    list(APPEND OMNITRACE_PAPI_EXTRA_ENV PAPI_PCP_ROOT=${OMNITRACE_PAPI_PCP_ROOT_DIR})
endif()

if("perf_event_uncore" IN_LIST OMNITRACE_PAPI_COMPONENTS AND NOT "perf_event" IN_LIST
                                                             OMNITRACE_PAPI_COMPONENTS)
    omnitrace_message(
        FATAL_ERROR
        "OMNITRACE_PAPI_COMPONENTS :: 'perf_event_uncore' requires 'perf_event' component"
        )
endif()

find_program(
    MAKE_EXECUTABLE
    NAMES make gmake
    PATH_SUFFIXES bin)

if(NOT MAKE_EXECUTABLE)
    omnitrace_message(
        FATAL_ERROR
        "make/gmake executable not found. Please re-run with -DMAKE_EXECUTABLE=/path/to/make"
        )
endif()

set(_PAPI_C_COMPILER ${CMAKE_C_COMPILER})
if(CMAKE_C_COMPILER_IS_CLANG)
    find_program(OMNITRACE_GNU_C_COMPILER NAMES gcc)
    if(OMNITRACE_GNU_C_COMPILER)
        set(_PAPI_C_COMPILER ${OMNITRACE_GNU_C_COMPILER})
    endif()
endif()
set(PAPI_C_COMPILER
    ${_PAPI_C_COMPILER}
    CACHE FILEPATH "C compiler used to compile PAPI")

include(ExternalProject)
externalproject_add(
    omnitrace-papi-build
    PREFIX ${PROJECT_BINARY_DIR}/external/papi
    SOURCE_DIR ${OMNITRACE_PAPI_SOURCE_DIR}/src
    BUILD_IN_SOURCE 1
    PATCH_COMMAND
        ${CMAKE_COMMAND} -E env CC=${PAPI_C_COMPILER}
        CFLAGS=-fPIC\ -O3\ -Wno-stringop-truncation LIBS=-lrt LDFLAGS=-lrt
        ${OMNITRACE_PAPI_EXTRA_ENV} <SOURCE_DIR>/configure
        --prefix=${OMNITRACE_PAPI_INSTALL_DIR} --with-static-lib=yes --with-shared-lib=no
        --with-perf-events --with-tests=no --with-components=${_OMNITRACE_PAPI_COMPONENTS}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC\ -O3\ -Wno-stringop-truncation
                      ${OMNITRACE_PAPI_EXTRA_ENV} ${MAKE_EXECUTABLE} static install
    BUILD_COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC\ -O3\ -Wno-stringop-truncation
                  ${OMNITRACE_PAPI_EXTRA_ENV} ${MAKE_EXECUTABLE} utils install-utils
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_OMNITRACE_PAPI_BUILD_BYPRODUCTS}")

# target for re-executing the installation
add_custom_target(
    omnitrace-papi-install
    COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC\ -O3\ -Wno-stringop-truncation
            ${OMNITRACE_PAPI_EXTRA_ENV} ${MAKE_EXECUTABLE} static install
    COMMAND ${CMAKE_COMMAND} -E env CFLAGS=-fPIC\ -O3\ -Wno-stringop-truncation
            ${OMNITRACE_PAPI_EXTRA_ENV} ${MAKE_EXECUTABLE} utils install-utils
    WORKING_DIRECTORY ${OMNITRACE_PAPI_SOURCE_DIR}/src
    COMMENT "Installing PAPI...")

add_custom_target(
    omnitrace-papi-clean
    COMMAND ${MAKE_EXECUTABLE} distclean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${OMNITRACE_PAPI_INSTALL_DIR}/include/*
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${OMNITRACE_PAPI_INSTALL_DIR}/lib/*
    COMMAND
        ${CMAKE_COMMAND} -E touch ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpapi.a
        ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpfm.a
        ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpfm.so
    WORKING_DIRECTORY ${OMNITRACE_PAPI_SOURCE_DIR}/src
    COMMENT "Cleaning PAPI...")

set(PAPI_ROOT_DIR
    ${OMNITRACE_PAPI_INSTALL_DIR}
    CACHE PATH "Root PAPI installation" FORCE)
set(PAPI_INCLUDE_DIR
    ${OMNITRACE_PAPI_INSTALL_DIR}/include
    CACHE PATH "PAPI include folder" FORCE)
set(PAPI_LIBRARY
    ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpapi.a
    CACHE FILEPATH "PAPI library" FORCE)
set(PAPI_pfm_LIBRARY
    ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpfm.so
    CACHE FILEPATH "PAPI library" FORCE)
set(PAPI_STATIC_LIBRARY
    ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpapi.a
    CACHE FILEPATH "PAPI library" FORCE)
set(PAPI_pfm_STATIC_LIBRARY
    ${OMNITRACE_PAPI_INSTALL_DIR}/lib/libpfm.a
    CACHE FILEPATH "PAPI library" FORCE)

target_include_directories(omnitrace-papi SYSTEM
                           INTERFACE $<BUILD_INTERFACE:${PAPI_INCLUDE_DIR}>)
target_link_libraries(omnitrace-papi INTERFACE $<BUILD_INTERFACE:${PAPI_LIBRARY}>
                                               $<BUILD_INTERFACE:${PAPI_pfm_LIBRARY}>)
omnitrace_target_compile_definitions(
    omnitrace-papi INTERFACE OMNITRACE_USE_PAPI $<BUILD_INTERFACE:TIMEMORY_USE_PAPI=1>)

install(
    DIRECTORY ${OMNITRACE_PAPI_INSTALL_DIR}/lib/
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/omnitrace
    COMPONENT papi
    FILES_MATCHING
    PATTERN "*.so*")

foreach(
    _UTIL_EXE
    papi_avail
    papi_clockres
    papi_command_line
    papi_component_avail
    papi_cost
    papi_decode
    papi_error_codes
    papi_event_chooser
    papi_hardware_avail
    papi_hl_output_writer.py
    papi_mem_info
    papi_multiplex_cost
    papi_native_avail
    papi_version
    papi_xml_event_info)
    string(REPLACE "_" "-" _UTIL_EXE_INSTALL_NAME "omnitrace-${_UTIL_EXE}")
    install(
        PROGRAMS ${OMNITRACE_PAPI_INSTALL_DIR}/bin/${_UTIL_EXE}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        RENAME ${_UTIL_EXE_INSTALL_NAME}
        COMPONENT papi
        OPTIONAL)
endforeach()
