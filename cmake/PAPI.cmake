# ======================================================================================
# PAPI.cmake
#
# Configure papi for rocprofiler-systems
#
# ======================================================================================

include_guard(GLOBAL)

rocprofiler_systems_checkout_git_submodule(
    RELATIVE_PATH external/papi
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    REPO_URL https://bitbucket.org/icl/papi.git
    REPO_BRANCH effd1ef4e0fd4b80e36546791277215a2d6b9eba
    TEST_FILE src/configure
)

set(PAPI_LIBPFM_SOVERSION "4.11.1" CACHE STRING "libpfm.so version")

set(ROCPROFSYS_PAPI_SOURCE_DIR ${PROJECT_BINARY_DIR}/external/papi/source)
set(ROCPROFSYS_PAPI_INSTALL_DIR ${PROJECT_BINARY_DIR}/external/papi/install)

if(NOT EXISTS "${ROCPROFSYS_PAPI_SOURCE_DIR}")
    execute_process(
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/external/papi
            ${ROCPROFSYS_PAPI_SOURCE_DIR}
    )
endif()

if(NOT EXISTS "${ROCPROFSYS_PAPI_INSTALL_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ROCPROFSYS_PAPI_INSTALL_DIR}
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ROCPROFSYS_PAPI_INSTALL_DIR}/include
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib
    )
    execute_process(
        COMMAND
            ${CMAKE_COMMAND} -E touch ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpapi.a
            ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpfm.a
            ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpfm.so
    )
    set(_ROCPROFSYS_PAPI_BUILD_BYPRODUCTS
        ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpapi.a
        ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpfm.a
        ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpfm.so
    )
endif()

# Set ROCPROFSYS_PAPI_CONFIGURE_JOBS for commands that need to be run nonparallel
set(ROCPROFSYS_PAPI_CONFIGURE_JOBS 1)

rocprofiler_systems_add_option(ROCPROFSYS_PAPI_AUTO_COMPONENTS
                               "Automatically enable components" OFF
)

# -------------- PACKAGES -----------------------------------------------------

set(_ROCPROFSYS_VALID_PAPI_COMPONENTS
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
    vmware
)
set(ROCPROFSYS_VALID_PAPI_COMPONENTS
    "${_ROCPROFSYS_VALID_PAPI_COMPONENTS}"
    CACHE STRING
    "Valid PAPI components"
)
mark_as_advanced(ROCPROFSYS_VALID_PAPI_COMPONENTS)

# default components which do not require 3rd-party headers or libraries
set(_ROCPROFSYS_PAPI_COMPONENTS
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

if(ROCPROFSYS_PAPI_AUTO_COMPONENTS)
    # lmsensors
    find_path(
        ROCPROFSYS_PAPI_LMSENSORS_ROOT_DIR
        NAMES include/sensors/sensors.h include/sensors.h
    )

    if(ROCPROFSYS_PAPI_LMSENSORS_ROOT_DIR)
        list(APPEND _ROCPROFSYS_PAPI_COMPONENTS lmsensors)
    endif()

    # pcp
    find_path(ROCPROFSYS_PAPI_PCP_ROOT_DIR NAMES include/pcp/impl.h)
    find_library(ROCPROFSYS_PAPI_PCP_LIBRARY NAMES pcp PATH_SUFFIXES lib lib64)

    if(ROCPROFSYS_PAPI_PCP_ROOT_DIR AND ROCPROFSYS_PAPI_PCP_LIBRARY)
        list(APPEND _ROCPROFSYS_PAPI_COMPONENTS pcp)
    endif()
endif()

# set the ROCPROFSYS_PAPI_COMPONENTS cache variable
set(ROCPROFSYS_PAPI_COMPONENTS
    "${_ROCPROFSYS_PAPI_COMPONENTS}"
    CACHE STRING
    "PAPI components"
)
rocprofiler_systems_add_feature(ROCPROFSYS_PAPI_COMPONENTS "PAPI components")
string(REPLACE ";" "\ " _ROCPROFSYS_PAPI_COMPONENTS "${ROCPROFSYS_PAPI_COMPONENTS}")
set(ROCPROFSYS_PAPI_EXTRA_ENV)

foreach(_COMP ${ROCPROFSYS_PAPI_COMPONENTS})
    string(
        REPLACE
        ";"
        ", "
        _ROCPROFSYS_VALID_PAPI_COMPONENTS_MSG
        "${ROCPROFSYS_VALID_PAPI_COMPONENTS}"
    )
    if(NOT "${_COMP}" IN_LIST ROCPROFSYS_VALID_PAPI_COMPONENTS)
        rocprofiler_systems_message(
            AUTHOR_WARNING
            "ROCPROFSYS_PAPI_COMPONENTS contains an unknown component '${_COMP}'. Known components: ${_ROCPROFSYS_VALID_PAPI_COMPONENTS_MSG}"
        )
    endif()
    unset(_ROCPROFSYS_VALID_PAPI_COMPONENTS_MSG)
endforeach()

if("rocm" IN_LIST ROCPROFSYS_PAPI_COMPONENTS)
    find_package(ROCmVersion REQUIRED)
    list(APPEND ROCPROFSYS_PAPI_EXTRA_ENV PAPI_ROCM_ROOT=${ROCmVersion_DIR})
endif()

if("lmsensors" IN_LIST ROCPROFSYS_PAPI_COMPONENTS AND ROCPROFSYS_PAPI_LMSENSORS_ROOT_DIR)
    list(
        APPEND
        ROCPROFSYS_PAPI_EXTRA_ENV
        PAPI_LMSENSORS_ROOT=${ROCPROFSYS_PAPI_LMSENSORS_ROOT_DIR}
    )
endif()

if("pcp" IN_LIST ROCPROFSYS_PAPI_COMPONENTS AND ROCPROFSYS_PAPI_PCP_ROOT_DIR)
    list(APPEND ROCPROFSYS_PAPI_EXTRA_ENV PAPI_PCP_ROOT=${ROCPROFSYS_PAPI_PCP_ROOT_DIR})
endif()

if(
    "perf_event_uncore" IN_LIST ROCPROFSYS_PAPI_COMPONENTS
    AND NOT "perf_event" IN_LIST ROCPROFSYS_PAPI_COMPONENTS
)
    rocprofiler_systems_message(
        FATAL_ERROR
        "ROCPROFSYS_PAPI_COMPONENTS :: 'perf_event_uncore' requires 'perf_event' component"
    )
endif()

find_program(MAKE_EXECUTABLE NAMES make gmake PATH_SUFFIXES bin)

if(NOT MAKE_EXECUTABLE)
    rocprofiler_systems_message(
        FATAL_ERROR
        "make/gmake executable not found. Please re-run with -DMAKE_EXECUTABLE=/path/to/make"
    )
endif()

set(_PAPI_C_COMPILER ${CMAKE_C_COMPILER})
if(CMAKE_C_COMPILER_IS_CLANG)
    find_program(ROCPROFSYS_GNU_C_COMPILER NAMES gcc)
    if(ROCPROFSYS_GNU_C_COMPILER)
        set(_PAPI_C_COMPILER ${ROCPROFSYS_GNU_C_COMPILER})
    endif()
endif()
set(PAPI_C_COMPILER ${_PAPI_C_COMPILER} CACHE FILEPATH "C compiler used to compile PAPI")

include(ExternalProject)
ExternalProject_Add(
    rocprofiler-systems-papi-build
    PREFIX ${PROJECT_BINARY_DIR}/external/papi
    SOURCE_DIR ${ROCPROFSYS_PAPI_SOURCE_DIR}/src
    BUILD_IN_SOURCE 1
    PATCH_COMMAND
        ${CMAKE_COMMAND} -E env CC=${PAPI_C_COMPILER}
        CFLAGS=-fPIC\ -O3\ -Wno-stringop-truncation\ -Wno-use-after-free LIBS=-lrt
        LDFLAGS=-lrt ${ROCPROFSYS_PAPI_EXTRA_ENV} <SOURCE_DIR>/configure --quiet
        --prefix=${ROCPROFSYS_PAPI_INSTALL_DIR} --with-static-lib=yes --with-shared-lib=no
        --with-perf-events --with-tests=no
        --with-components=${_ROCPROFSYS_PAPI_COMPONENTS}
        --libdir=${ROCPROFSYS_PAPI_INSTALL_DIR}/lib
    CONFIGURE_COMMAND
        ${CMAKE_COMMAND} -E env
        CFLAGS=-fPIC\ -O3\ -Wno-stringop-truncation\ -Wno-use-after-free
        ${ROCPROFSYS_PAPI_EXTRA_ENV} ${MAKE_EXECUTABLE} static install -s -j
        ${ROCPROFSYS_PAPI_CONFIGURE_JOBS}
    BUILD_COMMAND
        ${CMAKE_COMMAND} -E env
        CFLAGS=-fPIC\ -O3\ -Wno-stringop-truncation\ -Wno-use-after-free
        ${ROCPROFSYS_PAPI_EXTRA_ENV} ${MAKE_EXECUTABLE} utils install-utils -s
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_ROCPROFSYS_PAPI_BUILD_BYPRODUCTS}"
)

# target for re-executing the installation
add_custom_target(
    rocprofiler-systems-papi-install
    COMMAND
        ${CMAKE_COMMAND} -E env
        CFLAGS=-fPIC\ -O3\ -Wno-stringop-truncation\ -Wno-use-after-free
        ${ROCPROFSYS_PAPI_EXTRA_ENV} ${MAKE_EXECUTABLE} static install -s
    COMMAND
        ${CMAKE_COMMAND} -E env
        CFLAGS=-fPIC\ -O3\ -Wno-stringop-truncation\ -Wno-use-after-free
        ${ROCPROFSYS_PAPI_EXTRA_ENV} ${MAKE_EXECUTABLE} utils install-utils -s
    WORKING_DIRECTORY ${ROCPROFSYS_PAPI_SOURCE_DIR}/src
    COMMENT "Installing PAPI..."
)

add_custom_target(
    rocprofiler-systems-papi-clean
    COMMAND ${MAKE_EXECUTABLE} distclean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${ROCPROFSYS_PAPI_INSTALL_DIR}/include/*
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/*
    COMMAND
        ${CMAKE_COMMAND} -E touch ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpapi.a
        ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpfm.a
        ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpfm.so
    WORKING_DIRECTORY ${ROCPROFSYS_PAPI_SOURCE_DIR}/src
    COMMENT "Cleaning PAPI..."
)

set(PAPI_ROOT_DIR
    ${ROCPROFSYS_PAPI_INSTALL_DIR}
    CACHE PATH
    "Root PAPI installation"
    FORCE
)
set(PAPI_INCLUDE_DIR
    ${ROCPROFSYS_PAPI_INSTALL_DIR}/include
    CACHE PATH
    "PAPI include folder"
    FORCE
)
set(PAPI_LIBRARY
    ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpapi.a
    CACHE FILEPATH
    "PAPI library"
    FORCE
)
set(PAPI_pfm_LIBRARY
    ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpfm.so
    CACHE FILEPATH
    "PAPI library"
    FORCE
)
set(PAPI_STATIC_LIBRARY
    ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpapi.a
    CACHE FILEPATH
    "PAPI library"
    FORCE
)
set(PAPI_pfm_STATIC_LIBRARY
    ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/libpfm.a
    CACHE FILEPATH
    "PAPI library"
    FORCE
)

target_include_directories(
    rocprofiler-systems-papi
    SYSTEM
    INTERFACE $<BUILD_INTERFACE:${PAPI_INCLUDE_DIR}>
)
target_link_libraries(
    rocprofiler-systems-papi
    INTERFACE $<BUILD_INTERFACE:${PAPI_LIBRARY}> $<BUILD_INTERFACE:${PAPI_pfm_LIBRARY}>
)
rocprofiler_systems_target_compile_definitions(
    rocprofiler-systems-papi INTERFACE ROCPROFSYS_USE_PAPI
                                       $<BUILD_INTERFACE:TIMEMORY_USE_PAPI=1>
)

install(
    DIRECTORY ${ROCPROFSYS_PAPI_INSTALL_DIR}/lib/
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
    COMPONENT papi
    FILES_MATCHING
    PATTERN "*.so*"
)

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
    papi_xml_event_info
)
    string(REPLACE "_" "-" _UTIL_EXE_INSTALL_NAME "${BINARY_NAME_PREFIX}-${_UTIL_EXE}")

    # RPM installer on RedHat/RockyLinux throws error that #!/usr/bin/python should either
    # be #!/usr/bin/python2 or #!/usr/bin/python3
    if(_UTIL_EXE STREQUAL "papi_hl_output_writer.py")
        file(
            READ
            "${PROJECT_BINARY_DIR}/external/papi/source/src/high-level/scripts/${_UTIL_EXE}"
            _HL_OUTPUT_WRITER
        )
        string(
            REPLACE
            "#!/usr/bin/python\n"
            "#!/usr/bin/python3\n"
            _HL_OUTPUT_WRITER
            "${_HL_OUTPUT_WRITER}"
        )
        file(MAKE_DIRECTORY "${ROCPROFSYS_PAPI_INSTALL_DIR}/bin")
        file(
            WRITE
            "${ROCPROFSYS_PAPI_INSTALL_DIR}/bin/${_UTIL_EXE}3"
            "${_HL_OUTPUT_WRITER}"
        )
        set(_UTIL_EXE "${_UTIL_EXE}3")

        # python script file install to libexec
        install(
            PROGRAMS ${ROCPROFSYS_PAPI_INSTALL_DIR}/bin/${_UTIL_EXE}
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
            RENAME ${_UTIL_EXE_INSTALL_NAME}
            COMPONENT papi
            OPTIONAL
        )
    else()
        # Binary files moved to bin
        install(
            PROGRAMS ${ROCPROFSYS_PAPI_INSTALL_DIR}/bin/${_UTIL_EXE}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
            RENAME ${_UTIL_EXE_INSTALL_NAME}
            COMPONENT papi
            OPTIONAL
        )
    endif()
endforeach()
