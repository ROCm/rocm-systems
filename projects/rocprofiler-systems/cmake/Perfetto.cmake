# ======================================================================================
# Perfetto.cmake
#
# Configure perfetto for omnitrace
#
# ======================================================================================

include_guard(GLOBAL)

include(ExternalProject)
include(ProcessorCount)

# ---------------------------------------------------------------------------------------#
#
# executables and libraries
#
# ---------------------------------------------------------------------------------------#

find_program(
    OMNITRACE_COPY_EXECUTABLE
    NAMES cp
    PATH_SUFFIXES bin)

find_program(
    OMNITRACE_NINJA_EXECUTABLE
    NAMES ninja
    PATH_SUFFIXES bin)

mark_as_advanced(OMNITRACE_COPY_EXECUTABLE)
mark_as_advanced(OMNITRACE_NINJA_EXECUTABLE)

# ---------------------------------------------------------------------------------------#
#
# variables
#
# ---------------------------------------------------------------------------------------#

processorcount(NUM_PROCS_REAL)
math(EXPR _NUM_THREADS "${NUM_PROCS_REAL} - (${NUM_PROCS_REAL} / 2)")
if(_NUM_THREADS GREATER 8)
    set(_NUM_THREADS 8)
elseif(_NUM_THREADS LESS 1)
    set(_NUM_THREADS 1)
endif()

set(OMNITRACE_PERFETTO_SOURCE_DIR ${PROJECT_BINARY_DIR}/external/perfetto/source)
set(OMNITRACE_PERFETTO_TOOLS_DIR ${PROJECT_BINARY_DIR}/external/perfetto/source/tools)
set(OMNITRACE_PERFETTO_BINARY_DIR
    ${PROJECT_BINARY_DIR}/external/perfetto/source/out/linux)
set(OMNITRACE_PERFETTO_INSTALL_DIR
    ${PROJECT_BINARY_DIR}/external/perfetto/source/out/linux/stripped)
set(OMNITRACE_PERFETTO_LINK_FLAGS
    "-static-libgcc -static-libstdc++"
    CACHE STRING "Link flags for perfetto")
set(OMNITRACE_PERFETTO_BUILD_THREADS
    ${_NUM_THREADS}
    CACHE STRING "Number of threads to use when building perfetto tools")

if(CMAKE_CXX_COMPILER_IS_CLANG)
    set(PERFETTO_IS_CLANG true)
    set(OMNITRACE_PERFETTO_C_FLAGS
        ""
        CACHE STRING "Perfetto C flags")
    set(OMNITRACE_PERFETTO_CXX_FLAGS
        ""
        CACHE STRING "Perfetto C++ flags")
else()
    set(PERFETTO_IS_CLANG false)
    set(OMNITRACE_PERFETTO_C_FLAGS
        "-static-libgcc -static-libstdc++ -Wno-maybe-uninitialized"
        CACHE STRING "Perfetto C flags")
    set(OMNITRACE_PERFETTO_CXX_FLAGS
        "-static-libgcc -static-libstdc++ -Wno-maybe-uninitialized"
        CACHE STRING "Perfetto C++ flags")
endif()

mark_as_advanced(OMNITRACE_PERFETTO_C_FLAGS)
mark_as_advanced(OMNITRACE_PERFETTO_CXX_FLAGS)
mark_as_advanced(OMNITRACE_PERFETTO_LINK_FLAGS)

if(NOT OMNITRACE_NINJA_EXECUTABLE)
    set(OMNITRACE_NINJA_EXECUTABLE
        ${OMNITRACE_PERFETTO_TOOLS_DIR}/ninja
        CACHE FILEPATH "Ninja" FORCE)
endif()

# ---------------------------------------------------------------------------------------#
#
# source tree
#
# ---------------------------------------------------------------------------------------#

if(NOT EXISTS "${OMNITRACE_PERFETTO_SOURCE_DIR}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
                            ${PROJECT_BINARY_DIR}/external/perfetto)
    # cmake -E copy_directory fails for some reason
    execute_process(
        COMMAND ${OMNITRACE_COPY_EXECUTABLE} -r ${PROJECT_SOURCE_DIR}/external/perfetto/
                ${OMNITRACE_PERFETTO_SOURCE_DIR})
endif()

configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/args.gn.in
               ${OMNITRACE_PERFETTO_BINARY_DIR}/args.gn @ONLY)

# ---------------------------------------------------------------------------------------#
#
# build tools
#
# ---------------------------------------------------------------------------------------#

if(OMNITRACE_INSTALL_PERFETTO_TOOLS)
    find_program(
        OMNITRACE_CURL_EXECUTABLE
        NAMES curl
        PATH_SUFFIXES bin)

    if(NOT OMNITRACE_CURL_EXECUTABLE)
        omnitrace_message(
            SEND_ERROR
            "curl executable cannot be found. install-build-deps script for perfetto will fail"
            )
    endif()

    externalproject_add(
        omnitrace-perfetto-build
        PREFIX ${PROJECT_BINARY_DIR}/external/perfetto
        SOURCE_DIR ${OMNITRACE_PERFETTO_SOURCE_DIR}
        BUILD_IN_SOURCE 1
        PATCH_COMMAND ${OMNITRACE_PERFETTO_TOOLS_DIR}/install-build-deps
        CONFIGURE_COMMAND ${OMNITRACE_PERFETTO_TOOLS_DIR}/gn gen
                          ${OMNITRACE_PERFETTO_BINARY_DIR}
        BUILD_COMMAND ${OMNITRACE_NINJA_EXECUTABLE} -C ${OMNITRACE_PERFETTO_BINARY_DIR} -j
                      ${OMNITRACE_PERFETTO_BUILD_THREADS}
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ${OMNITRACE_PERFETTO_BINARY_DIR}/args.gn)

    add_custom_target(
        omnitrace-perfetto-clean
        COMMAND ${OMNITRACE_NINJA_EXECUTABLE} -t clean
        COMMAND ${CMAKE_COMMAND} -E rm -rf
                ${PROJECT_BINARY_DIR}/external/perfetto/src/omnitrace-perfetto-build-stamp
        WORKING_DIRECTORY ${OMNITRACE_PERFETTO_BINARY_DIR}
        COMMENT "Cleaning Perfetto...")

    install(
        DIRECTORY ${OMNITRACE_PERFETTO_INSTALL_DIR}/
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/omnitrace
        COMPONENT perfetto
        FILES_MATCHING
        PATTERN "*libperfetto.so*")

    foreach(_FILE perfetto traced tracebox traced_probes traced_perf trigger_perfetto)
        if("${_FILE}" STREQUAL "perfetto")
            string(REPLACE "_" "-" _INSTALL_FILE "omnitrace-${_FILE}")
        else()
            string(REPLACE "_" "-" _INSTALL_FILE "omnitrace-perfetto-${_FILE}")
        endif()
        install(
            PROGRAMS ${OMNITRACE_PERFETTO_INSTALL_DIR}/${_FILE}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT perfetto
            RENAME ${_INSTALL_FILE}
            OPTIONAL)
    endforeach()
endif()

# ---------------------------------------------------------------------------------------#
#
# perfetto static library
#
# ---------------------------------------------------------------------------------------#

add_library(omnitrace-perfetto-library STATIC)
add_library(omnitrace::omnitrace-perfetto-library ALIAS omnitrace-perfetto-library)
target_sources(
    omnitrace-perfetto-library PRIVATE ${OMNITRACE_PERFETTO_SOURCE_DIR}/sdk/perfetto.cc
                                       ${OMNITRACE_PERFETTO_SOURCE_DIR}/sdk/perfetto.h)
target_link_libraries(
    omnitrace-perfetto-library
    PRIVATE omnitrace::omnitrace-threading omnitrace::omnitrace-static-libgcc
            omnitrace::omnitrace-static-libstdcxx omnitrace::omnitrace-compile-options)
set_target_properties(
    omnitrace-perfetto-library
    PROPERTIES OUTPUT_NAME perfetto
               ARCHIVE_OUTPUT_DIRECTORY ${OMNITRACE_PERFETTO_BINARY_DIR}
               POSITION_INDEPENDENT_CODE ON
               CXX_VISIBILITY_PRESET "internal")

set(perfetto_DIR ${OMNITRACE_PERFETTO_SOURCE_DIR})
set(PERFETTO_ROOT_DIR
    ${OMNITRACE_PERFETTO_SOURCE_DIR}
    CACHE PATH "Root Perfetto installation" FORCE)
set(PERFETTO_INCLUDE_DIR
    ${OMNITRACE_PERFETTO_SOURCE_DIR}/sdk
    CACHE PATH "Perfetto include folder" FORCE)
set(PERFETTO_LIBRARY
    ${OMNITRACE_PERFETTO_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}perfetto${CMAKE_STATIC_LIBRARY_SUFFIX}
    CACHE FILEPATH "Perfetto library" FORCE)

mark_as_advanced(PERFETTO_ROOT_DIR)
mark_as_advanced(PERFETTO_INCLUDE_DIR)
mark_as_advanced(PERFETTO_LIBRARY)

# ---------------------------------------------------------------------------------------#
#
# perfetto interface library
#
# ---------------------------------------------------------------------------------------#

omnitrace_target_compile_definitions(omnitrace-perfetto INTERFACE OMNITRACE_USE_PERFETTO)
target_include_directories(omnitrace-perfetto SYSTEM
                           INTERFACE $<BUILD_INTERFACE:${PERFETTO_INCLUDE_DIR}>)
target_link_libraries(
    omnitrace-perfetto INTERFACE $<BUILD_INTERFACE:${PERFETTO_LIBRARY}>
                                 $<BUILD_INTERFACE:omnitrace::omnitrace-threading>)
