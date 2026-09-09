## Copyright (c) 2019-2026 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.

# Available defines to be used when building rocdbgapi:
#
# THE_ROCK_BUILD
#
# Satisfy TheRock's dependency on Threads. Default is OFF.
#
# If set to ON, include Threads when building. This is required when
# rocdbgapi gets built within TheRock build platform due to a dependency
# coming from amd_comgr.
#
# If building on its own, this does not need to be set.
#
#
# API_TRACING
#
# Enable/Disable API tracing. Default is ON.
#
# For debugging and stepping through code, disabling API_TRACING is
# recommended.
#
#
# ENABLE_ASSERTIONS
#
# Enable/Disable assertions. Default is ON for DEBUG builds and OFF for
# non-DEBUG builds.
#
#
# BUILD_SHARED_LIBS
#
# Enable building ROCdbgapi either as a shared library or static library.
# The default is ON, which builds ROCdbgapi as a shared library.
#
#
# ROCdbgapi has the following dependencies:
#
# amd_comgr
# libbacktrace (optional)
# GIT (optional)
# Doxygen (optional)
# Threads (on request, see THE_ROCK_BUILD)

cmake_minimum_required(VERSION 3.21)

project(amd-dbgapi VERSION 0.80.0 LANGUAGES CXX DESCRIPTION "ROCm Debugger API")

include(GNUInstallDirs)
include(CheckCXXSourceCompiles)

# =============================================================================
# Helper Functions
# =============================================================================

# Determine an imported target's property from the target's imported
# configuration. TARGET_NAME is the target's name. E.g., if PROPERTY_TYPE is
# IMPORTED_LOCATION, this fetches the right IMPORTED_LOCATION_<config> (e.g.,
# IMPORTED_LOCATION_RELEASE) from TARGET_NAME's properties.
function(get_imported_property TARGET_NAME PROPERTY_TYPE PROPERTY_VAR)
    if(NOT TARGET ${TARGET_NAME})
        message(FATAL_ERROR "Target ${TARGET_NAME} not found.")
    endif()

    # Retrieve the list of all configurations provided by the imported target.
    get_target_property(
        AVAILABLE_CONFIGS
        ${TARGET_NAME}
        IMPORTED_CONFIGURATIONS
    )

    set(SELECTED_CONFIG "")

    # If CMAKE_BUILD_TYPE is set, try an exact match. This is what cmake does too:
    # https://gitlab.kitware.com/cmake/cmake/blob/v3.29.6/Source/cmTarget.cxx#L3378-3387
    if(CMAKE_BUILD_TYPE)
        string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
        # Check if the specified build type matches any of the available configurations.
        if("${CMAKE_BUILD_TYPE_UPPER}" IN_LIST AVAILABLE_CONFIGS)
            set(SELECTED_CONFIG "${CMAKE_BUILD_TYPE_UPPER}")
            get_target_property(
                PROPERTY_VALUE
                ${TARGET_NAME}
                ${PROPERTY_TYPE}_${CMAKE_BUILD_TYPE_UPPER}
            )
            set(${PROPERTY_VAR} ${PROPERTY_VALUE} PARENT_SCOPE)
            message(
                STATUS
                "Selected configuration for ${TARGET_NAME}: ${SELECTED_CONFIG} (matched CMAKE_BUILD_TYPE)"
            )
            return()
        endif()
    endif()

    # If the build type wasn't specified or didn't match any available
    # configurations, pick the first valid one from all available configurations.
    # This matches what cmake does too:
    # https://gitlab.kitware.com/cmake/cmake/blob/v3.29.6/Source/cmTarget.cxx#L3389-3420
    if(SELECTED_CONFIG STREQUAL "")
        foreach(CONFIG ${AVAILABLE_CONFIGS})
            string(TOUPPER "${CONFIG}" CONFIG_UPPER)
            get_target_property(
                PROPERTY_VALUE
                ${TARGET_NAME}
                ${PROPERTY_TYPE}_${CONFIG_UPPER}
            )
            if(
                NOT ${PROPERTY_VALUE}
                    STREQUAL
                    "${PROPERTY_TYPE}_${CONFIG_UPPER}-NOTFOUND"
            )
                set(SELECTED_CONFIG ${CONFIG_UPPER})
                message(
                    STATUS
                    "Selected configuration for ${TARGET_NAME}: ${SELECTED_CONFIG}"
                )
                set(${PROPERTY_VAR} ${PROPERTY_VALUE} PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endif()

    if(SELECTED_CONFIG STREQUAL "")
        message(
            FATAL_ERROR
            "Could not determine the appropriate configuration for ${TARGET_NAME} library."
        )
    endif()
endfunction()

# Generate build info string for version identification
# Returns: developer-build-<user>-git-<hash> or just developer-build
function(generate_build_info OUTPUT_VAR)
    set(build_info developer-build)

    if(DEFINED ENV{USER})
        set(build_info ${build_info}-$ENV{USER})
    endif()

    find_package(Git)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            OUTPUT_VARIABLE build_revision
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    else()
        message(STATUS "GIT not found")
    endif()

    if(DEFINED build_revision)
        set(build_info ${build_info}-git-${build_revision})
    endif()

    set(${OUTPUT_VAR} ${build_info} PARENT_SCOPE)
endfunction()

# Configure libbacktrace support for the target
# Tries find_package first (for TheRock builds), then compiler fallback
function(configure_libbacktrace TARGET_NAME)
    # First try find_package, as this is the preferred method when rocdbgapi is
    # built within TheRock build platform.
    find_package(libbacktrace CONFIG)

    if(libbacktrace_FOUND)
        # Fetch the include directory for libbacktrace.
        get_property(
            LIBBACKTRACE_INCLUDE_DIR
            TARGET libbacktrace::libbacktrace
            PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        )

        # Enable libbacktrace in the sources.
        target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_BACKTRACE)

        # Set up including the header in the compilation and linking against libbacktrace.
        target_include_directories(
            ${TARGET_NAME}
            PRIVATE ${LIBBACKTRACE_INCLUDE_DIR}
        )
        target_link_libraries(${TARGET_NAME} PRIVATE libbacktrace::libbacktrace)
        return()
    endif()

    # If find_package didn't work, try compiler fallback (not on MSVC)
    message(
        STATUS
        "[libbacktrace check] Could not find libbacktrace through find_package."
    )
    if(NOT MSVC)
        message(STATUS "[libbacktrace check] Trying compiler fallback.")

        message(
            STATUS
            "[libbacktrace check] Executing command: ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} --print-file-name libbacktrace.a"
        )

        # See if the compiler knows where libbacktrace.a is. It is worth mentioning
        # that both gcc and llvm have the same behavior when --print-file-name is
        # used but the compiler cannot locate the file anywhere. They output the
        # filename passed as argument to --print-file-name unchanged.
        #
        # This makes it so the libbacktrace_check below will always pass. Though
        # incorrect, this is somewhat harmless as we will call find_path and
        # find_library later and we could end up finding includes and library
        # depending on the system environment variables.
        execute_process(
            COMMAND
                ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS} --print-file-name
                libbacktrace.a
            RESULT_VARIABLE libbacktrace_check
            OUTPUT_VARIABLE libbacktrace_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        message(
            STATUS
            "[libbacktrace check] Command output: ${libbacktrace_path}"
        )
    else()
        set(libbacktrace_check 1)
    endif()

    if(${libbacktrace_check} EQUAL 0)
        cmake_path(GET libbacktrace_path PARENT_PATH BACKTRACE_PREFIX)
        message(
            STATUS
            "[libbacktrace check] libbacktrace.a found at ${libbacktrace_path}"
        )

        find_path(
            BACKTRACE_INCLUDE_DIR
            backtrace-supported.h
            PATHS ${BACKTRACE_PREFIX}/include
        )
        message(
            STATUS
            "[libbacktrace check] Looking for backtrace-supported.h in ${BACKTRACE_INCLUDE_DIR}"
        )

        if(BACKTRACE_INCLUDE_DIR)
            find_library(BACKTRACE_LIB "backtrace" PATHS ${BACKTRACE_PREFIX})
        endif()

        if(BACKTRACE_LIB)
            message(
                STATUS
                "[libbacktrace check] Library found at ${BACKTRACE_LIB}"
            )

            # Enable libbacktrace in the sources.
            target_compile_definitions(${TARGET_NAME} PRIVATE HAVE_BACKTRACE)
            # Set up including the header in the compilation and linking against libbacktrace.
            target_include_directories(
                ${TARGET_NAME}
                PRIVATE ${BACKTRACE_INCLUDE_DIR}
            )
            target_link_libraries(${TARGET_NAME} PRIVATE ${BACKTRACE_LIB})
        endif()
    else()
        message(
            STATUS
            "[libbacktrace check] libbacktrace.a not found. Skipping."
        )
    endif()
endfunction()

# =============================================================================
# End Helper Functions
# =============================================================================

# Convert the project's name to uppercase and replace '-' with '_'.
# This transforms "amd-dbgapi" to "AMD_DBGAPI" for use in:
# - Symbol versioning (exportmap.in): AMD_DBGAPI_0.54, AMD_DBGAPI_0.56, etc.
# - API documentation (amd-dbgapi.h.in): symbol version string references
# - Doxygen configuration (docs/doxygen/cmake_setting/Doxyfile.in): project name
string(TOUPPER "${PROJECT_NAME}" AMD_DBGAPI_NAME)
string(REPLACE "-" "_" AMD_DBGAPI_NAME ${AMD_DBGAPI_NAME})

# Set default for ENABLE_ASSERTIONS based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ENABLE_ASSERTIONS_DEFAULT ON)
else()
    set(ENABLE_ASSERTIONS_DEFAULT OFF)
endif()

# Configuration options
option(THE_ROCK_BUILD "Satisfy TheRock's dependency on Threads" OFF)
option(API_TRACING "Enable API tracing" ON)
option(ENABLE_ASSERTIONS "Enable assertions" ${ENABLE_ASSERTIONS_DEFAULT})
# Allow building dbgapi as either a shared or static library.
# Note add_library internally checks BUILD_SHARED_LIBS, but CMake does
# not define BUILD_SHARED_LIBS by default.  See
# <https://cmake.org/cmake/help/latest/variable/BUILD_SHARED_LIBS.html>.
option(BUILD_SHARED_LIBS "Build amd-dbgapi as a shared library" ON)

if(ENABLE_ASSERTIONS)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        # Add -UNDEBUG to the CXXFLAGS (project is C++ only, no CFLAGS needed)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>)
        # Remove -DNDEBUG from the CXXFLAGS
        foreach(
            flags
            IN
            ITEMS
                CMAKE_CXX_FLAGS_RELEASE
                CMAKE_CXX_FLAGS_RELWITHDEBINFO
                CMAKE_CXX_FLAGS_MINSIZEREL
        )
            string(
                REGEX REPLACE
                "(^| )-D *NDEBUG($| )"
                " "
                "${flags}"
                "${${flags}}"
            )
        endforeach()
    endif()
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_definitions(NDEBUG)
    endif()
endif()

# Define this before amd-dbgapi.h is generated. amd-dbgapi.h.in uses
# this to define the right export/import decorators in amd-dbgapi.h
# for clients to use.
set(AMD_DBGAPI_STATIC_LIB "0")
if(NOT BUILD_SHARED_LIBS)
    set(AMD_DBGAPI_STATIC_LIB "1")
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/amd-dbgapi.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/amd-dbgapi/amd-dbgapi.h
    @ONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/exportmap.in
    ${CMAKE_CURRENT_BINARY_DIR}/src/exportmap
    @ONLY
)

add_library(
    amd-dbgapi
    src/agent.cpp
    src/architecture.cpp
    src/callbacks.cpp
    src/code_object.cpp
    src/debug.cpp
    src/dispatch.cpp
    src/displaced_stepping.cpp
    src/event.cpp
    src/exception.cpp
    src/initialization.cpp
    src/logging.cpp
    src/memory.cpp
    src/os_driver.cpp
    src/process.cpp
    src/queue.cpp
    src/register.cpp
    src/utils.cpp
    src/versioning.cpp
    src/watchpoint.cpp
    src/wave.cpp
    src/workgroup.cpp
)

set_target_properties(
    amd-dbgapi
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS ON
        CXX_VISIBILITY_PRESET hidden
        OUTPUT_NAME "rocm-dbgapi"
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
        DEFINE_SYMBOL "AMD_DBGAPI_EXPORTS"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(MSVC)
    # CMake adds /GR to CMAKE_CXX_FLAGS, remove it to avoid:
    #  cl : Command line warning D9025 : overriding '/GR' with '/GR-'
    string(REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    target_compile_options(amd-dbgapi PRIVATE /GR- /permissive-)
else()
    target_compile_options(
        amd-dbgapi
        PRIVATE
            -fno-rtti
            -Werror
            -Wall
            -Wextra
            -Wshadow
            -Wno-attributes
            -Wconversion
    ) #-pedantic)
endif()

target_compile_definitions(
    amd-dbgapi
    PRIVATE __STDC_LIMIT_MACROS __STDC_CONSTANT_MACROS __STDC_FORMAT_MACROS
)

if(MINGW)
    target_link_options(amd-dbgapi PRIVATE -static-libgcc -static-libstdc++)
endif()

# Configure libbacktrace support (optional dependency)
configure_libbacktrace(amd-dbgapi)

if(API_TRACING)
    target_compile_definitions(amd-dbgapi PRIVATE WITH_API_TRACING)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_sources(amd-dbgapi PRIVATE src/os_driver_kfd.cpp src/utils_linux.cpp)

    set(CMAKE_REQUIRED_FLAGS_BKP "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++17")
    # For some platforms (PPC64), linux/types.h uses L64, but this code assumes
    # LL64 (i.e.  __u64 should be a long long unsigned, not long unsigned).
    check_cxx_source_compiles(
        "#include <linux/types.h>
     #include <type_traits>
     int main () { static_assert (std::is_same_v<long long unsigned, __u64>); return 0; }"
        LINUX_TYPES_HAS_LL64
    )

    if(NOT LINUX_TYPES_HAS_LL64)
        # Check if using __SANE_USERSPACE_TYPES__ gives us LL64
        check_cxx_source_compiles(
            "#define __SANE_USERSPACE_TYPES__
       #include <linux/types.h>
       #include <type_traits>
       int main () { static_assert (std::is_same_v<long long unsigned, __u64>); return 0; }"
            LINUX_TYPES_LL64_NEEDS_SANE_USERSPACE_TYPES
        )
        if(LINUX_TYPES_LL64_NEEDS_SANE_USERSPACE_TYPES)
            target_compile_definitions(
                amd-dbgapi
                PRIVATE "__SANE_USERSPACE_TYPES__"
            )
        else()
            message(
                FATAL_ERROR
                "A LL64 configuration is required for the Linux backend"
            )
        endif()
    endif()
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_BKP}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_sources(
        amd-dbgapi
        PRIVATE src/os_driver_kmd.cpp src/utils_windows.cpp
    )

    target_include_directories(
        amd-dbgapi
        PRIVATE src/windows/imported ${D3DKMT_INCLUDE_PATH}
    )

    # Including libdxg raises some warnings (unknown pragmas) as well as
    # "X too small to hold all values of Y".  The latter warning cannot
    # be silenced, so tell the compiler to treat the libdxg directory as
    # a system headers include path, so that warnings are disabled for
    # that directory's headers.
    target_include_directories(
        amd-dbgapi
        SYSTEM
        PRIVATE third_party/libdxg/include
    )

    target_compile_definitions(
        amd-dbgapi
        PRIVATE D3DKMDT_SPECIAL_MULTIPLATFORM_TOOL
    )
else()
    message(FATAL_ERROR "Platform ${CMAKE_SYSTEM_NAME} is not supported")
endif()

find_package(
    amd_comgr
    REQUIRED
    CONFIG
    PATHS /opt/rocm/
    PATH_SUFFIXES lib/cmake/amd_comgr
)
message(STATUS "Code Object Manager found at ${amd_comgr_DIR}.")

# Depending on how we're building rocdbgapi, we may have a dependency
# on Threads coming from amd_comgr. If we've been instructed to include
# Threads, do so now.
if(THE_ROCK_BUILD)
    # We need to pull in Threads because we link with it.
    find_package(Threads REQUIRED)
endif()

# If building a static library, retrieve the amd_comgr library's name
# from the cmake package, to put in the .pc file.
set(PC_EXTRA_LIBS "")
if(NOT BUILD_SHARED_LIBS)
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        # On Windows, look for the import lib.  If amd_comgr was built
        # with MSVC, then the import lib will look like e.g.,
        # amd_comgr_2.lib.  I.e., we can't hardcode whether we have
        # lib${name}.a or ${name}.lib, nor can we assume that ${name} is
        # exactly "amd_comgr".
        get_imported_property(amd_comgr IMPORTED_IMPLIB AMD_COMGR_IMPLIB)
        cmake_path(GET AMD_COMGR_IMPLIB FILENAME AMD_COMGR_IMPLIB_BASENAME)
        cmake_path(GET AMD_COMGR_IMPLIB PARENT_PATH AMD_COMGR_IMPLIB_DIR)
        set(PC_EXTRA_LIBS
            "-L${AMD_COMGR_IMPLIB_DIR} -l:${AMD_COMGR_IMPLIB_BASENAME}"
        )
    else()
        get_imported_property(amd_comgr IMPORTED_LOCATION AMD_COMGR_LOCATION)
        cmake_path(GET AMD_COMGR_LOCATION PARENT_PATH AMD_COMGR_LOCATION_DIR)
        set(PC_EXTRA_LIBS "-L${AMD_COMGR_LOCATION_DIR} -lamd_comgr")
    endif()
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/amd-dbgapi.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/share/pkgconfig/amd-dbgapi.pc
    @ONLY
)

generate_build_info(build_info)

target_link_libraries(amd-dbgapi PRIVATE amd_comgr ${CMAKE_DL_LIBS})

set_source_files_properties(
    src/versioning.cpp
    src/initialization.cpp
    PROPERTIES
        COMPILE_DEFINITIONS
            "AMD_DBGAPI_VERSION_PATCH=${PROJECT_VERSION_PATCH};AMD_DBGAPI_BUILD_INFO=\"${PROJECT_VERSION}-${build_info}\""
)

target_include_directories(
    amd-dbgapi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/amd-dbgapi>
        $<INSTALL_INTERFACE:include>
)

if(NOT WIN32)
    target_link_options(
        amd-dbgapi
        PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/src/exportmap"
        PRIVATE "LINKER:--no-undefined"
    )
endif()

set(AMD_DBGAPI_CONFIG_NAME amd-dbgapi-config.cmake)
set(AMD_DBGAPI_TARGETS_NAME amd-dbgapi-targets.cmake)
set(AMD_DBGAPI_PACKAGE_PREFIX ${CMAKE_INSTALL_LIBDIR}/cmake/amd-dbgapi)

# Generate the build-tree package.
set(AMD_DBGAPI_PREFIX_CODE)
set(AMD_DBGAPI_TARGETS_PATH
    "${CMAKE_CURRENT_BINARY_DIR}/${AMD_DBGAPI_PACKAGE_PREFIX}/${AMD_DBGAPI_TARGETS_NAME}"
)
export(
    TARGETS amd-dbgapi
    FILE "${AMD_DBGAPI_PACKAGE_PREFIX}/${AMD_DBGAPI_TARGETS_NAME}"
)
configure_file(
    "cmake/${AMD_DBGAPI_CONFIG_NAME}.in"
    "${AMD_DBGAPI_PACKAGE_PREFIX}/${AMD_DBGAPI_CONFIG_NAME}"
    @ONLY
)

install(
    TARGETS amd-dbgapi
    EXPORT amd-dbgapi-export
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT dev
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/amd-dbgapi/amd-dbgapi.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd-dbgapi
    COMPONENT dev
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/share/pkgconfig/amd-dbgapi.pc"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/pkgconfig
    COMPONENT dev
)

install(
    FILES "LICENSE.txt"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/rocm-dbgapi
    COMPONENT dev
)

# Generate the install-tree package.
set(AMD_DBGAPI_PREFIX_CODE
    "
# Derive absolute install prefix from config file path.
cmake_path(GET CMAKE_CURRENT_LIST_FILE PARENT_PATH AMD_DBGAPI_PREFIX)"
)
# Convert path to list to count components (e.g., "lib/cmake/amd-dbgapi" -> 3 components)
# We need to traverse up the directory tree once for each path component
string(REGEX REPLACE "/" ";" path_components "${AMD_DBGAPI_PACKAGE_PREFIX}")
foreach(p IN LISTS path_components)
    set(AMD_DBGAPI_PREFIX_CODE
        "${AMD_DBGAPI_PREFIX_CODE}
cmake_path(GET AMD_DBGAPI_PREFIX PARENT_PATH AMD_DBGAPI_PREFIX)"
    )
endforeach()
set(AMD_DBGAPI_TARGETS_PATH
    "\${AMD_DBGAPI_PREFIX}/${AMD_DBGAPI_PACKAGE_PREFIX}/${AMD_DBGAPI_TARGETS_NAME}"
)
configure_file(
    "cmake/${AMD_DBGAPI_CONFIG_NAME}.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${AMD_DBGAPI_CONFIG_NAME}.install"
    @ONLY
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${AMD_DBGAPI_CONFIG_NAME}.install"
    DESTINATION "${AMD_DBGAPI_PACKAGE_PREFIX}"
    RENAME "${AMD_DBGAPI_CONFIG_NAME}"
    COMPONENT dev
)
install(
    EXPORT amd-dbgapi-export
    DESTINATION "${AMD_DBGAPI_PACKAGE_PREFIX}"
    FILE "${AMD_DBGAPI_TARGETS_NAME}"
    COMPONENT dev
)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/cmake_setting/Doxyfile.in
    )
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/doc/html/index.html
            ${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}/doc/latex pdf
        MAIN_DEPENDENCY ${DOXYGEN_OUT}
        ${DOXYGEN_IN}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/include/amd-dbgapi.h.in
            ${CMAKE_CURRENT_BINARY_DIR}/include/amd-dbgapi/amd-dbgapi.h
        COMMENT "Generating documentation"
    )

    add_custom_target(
        doc
        DEPENDS
            ${CMAKE_CURRENT_BINARY_DIR}/doc/html/index.html
            ${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf
    )

    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/rocm-dbgapi
        RENAME "amd-dbgapi.pdf"
        OPTIONAL
        COMPONENT dev
    )

    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/doc/html/"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/html/amd-dbgapi
        OPTIONAL
        COMPONENT dev
    )
endif()
