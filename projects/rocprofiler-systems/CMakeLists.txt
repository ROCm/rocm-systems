cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND CMAKE_CURRENT_SOURCE_DIR STREQUAL
                                                  CMAKE_SOURCE_DIR)
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(AUTHOR_WARNING "In-source build")
endif()

if(NOT UNIX OR APPLE)
    message(
        AUTHOR_WARNING
            "omnitrace only supports Linux. Configure and/or build is likely to fail")
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
string(REGEX REPLACE "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)" "\\1.\\2.\\3" OMNITRACE_VERSION
                     "${FULL_VERSION_STRING}")

project(
    omnitrace
    LANGUAGES C CXX
    VERSION ${OMNITRACE_VERSION}
    DESCRIPTION "CPU/GPU Application tracing with static/dynamic binary instrumentation"
    HOMEPAGE_URL "https://github.com/AMDResearch/omnitrace")

message(
    STATUS
        "[${PROJECT_NAME}] version ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    )
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/cmake/Modules
                      ${CMAKE_MODULE_PATH})
set(BUILD_SHARED_LIBS
    ON
    CACHE BOOL "Build shared libraries")
set(BUILD_STATIC_LIBS
    OFF
    CACHE BOOL "Build static libraries")
set(CMAKE_POSITION_INDEPENDENT_CODE
    ON
    CACHE BOOL "Build position independent code")

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE
        RelWithDebInfo
        CACHE STRING "Build type" FORCE)
endif()

include(GNUInstallDirs) # install directories
include(MacroUtilities) # various functions and macros
include(Compilers) # compiler identification
include(BuildSettings) # compiler flags

# force this because dyninst always installs to lib
set(CMAKE_INSTALL_LIBDIR
    "lib"
    CACHE STRING "Object code libraries (lib)" FORCE)
set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "CXX language standard")
omnitrace_add_feature(CMAKE_BUILD_TYPE "Build optimization level")
omnitrace_add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
omnitrace_add_feature(CMAKE_CXX_COMPILER "C++ compiler")
omnitrace_add_feature(CMAKE_CXX_STANDARD "CXX language standard")
omnitrace_add_option(CMAKE_CXX_STANDARD_REQUIRED "Require C++ language standard" ON)
omnitrace_add_option(CMAKE_CXX_EXTENSIONS "Compiler specific language extensions" OFF)
omnitrace_add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Enable rpath to linked libraries"
                     ON)

omnitrace_add_option(OMNITRACE_USE_CLANG_TIDY "Enable clang-tidy" OFF)
omnitrace_add_option(OMNITRACE_USE_MPI "Enable MPI support" OFF)
omnitrace_add_option(OMNITRACE_USE_HIP "Enable HIP support" ON)
omnitrace_add_option(OMNITRACE_USE_PAPI "Enable HW counter support via PAPI" OFF)
omnitrace_add_option(OMNITRACE_USE_ROCTRACER "Enable roctracer support"
                     ${OMNITRACE_USE_HIP})
omnitrace_add_option(
    OMNITRACE_USE_ROCM_SMI "Enable rocm-smi support for power/temp/etc. sampling"
    ${OMNITRACE_USE_HIP})
omnitrace_add_option(OMNITRACE_USE_MPI_HEADERS
                     "Enable wrapping MPI functions w/o enabling MPI dependency" OFF)
omnitrace_add_option(OMNITRACE_USE_OMPT "Enable OpenMP tools support" OFF)
omnitrace_add_option(OMNITRACE_BUILD_DYNINST "Build dyninst from submodule" OFF)
omnitrace_add_option(OMNITRACE_BUILD_EXAMPLES "Enable building the examples" OFF)
omnitrace_add_option(OMNITRACE_BUILD_TESTING "Enable building the testing suite" OFF)
omnitrace_add_option(OMNITRACE_CUSTOM_DATA_SOURCE "Enable custom data source" OFF)
omnitrace_add_option(OMNITRACE_BUILD_HIDDEN_VISIBILITY
                     "Build with hidden visibility (disable for Debug builds)" ON)
omnitrace_add_option(OMNITRACE_BUILD_CI "Enable internal asserts, etc." OFF ADVANCED
                     NO_FEATURE)

if(NOT "$ENV{OMNITRACE_CI}" STREQUAL "")
    message(
        AUTHOR_WARNING
            "OMNITRACE_CI environment variable ($ENV{OMNITRACE_CI}) is overridding the OMNITRACE_BUILD_CI cache value"
        )
    set(OMNITRACE_BUILD_CI
        "$ENV{OMNITRACE_CI}"
        CACHE BOOL "Enable internal asserts, etc" FORCE)
endif()

if(NOT OMNITRACE_USE_HIP)
    set(OMNITRACE_USE_ROCTRACER
        OFF
        CACHE BOOL "Disabled via OMNITRACE_USE_HIP=OFF" FORCE)
    set(OMNITRACE_USE_ROCM_SMI
        OFF
        CACHE BOOL "Disabled via OMNITRACE_USE_HIP=OFF" FORCE)
endif()

if(OMNITRACE_BUILD_TESTING)
    set(OMNITRACE_BUILD_EXAMPLES
        ON
        CACHE BOOL "Enable building the examples" FORCE)
endif()

include(ProcessorCount)
processorcount(OMNITRACE_PROCESSOR_COUNT)
math(EXPR OMNITRACE_THREAD_COUNT "16 * ${OMNITRACE_PROCESSOR_COUNT}")
if(OMNITRACE_THREAD_COUNT LESS 128)
    set(OMNITRACE_THREAD_COUNT 128)
endif()
set(OMNITRACE_MAX_THREADS
    "${OMNITRACE_THREAD_COUNT}"
    CACHE
        STRING
        "Maximum number of threads in the host application. Likely only needs to be increased if host app does not use thread-pool but creates many threads"
    )
omnitrace_add_feature(
    OMNITRACE_MAX_THREADS
    "Maximum number of total threads supported in the host application (default: max of 128 or 16 * nproc)"
    )

# default visibility settings
set(CMAKE_C_VISIBILITY_PRESET
    "default"
    CACHE STRING "Visibility preset for non-inline C functions")
set(CMAKE_CXX_VISIBILITY_PRESET
    "default"
    CACHE STRING "Visibility preset for non-inline C++ functions/objects")
set(CMAKE_VISIBILITY_INLINES_HIDDEN
    OFF
    CACHE BOOL "Visibility preset for inline functions")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(Formatting) # format target
include(Packages) # finds third-party libraries

omnitrace_activate_clang_tidy()

# custom visibility settings
if(OMNITRACE_BUILD_HIDDEN_VISIBILITY)
    set(CMAKE_C_VISIBILITY_PRESET "internal")
    set(CMAKE_CXX_VISIBILITY_PRESET "internal")
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
endif()

if(OMNITRACE_BUILD_TESTING OR "$ENV{OMNITRACE_CI}" MATCHES "[1-9]+|ON|on|y|yes")
    include(CTest)
    enable_testing()
endif()

# ------------------------------------------------------------------------------#
#
# library and executables
#
# ------------------------------------------------------------------------------#

add_subdirectory(source)

# ------------------------------------------------------------------------------#
#
# miscellaneous installs
#
# ------------------------------------------------------------------------------#

configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/setup-env.sh.in
               ${PROJECT_BINARY_DIR}/install-tree/setup-env.sh @ONLY)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/modulefile.in
    ${PROJECT_BINARY_DIR}/install-tree/modulefiles/${PROJECT_NAME}/${OMNITRACE_VERSION}
    @ONLY)

install(
    PROGRAMS ${PROJECT_SOURCE_DIR}/scripts/omnitrace-merge.jl
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)

install(
    FILES ${PROJECT_SOURCE_DIR}/roctrace.cfg
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}
    OPTIONAL)

install(
    FILES ${PROJECT_BINARY_DIR}/install-tree/setup-env.sh
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}
    OPTIONAL)

install(
    FILES
        ${PROJECT_BINARY_DIR}/install-tree/modulefiles/${PROJECT_NAME}/${OMNITRACE_VERSION}
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}
    OPTIONAL)

# ------------------------------------------------------------------------------#
#
# examples
#
# ------------------------------------------------------------------------------#

if(OMNITRACE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ------------------------------------------------------------------------------#
#
# tests
#
# ------------------------------------------------------------------------------#

if(OMNITRACE_BUILD_TESTING)
    add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------#
#
# packaging
#
# ------------------------------------------------------------------------------#

include(ConfigCPack)

# ------------------------------------------------------------------------------#
#
# config info
#
# ------------------------------------------------------------------------------#

omnitrace_print_features()
