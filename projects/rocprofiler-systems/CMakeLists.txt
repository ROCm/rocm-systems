cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

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

find_package(Git)

if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags
        OUTPUT_VARIABLE OMNITRACE_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _GIT_DESCRIBE_RESULT
        ERROR_QUIET)
    if(NOT _GIT_DESCRIBE_RESULT EQUAL 0)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe
            OUTPUT_VARIABLE OMNITRACE_GIT_DESCRIBE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _GIT_DESCRIBE_RESULT
            ERROR_QUIET)
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        OUTPUT_VARIABLE OMNITRACE_GIT_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
else()
    set(OMNITRACE_GIT_DESCRIBE "v${OMNITRACE_VERSION}")
    set(OMNITRACE_GIT_REVISION "")
endif()

message(
    STATUS
        "[${PROJECT_NAME}] version ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH} (${FULL_VERSION_STRING})"
    )
message(STATUS "[${PROJECT_NAME}] git revision: ${OMNITRACE_GIT_REVISION}")
message(STATUS "[${PROJECT_NAME}] git describe: ${OMNITRACE_GIT_DESCRIBE}")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${PROJECT_SOURCE_DIR}/cmake/Modules
                      ${PROJECT_SOURCE_DIR}/source/python/cmake ${CMAKE_MODULE_PATH})
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
        Release
        CACHE STRING "Build type" FORCE)
else()
    set(VALID_BUILD_TYPES "Release" "RelWithDebInfo" "Debug" "MinSizeRel")
    if(NOT "${CMAKE_BUILD_TYPE}" IN_LIST VALID_BUILD_TYPES)
        string(REPLACE ";" ", " _VALID_BUILD_TYPES "${VALID_BUILD_TYPES}")
        message(
            FATAL_ERROR
                "Invalid CMAKE_BUILD_TYPE :: ${CMAKE_BUILD_TYPE}. Valid build types are: ${_VALID_BUILD_TYPES}"
            )
    endif()
endif()
set(_STRIP_LIBRARIES_DEFAULT OFF)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(_STRIP_LIBRARIES_DEFAULT ON)
endif()

if(DEFINED CMAKE_INSTALL_LIBDIR AND NOT DEFINED CMAKE_DEFAULT_INSTALL_LIBDIR)
    # always have a fresh install
    unset(CMAKE_INSTALL_LIBDIR CACHE)
    include(GNUInstallDirs) # install directories
    # force this because dyninst always installs to lib
    set(CMAKE_DEFAULT_INSTALL_LIBDIR
        "${CMAKE_INSTALL_LIBDIR}"
        CACHE STRING "Object code libraries" FORCE)
endif()

if(NOT "$ENV{OMNITRACE_CI}" STREQUAL "")
    set(CI_BUILD $ENV{OMNITRACE_CI})
else()
    set(CI_BUILD OFF)
endif()

include(GNUInstallDirs) # install directories
include(MacroUtilities) # various functions and macros

if(CI_BUILD)
    omnitrace_add_option(OMNITRACE_BUILD_CI "Enable internal asserts, etc." ON ADVANCED
                         NO_FEATURE)
    omnitrace_add_option(OMNITRACE_BUILD_TESTING "Enable building the testing suite" ON
                         ADVANCED)
    omnitrace_add_option(OMNITRACE_BUILD_DEBUG
                         "Enable building with extensive debug symbols" OFF ADVANCED)
    omnitrace_add_option(
        OMNITRACE_BUILD_HIDDEN_VISIBILITY
        "Build with hidden visibility (disable for Debug builds)" OFF ADVANCED)
    omnitrace_add_option(OMNITRACE_STRIP_LIBRARIES "Strip the libraries" OFF ADVANCED)
else()
    omnitrace_add_option(OMNITRACE_BUILD_CI "Enable internal asserts, etc." OFF ADVANCED
                         NO_FEATURE)
    omnitrace_add_option(OMNITRACE_BUILD_EXAMPLES "Enable building the examples" OFF
                         ADVANCED)
    omnitrace_add_option(OMNITRACE_BUILD_TESTING "Enable building the testing suite" OFF
                         ADVANCED)
    omnitrace_add_option(OMNITRACE_BUILD_DEBUG
                         "Enable building with extensive debug symbols" OFF ADVANCED)
    omnitrace_add_option(
        OMNITRACE_BUILD_HIDDEN_VISIBILITY
        "Build with hidden visibility (disable for Debug builds)" ON ADVANCED)
    omnitrace_add_option(OMNITRACE_STRIP_LIBRARIES "Strip the libraries"
                         ${_STRIP_LIBRARIES_DEFAULT} ADVANCED)
endif()

include(Compilers) # compiler identification
include(BuildSettings) # compiler flags

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
set(CMAKE_INSTALL_MESSAGE
    "LAZY"
    CACHE STRING "Installation message")
mark_as_advanced(CMAKE_INSTALL_MESSAGE)

omnitrace_add_option(OMNITRACE_USE_CLANG_TIDY "Enable clang-tidy" OFF)
omnitrace_add_option(OMNITRACE_USE_BFD
                     "Enable BFD support (map call-stack samples to LOC)" ON)
omnitrace_add_option(OMNITRACE_USE_MPI "Enable MPI support" OFF)
omnitrace_add_option(OMNITRACE_USE_HIP "Enable HIP support" ON)
omnitrace_add_option(OMNITRACE_USE_PAPI "Enable HW counter support via PAPI" ON)
omnitrace_add_option(OMNITRACE_USE_ROCTRACER "Enable roctracer support"
                     ${OMNITRACE_USE_HIP})
omnitrace_add_option(OMNITRACE_USE_ROCPROFILER "Enable rocprofiler support"
                     ${OMNITRACE_USE_HIP})
omnitrace_add_option(
    OMNITRACE_USE_ROCM_SMI "Enable rocm-smi support for power/temp/etc. sampling"
    ${OMNITRACE_USE_HIP})
omnitrace_add_option(OMNITRACE_USE_RCCL "Enable RCCL support" ${OMNITRACE_USE_HIP})
omnitrace_add_option(OMNITRACE_USE_MPI_HEADERS
                     "Enable wrapping MPI functions w/o enabling MPI dependency" ON)
omnitrace_add_option(OMNITRACE_USE_OMPT "Enable OpenMP tools support" ON)
omnitrace_add_option(OMNITRACE_USE_PYTHON "Enable Python support" OFF)
omnitrace_add_option(OMNITRACE_BUILD_DYNINST "Build dyninst from submodule" OFF)
omnitrace_add_option(OMNITRACE_BUILD_LIBUNWIND "Build libunwind from submodule" ON)
omnitrace_add_option(OMNITRACE_BUILD_CODECOV "Build for code coverage" OFF)
omnitrace_add_option(OMNITRACE_INSTALL_PERFETTO_TOOLS
                     "Install perfetto tools (i.e. traced, perfetto, etc.)" OFF)

if(OMNITRACE_USE_PAPI)
    omnitrace_add_option(OMNITRACE_BUILD_PAPI "Build PAPI from submodule" ON)
endif()

if(OMNITRACE_USE_PYTHON)
    omnitrace_add_option(OMNITRACE_BUILD_PYTHON
                         "Build python bindings with internal pybind11" ON)
elseif("$ENV{OMNITRACE_CI}")
    # quiet warnings in dashboard
    if(OMNITRACE_PYTHON_ENVS OR OMNITRACE_PYTHON_PREFIX)
        omnitrace_message(
            STATUS
            "Ignoring values of OMNITRACE_PYTHON_ENVS and/or OMNITRACE_PYTHON_PREFIX")
    endif()
endif()

if(NOT OMNITRACE_USE_HIP)
    set(OMNITRACE_USE_ROCTRACER
        OFF
        CACHE BOOL "Disabled via OMNITRACE_USE_HIP=OFF" FORCE)
    set(OMNITRACE_USE_ROCPROFILER
        OFF
        CACHE BOOL "Disabled via OMNITRACE_USE_HIP=OFF" FORCE)
    set(OMNITRACE_USE_ROCM_SMI
        OFF
        CACHE BOOL "Disabled via OMNITRACE_USE_HIP=OFF" FORCE)
    set(OMNITRACE_USE_RCCL
        OFF
        CACHE BOOL "Disabled via OMNITRACE_USE_HIP=OFF" FORCE)
elseif(
    OMNITRACE_USE_HIP
    AND NOT OMNITRACE_USE_ROCTRACER
    AND NOT OMNITRACE_USE_ROCPROFILER
    AND NOT OMNITRACE_USE_ROCM_SMI
    AND NOT OMNITRACE_USE_RCCL)
    omnitrace_message(
        AUTHOR_WARNING
        "Setting OMNITRACE_USE_HIP=OFF because roctracer, rocprofiler, rccl, and rocm-smi options are disabled"
        )
    set(OMNITRACE_USE_HIP OFF)
endif()

if(OMNITRACE_BUILD_TESTING)
    set(OMNITRACE_BUILD_EXAMPLES
        ON
        CACHE BOOL "Enable building the examples" FORCE)
endif()

include(ProcessorCount)
processorcount(OMNITRACE_PROCESSOR_COUNT)

if(OMNITRACE_PROCESSOR_COUNT LESS 8)
    set(OMNITRACE_THREAD_COUNT 128)
else()
    math(EXPR OMNITRACE_THREAD_COUNT "16 * ${OMNITRACE_PROCESSOR_COUNT}")
    compute_pow2_ceil(OMNITRACE_THREAD_COUNT "16 * ${OMNITRACE_PROCESSOR_COUNT}")

    # set the default to 2048 if it could not be calculated
    if(OMNITRACE_THREAD_COUNT LESS 2)
        set(OMNITRACE_THREAD_COUNT 2048)
    endif()
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

compute_pow2_ceil(_MAX_THREADS "${OMNITRACE_MAX_THREADS}")

if(_MAX_THREADS GREATER 0 AND NOT OMNITRACE_MAX_THREADS EQUAL _MAX_THREADS)
    omnitrace_message(
        FATAL_ERROR
        "Error! OMNITRACE_MAX_THREADS must be a power of 2. Recommendation: ${_MAX_THREADS}"
        )
elseif(NOT OMNITRACE_MAX_THREADS EQUAL _MAX_THREADS)
    omnitrace_message(
        AUTHOR_WARNING
        "OMNITRACE_MAX_THREADS (=${OMNITRACE_MAX_THREADS}) must be a power of 2. We were unable to verify it so we are emitting this warning instead. Estimate resulted in: ${_MAX_THREADS}"
        )
endif()

set(OMNITRACE_MAX_UNWIND_DEPTH
    "64"
    CACHE
        STRING
        "Maximum call-stack depth to search during call-stack unwinding. Decreasing this value will result in sampling consuming less memory"
    )
omnitrace_add_feature(
    OMNITRACE_MAX_UNWIND_DEPTH
    "Maximum call-stack depth to search during call-stack unwinding. Decreasing this value will result in sampling consuming less memory"
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
    enable_testing()
    include(CTest)
endif()

# ------------------------------------------------------------------------------#
#
# library and executables
#
# ------------------------------------------------------------------------------#

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME core)

if(OMNITRACE_BUILD_CODECOV)
    omnitrace_save_variables(CODECOV_FLAGS VARIABLES CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    foreach(_BUILD_TYPE DEBUG MINSIZEREL RELWITHDEBINFO RELEASE)
        omnitrace_save_variables(CODECOV_FLAGS VARIABLES CMAKE_C_FLAGS_${_BUILD_TYPE}
                                                         CMAKE_CXX_FLAGS_${_BUILD_TYPE})
    endforeach()

    foreach(_BUILD_TYPE DEBUG MINSIZEREL RELWITHDEBINFO RELEASE)
        set(CMAKE_C_FLAGS_${_BUILD_TYPE}
            "-Og -g3 -fno-omit-frame-pointer -fprofile-abs-path -fprofile-arcs -ftest-coverage"
            )
        set(CMAKE_CXX_FLAGS_${_BUILD_TYPE}
            "-Og -g3 -fno-omit-frame-pointer -fprofile-abs-path -fprofile-arcs -ftest-coverage"
            )
    endforeach()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

add_subdirectory(source)

if(OMNITRACE_BUILD_CODECOV)
    omnitrace_restore_variables(CODECOV_FLAGS VARIABLES CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

    foreach(_BUILD_TYPE DEBUG MINSIZEREL RELWITHDEBINFO RELEASE)
        omnitrace_restore_variables(
            CODECOV_FLAGS VARIABLES CMAKE_C_FLAGS_${_BUILD_TYPE}
                                    CMAKE_CXX_FLAGS_${_BUILD_TYPE})
    endforeach()
endif()

# ------------------------------------------------------------------------------#
#
# miscellaneous installs
#
# ------------------------------------------------------------------------------#

if(NOT OMNITRACE_USE_ROCTRACER AND NOT OMNITRACE_USE_ROCPROFILER)
    set(OMNITRACE_HSA_ENV "# ")
endif()

if(NOT OMNITRACE_USE_ROCPROFILER)
    set(OMNITRACE_ROCP_ENV "# ")
endif()

configure_file(
    ${PROJECT_SOURCE_DIR}/omnitrace.cfg
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/omnitrace.cfg
    COPYONLY)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/setup-env.sh.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/setup-env.sh @ONLY)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/modulefile.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${OMNITRACE_VERSION}
    @ONLY)

install(
    FILES ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/setup-env.sh
          ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/omnitrace.cfg
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}
    COMPONENT setup)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${OMNITRACE_VERSION}
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}
    COMPONENT setup)

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

# install the validate-causal-json python script as a utility
configure_file(
    ${PROJECT_SOURCE_DIR}/tests/validate-causal-json.py
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/omnitrace-causal-print
    USE_SOURCE_PERMISSIONS COPYONLY)

install(PROGRAMS ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/omnitrace-causal-print
        DESTINATION ${CMAKE_INSTALL_BINDIR})

# ------------------------------------------------------------------------------#
#
# packaging
#
# ------------------------------------------------------------------------------#

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME core)
include(ConfigInstall)
include(ConfigCPack)

# ------------------------------------------------------------------------------#
#
# config info
#
# ------------------------------------------------------------------------------#

omnitrace_print_features()
