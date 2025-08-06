cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

if(
    CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR
    AND CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR
)
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(AUTHOR_WARNING "In-source build")
endif()

# find_package() uses upper-case <PACKAGENAME>_ROOT variables.
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

if(NOT UNIX OR APPLE)
    message(
        AUTHOR_WARNING
        "rocprofiler-systems only supports Linux. Configure and/or build is likely to fail"
    )
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FULL_VERSION_STRING LIMIT_COUNT 1)
string(REGEX REPLACE "(\n|\r)" "" FULL_VERSION_STRING "${FULL_VERSION_STRING}")
string(
    REGEX REPLACE
    "([0-9]+)\.([0-9]+)\.([0-9]+)(.*)"
    "\\1.\\2.\\3"
    ROCPROFSYS_VERSION
    "${FULL_VERSION_STRING}"
)

project(
    rocprofiler-systems
    LANGUAGES C CXX
    VERSION ${ROCPROFSYS_VERSION}
    DESCRIPTION "CPU/GPU Application tracing with static/dynamic binary instrumentation"
    HOMEPAGE_URL "https://github.com/ROCm/rocprofiler-systems"
)
set(PROJECT_NAME_UNDERSCORED "rocprofiler_systems")

set(BINARY_NAME_PREFIX "rocprof-sys")

set(ROCPROFSYS_CTEST_ONLY FALSE CACHE BOOL "CTest only build")

find_package(Git)

if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags
        OUTPUT_VARIABLE ROCPROFSYS_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _GIT_DESCRIBE_RESULT
        ERROR_QUIET
    )
    if(NOT _GIT_DESCRIBE_RESULT EQUAL 0)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe
            OUTPUT_VARIABLE ROCPROFSYS_GIT_DESCRIBE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _GIT_DESCRIBE_RESULT
            ERROR_QUIET
        )
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        OUTPUT_VARIABLE ROCPROFSYS_GIT_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
else()
    set(ROCPROFSYS_GIT_DESCRIBE "v${ROCPROFSYS_VERSION}")
    set(ROCPROFSYS_GIT_REVISION "")
endif()

message(
    STATUS
    "[${PROJECT_NAME}] version ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH} (${FULL_VERSION_STRING})"
)
message(STATUS "[${PROJECT_NAME}] git revision: ${ROCPROFSYS_GIT_REVISION}")
message(STATUS "[${PROJECT_NAME}] git describe: ${ROCPROFSYS_GIT_DESCRIBE}")
set(CMAKE_MODULE_PATH
    ${PROJECT_SOURCE_DIR}/cmake
    ${PROJECT_SOURCE_DIR}/cmake/Modules
    ${PROJECT_SOURCE_DIR}/source/python/cmake
    ${CMAKE_MODULE_PATH}
)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")
set(BUILD_STATIC_LIBS OFF CACHE BOOL "Build static libraries")
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build position independent code")

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    cmake_policy(SET CMP0135 NEW)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
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
        CACHE STRING
        "Object code libraries"
        FORCE
    )
endif()

if(NOT "$ENV{ROCPROFSYS_CI}" STREQUAL "")
    set(CI_BUILD $ENV{ROCPROFSYS_CI})
else()
    set(CI_BUILD OFF)
endif()

include(GNUInstallDirs) # install directories
include(MacroUtilities) # various functions and macros

if(CI_BUILD)
    rocprofiler_systems_add_option(ROCPROFSYS_BUILD_CI "Enable internal asserts, etc." ON
                                   ADVANCED NO_FEATURE
    )
    rocprofiler_systems_add_option(ROCPROFSYS_BUILD_TESTING
                                   "Enable building the testing suite" ON ADVANCED
    )
    rocprofiler_systems_add_option(
        ROCPROFSYS_BUILD_DEBUG "Enable building with extensive debug symbols" OFF
        ADVANCED
    )
    rocprofiler_systems_add_option(
        ROCPROFSYS_BUILD_HIDDEN_VISIBILITY
        "Build with hidden visibility (disable for Debug builds)" OFF ADVANCED
    )
    rocprofiler_systems_add_option(ROCPROFSYS_STRIP_LIBRARIES "Strip the libraries" OFF
                                   ADVANCED
    )
else()
    rocprofiler_systems_add_option(ROCPROFSYS_BUILD_CI "Enable internal asserts, etc."
                                   OFF ADVANCED NO_FEATURE
    )
    rocprofiler_systems_add_option(ROCPROFSYS_BUILD_EXAMPLES
                                   "Enable building the examples" OFF ADVANCED
    )
    rocprofiler_systems_add_option(ROCPROFSYS_BUILD_TESTING
                                   "Enable building the testing suite" OFF ADVANCED
    )
    rocprofiler_systems_add_option(
        ROCPROFSYS_BUILD_DEBUG "Enable building with extensive debug symbols" OFF
        ADVANCED
    )
    rocprofiler_systems_add_option(
        ROCPROFSYS_BUILD_HIDDEN_VISIBILITY
        "Build with hidden visibility (disable for Debug builds)" ON ADVANCED
    )
    rocprofiler_systems_add_option(ROCPROFSYS_STRIP_LIBRARIES "Strip the libraries"
                                   ${_STRIP_LIBRARIES_DEFAULT} ADVANCED
    )
endif()

include(Compilers) # compiler identification
include(BuildSettings) # compiler flags

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Object code libraries (lib)" FORCE)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "CXX language standard")

rocprofiler_systems_add_feature(CMAKE_BUILD_TYPE "Build optimization level")
rocprofiler_systems_add_feature(CMAKE_INSTALL_PREFIX "Installation prefix")
rocprofiler_systems_add_feature(CMAKE_CXX_COMPILER "C++ compiler")
rocprofiler_systems_add_feature(CMAKE_CXX_STANDARD "CXX language standard")
rocprofiler_systems_add_option(CMAKE_CXX_STANDARD_REQUIRED
                               "Require C++ language standard" ON
)
rocprofiler_systems_add_option(CMAKE_CXX_EXTENSIONS
                               "Compiler specific language extensions" OFF
)
rocprofiler_systems_add_option(CMAKE_INSTALL_RPATH_USE_LINK_PATH
                               "Enable rpath to linked libraries" ON
)
set(CMAKE_INSTALL_MESSAGE "LAZY" CACHE STRING "Installation message")
mark_as_advanced(CMAKE_INSTALL_MESSAGE)

rocprofiler_systems_add_option(ROCPROFSYS_USE_CLANG_TIDY "Enable clang-tidy" OFF)
rocprofiler_systems_add_option(ROCPROFSYS_USE_BFD
                               "Enable BFD support (map call-stack samples to LOC)" ON
)
rocprofiler_systems_add_option(ROCPROFSYS_USE_MPI "Enable MPI support" OFF)
rocprofiler_systems_add_option(ROCPROFSYS_USE_ROCM "Enable ROCm support" ON)
rocprofiler_systems_add_option(ROCPROFSYS_USE_PAPI "Enable HW counter support via PAPI"
                               ON
)
rocprofiler_systems_add_option(
    ROCPROFSYS_USE_MPI_HEADERS
    "Enable wrapping MPI functions w/o enabling MPI dependency" ON
)
rocprofiler_systems_add_option(ROCPROFSYS_USE_OMPT "Enable OpenMP tools support" ON)
rocprofiler_systems_add_option(ROCPROFSYS_USE_PYTHON "Enable Python support" OFF)
rocprofiler_systems_add_option(ROCPROFSYS_BUILD_DYNINST "Build dyninst from submodule"
                               OFF
)
rocprofiler_systems_add_option(ROCPROFSYS_BUILD_LIBUNWIND
                               "Build libunwind from submodule" ON
)
rocprofiler_systems_add_option(ROCPROFSYS_BUILD_CODECOV "Build for code coverage" OFF)
rocprofiler_systems_add_option(ROCPROFSYS_INSTALL_PERFETTO_TOOLS
                               "Install perfetto tools (i.e. traced, perfetto, etc.)" OFF
)
rocprofiler_systems_add_option(ROCPROFILER_BUILD_SQLITE3
                               "Enable building sqlite3 library internally" OFF
)
rocprofiler_systems_add_option(ROCPROFSYS_ENABLE_BENCHMARK
                               "Enable performance benchmarking capabilities for the project" OFF
)

if(ROCPROFSYS_USE_PAPI)
    rocprofiler_systems_add_option(ROCPROFSYS_BUILD_PAPI "Build PAPI from submodule" ON)
endif()

if(ROCPROFSYS_USE_PYTHON)
    rocprofiler_systems_add_option(ROCPROFSYS_BUILD_PYTHON
                                   "Build python bindings with internal pybind11" ON
    )
elseif("$ENV{ROCPROFSYS_CI}")
    # quiet warnings in dashboard
    if(ROCPROFSYS_PYTHON_ENVS OR ROCPROFSYS_PYTHON_PREFIX)
        rocprofiler_systems_message(
            STATUS
            "Ignoring values of ROCPROFSYS_PYTHON_ENVS and/or ROCPROFSYS_PYTHON_PREFIX"
        )
    endif()
endif()

if(ROCPROFSYS_BUILD_TESTING)
    set(ROCPROFSYS_BUILD_EXAMPLES ON CACHE BOOL "Enable building the examples" FORCE)
endif()

include(ProcessorCount)
ProcessorCount(ROCPROFSYS_PROCESSOR_COUNT)

if(ROCPROFSYS_PROCESSOR_COUNT LESS 8)
    set(ROCPROFSYS_THREAD_COUNT 128)
else()
    math(EXPR ROCPROFSYS_THREAD_COUNT "16 * ${ROCPROFSYS_PROCESSOR_COUNT}")
    compute_pow2_ceil(ROCPROFSYS_THREAD_COUNT "16 * ${ROCPROFSYS_PROCESSOR_COUNT}")

    # set the default to 2048 if it could not be calculated
    if(ROCPROFSYS_THREAD_COUNT LESS 2)
        set(ROCPROFSYS_THREAD_COUNT 2048)
    endif()
endif()

set(ROCPROFSYS_MAX_THREADS
    "${ROCPROFSYS_THREAD_COUNT}"
    CACHE STRING
    "Maximum number of threads in the host application. Likely only needs to be increased if host app does not use thread-pool but creates many threads"
)
rocprofiler_systems_add_feature(
    ROCPROFSYS_MAX_THREADS
    "Maximum number of total threads supported in the host application (default: max of 128 or 16 * nproc)"
)

compute_pow2_ceil(_MAX_THREADS "${ROCPROFSYS_MAX_THREADS}")

if(_MAX_THREADS GREATER 0 AND NOT ROCPROFSYS_MAX_THREADS EQUAL _MAX_THREADS)
    rocprofiler_systems_message(
        FATAL_ERROR
        "Error! ROCPROFSYS_MAX_THREADS must be a power of 2. Recommendation: ${_MAX_THREADS}"
    )
elseif(NOT ROCPROFSYS_MAX_THREADS EQUAL _MAX_THREADS)
    rocprofiler_systems_message(
        AUTHOR_WARNING
        "ROCPROFSYS_MAX_THREADS (=${ROCPROFSYS_MAX_THREADS}) must be a power of 2. We were unable to verify it so we are emitting this warning instead. Estimate resulted in: ${_MAX_THREADS}"
    )
endif()

set(ROCPROFSYS_MAX_UNWIND_DEPTH
    "64"
    CACHE STRING
    "Maximum call-stack depth to search during call-stack unwinding. Decreasing this value will result in sampling consuming less memory"
)
rocprofiler_systems_add_feature(
    ROCPROFSYS_MAX_UNWIND_DEPTH
    "Maximum call-stack depth to search during call-stack unwinding. Decreasing this value will result in sampling consuming less memory"
)

# default visibility settings
set(CMAKE_C_VISIBILITY_PRESET
    "default"
    CACHE STRING
    "Visibility preset for non-inline C functions"
)
set(CMAKE_CXX_VISIBILITY_PRESET
    "default"
    CACHE STRING
    "Visibility preset for non-inline C++ functions/objects"
)
set(CMAKE_VISIBILITY_INLINES_HIDDEN
    OFF
    CACHE BOOL
    "Visibility preset for inline functions"
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(Formatting) # format target
include(Packages) # finds third-party libraries

rocprofiler_systems_activate_clang_tidy()

# custom visibility settings
if(ROCPROFSYS_BUILD_HIDDEN_VISIBILITY)
    set(CMAKE_C_VISIBILITY_PRESET "internal")
    set(CMAKE_CXX_VISIBILITY_PRESET "internal")
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
endif()

if(ROCPROFSYS_BUILD_TESTING OR "$ENV{ROCPROFSYS_CI}" MATCHES "[1-9]+|ON|on|y|yes")
    enable_testing()
    include(CTest)
endif()

if(ROCPROFSYS_ENABLE_BENCHMARK)
    add_compile_definitions(-DROCPROFSYS_USE_BENCHMARK=1)
endif()

# ------------------------------------------------------------------------------#
#
# library and executables
#
# ------------------------------------------------------------------------------#

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME core)

if(ROCPROFSYS_BUILD_CODECOV)
    rocprofiler_systems_save_variables(CODECOV_FLAGS VARIABLES CMAKE_C_FLAGS
                                                               CMAKE_CXX_FLAGS
    )
    foreach(_BUILD_TYPE DEBUG MINSIZEREL RELWITHDEBINFO RELEASE)
        rocprofiler_systems_save_variables(
            CODECOV_FLAGS VARIABLES CMAKE_C_FLAGS_${_BUILD_TYPE}
                                    CMAKE_CXX_FLAGS_${_BUILD_TYPE}
        )
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

if(ROCPROFSYS_BUILD_CODECOV)
    rocprofiler_systems_restore_variables(CODECOV_FLAGS VARIABLES CMAKE_C_FLAGS
                                                                  CMAKE_CXX_FLAGS
    )

    foreach(_BUILD_TYPE DEBUG MINSIZEREL RELWITHDEBINFO RELEASE)
        rocprofiler_systems_restore_variables(
            CODECOV_FLAGS VARIABLES CMAKE_C_FLAGS_${_BUILD_TYPE}
                                    CMAKE_CXX_FLAGS_${_BUILD_TYPE}
        )
    endforeach()
endif()

# ------------------------------------------------------------------------------#
#
# miscellaneous installs
#
# ------------------------------------------------------------------------------#

configure_file(
    ${PROJECT_SOURCE_DIR}/LICENSE
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/doc/${PROJECT_NAME}/LICENSE
    COPYONLY
)

configure_file(
    ${PROJECT_SOURCE_DIR}/perfetto.cfg
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/perfetto.cfg
    COPYONLY
)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/setup-env.sh.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/setup-env.sh
    @ONLY
)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/modulefile.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${ROCPROFSYS_VERSION}
    @ONLY
)

configure_file(
    ${PROJECT_SOURCE_DIR}/scripts/merge-multiprocess-output.sh
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/rocprof-sys-merge-output.sh
    COPYONLY
)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/setup-env.sh
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/perfetto.cfg
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}
    COMPONENT setup
)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${ROCPROFSYS_VERSION}
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}
    COMPONENT setup
)

install(
    FILES ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/doc/${PROJECT_NAME}/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${PROJECT_NAME}
    COMPONENT setup
)

install(
    PROGRAMS
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}/rocprof-sys-merge-output.sh
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME}
    COMPONENT setup
)

# ------------------------------------------------------------------------------#
#
# install
#
# ------------------------------------------------------------------------------#

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME core)
include(ConfigInstall)

# ------------------------------------------------------------------------------#
#
# examples
#
# ------------------------------------------------------------------------------#

if(ROCPROFSYS_BUILD_EXAMPLES)
    set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME examples)
    add_subdirectory(examples)
endif()

# ------------------------------------------------------------------------------#
#
# tests
#
# ------------------------------------------------------------------------------#

if(ROCPROFSYS_BUILD_TESTING)
    set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME testing)
    add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------#
#
# packaging
#
# ------------------------------------------------------------------------------#

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME core)
include(ConfigCPack)

# ------------------------------------------------------------------------------#
#
# config info
#
# ------------------------------------------------------------------------------#

rocprofiler_systems_print_features()
