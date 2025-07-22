# include guard
include_guard(GLOBAL)

# ##########################################################################################
#
# Configure and aggregate system options, features, and test infrastructure for ctest build.
#
# ##########################################################################################

#___________________________________________________________________________________________
# From CMakeLists.txt
rocprofiler_systems_add_option(ROCPROFSYS_USE_MPI "Enable MPI support" OFF)
rocprofiler_systems_add_option(ROCPROFSYS_USE_ROCM "Enable ROCm support" ON)
rocprofiler_systems_add_option(ROCPROFSYS_USE_PYTHON "Enable Python support" ON)
rocprofiler_systems_add_option(ROCPROFSYS_BUILD_TESTING
                                   "Enable building the testing suite" ON ADVANCED
)
rocprofiler_systems_add_option(ROCPROFSYS_BUILD_PYTHON
                                   "Build python bindings with internal pybind11" OFF
)
rocprofiler_systems_add_option(
    ROCPROFSYS_USE_MPI_HEADERS
    "Enable wrapping MPI functions w/o enabling MPI dependency" ON
)

set(ROCPROFSYS_ABORT_FAIL_REGEX
    "### ERROR ###|unknown-hash=|address of faulting memory reference|exiting with non-zero exit code|terminate called after throwing an instance|calling abort.. in |Exit code: [1-9]"
    CACHE INTERNAL
    "Regex to catch abnormal exits when a PASS_REGULAR_EXPRESSION is set"
    FORCE
)

if(NOT "$ENV{ROCPROFSYS_CI}" STREQUAL "")
    set(CI_BUILD $ENV{ROCPROFSYS_CI})
else()
    set(CI_BUILD OFF)
endif()

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
#___________________________________________________________________________________________
# Python configuration from cmake/Packages.cmake
if(ROCPROFSYS_USE_PYTHON)
    find_package(pybind11 REQUIRED)
    include(ConfigPython)
    include(PyBind11Tools)
    rocprofiler_systems_find_python(_PY REQUIRED)
    set(ROCPROFSYS_PYTHON_ROOT_DIRS "${_PY_ROOT_DIR}" CACHE INTERNAL "" FORCE)
    set(ROCPROFSYS_PYTHON_VERSIONS "${_PY_VERSION}" CACHE INTERNAL "" FORCE)
    set(DISABLE_PYTHON_TESTS OFF)
    # Get Python3 executable
    find_package(Python3 COMPONENTS Interpreter)
    if(NOT Python3_FOUND)
        rocprofiler_systems_message(
            AUTHOR_WARNING
            "Python3 not found. Disabling Python tests."
        )
        set(DISABLE_PYTHON_TESTS ON)
    endif()
    # Ensure rocprofsys module is present
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import rocprofsys"
        RESULT_VARIABLE FOUND_MODULE_ROCPROFSYS
    )
    if(NOT FOUND_MODULE_ROCPROFSYS EQUAL 0)
        set(DISABLE_PYTHON_TESTS ON)
        rocprofiler_systems_message(
            AUTHOR_WARNING
            "Python3 module 'rocprofsys' not found. Disabling Python tests."
        )
    endif()
    # Disable Python tests if python3 or rocprofsys module not found
    if(DISABLE_PYTHON_TESTS)
        if(NOT DEFINED ROCPROFSYS_DISABLE_EXAMPLES)
            set(ROCPROFSYS_DISABLE_EXAMPLES
                "python"
                CACHE STRING
                "Disabled examples"
                FORCE
            )
        else()
            list(APPEND ROCPROFSYS_DISABLE_EXAMPLES python)
            set(ROCPROFSYS_DISABLE_EXAMPLES
                "${ROCPROFSYS_DISABLE_EXAMPLES};python"
                CACHE STRING
                "Disabled examples"
                FORCE
            )
        endif()
    endif()
endif()

#___________________________________________________________________________________________
# Set ${ROCPROFSYS_MAX_THREADS}
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
#___________________________________________________________________________________________
# Installs Python scripts from /cmake/ConfigInstall.cmake
configure_file(
    ${CTEST_CONFIG_D}/validate-causal-json.py
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprof-sys-causal-print
    COPYONLY
)
install(
    PROGRAMS ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprof-sys-causal-print
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT scripts
)
#___________________________________________________________________________________________
# Set $<TARGET_FILE:rocprofiler-systems-user-library>
#       (For ctest "rocprofiler-systems-instrument-simulate-lib-basename")
find_file(
    ROCPROFSYS_USER_LIBRARY_PATH
    NAMES librocprof-sys-user.so.${PACKAGE_VERSION}
    PATHS ${ROCM_PATH}/lib
    NO_DEFAULT_PATH
)
if(NOT ROCPROFSYS_USER_LIBRARY_PATH)
    message(FATAL_ERROR "Could not find librocprof-sys-user.so in /opt/rocm/lib.")
else()
    message(STATUS "Found pre-installed user library: ${ROCPROFSYS_USER_LIBRARY_PATH}")
    # Create an IMPORTED target that represents the pre-built library
    add_library(rocprofiler-systems-user-library SHARED IMPORTED)
    set_target_properties(
        rocprofiler-systems-user-library
        PROPERTIES IMPORTED_LOCATION "${ROCPROFSYS_USER_LIBRARY_PATH}"
    )
    # Namespace alias required
    if(NOT TARGET rocprofiler-systems::rocprofiler-systems-user-library)
        add_library(
            rocprofiler-systems::rocprofiler-systems-user-library
            ALIAS rocprofiler-systems-user-library
        )
    endif()

    # Define the name the test executable is looking for
    set(EXPECTED_LIBRARY_NAME "librocprofiler-systems-user-library.so")
    set(SYMLINK_DIR "${PROJECT_BINARY_DIR}/lib")
    set(SYMLINK_PATH "${SYMLINK_DIR}/${EXPECTED_LIBRARY_NAME}")

    # Ensure the directory exists
    file(MAKE_DIRECTORY ${SYMLINK_DIR})
    # Create the symbolic link, overwriting if it already exists
    execute_process(
        COMMAND
            ${CMAKE_COMMAND} -E create_symlink ${ROCPROFSYS_USER_LIBRARY_PATH}
            ${SYMLINK_PATH}
    )

    message(
        STATUS
        "Created symlink for tests: ${SYMLINK_PATH} -> ${ROCPROFSYS_USER_LIBRARY_PATH}"
    )
endif()
