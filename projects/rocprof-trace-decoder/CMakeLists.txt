#
# Build Standaone Library for Advanced Thread Tracer
#
cmake_minimum_required(VERSION 3.16.0)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake)

set(VERSION_PATCH ${VERSION_PATCH})

# Set module name and project name.
project(
    rocprof-trace-decoder
    VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
    LANGUAGES C CXX)
set(PROJECT_NAME ${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/cmake)
set(TTD_PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR})

set(PROJECT_VERSION_MAJOR ${VERSION_MAJOR})
set(PROJECT_VERSION_MINOR ${VERSION_MINOR})
set(PROJECT_VERSION_PATCH ${VERSION_PATCH})

# Setup the package version.
message("-- BUILD-VERSION: ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
set(LIB_VERSION_STRING "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Default install prefix to /opt/rocm for ROCm projects
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX
        "/opt/rocm"
        CACHE PATH "Default ROCm install prefix" FORCE)
endif()

include(GNUInstallDirs) # install directories
set(CMAKE_INSTALL_LIBDIR "lib") # rocm doesn't use lib64

set(TTD_ALL_VERSIONS ON)

option(BUILD_TESTS "Enable Building Tests for ROCProf Trace Decoder" OFF)
option(BUILD_UNIT_TESTS "Enable Building Unit Tests for ROCProf Trace Decoder" ON)
set(BUILD_PYTHON
    "AUTO"
    CACHE STRING "Install Python API: AUTO, ON, or OFF")

set_property(CACHE BUILD_PYTHON PROPERTY STRINGS AUTO ON OFF)
option(DISABLE_COMGR "Disable dependency on amd_comgr disassembly" OFF)

add_subdirectory(source)

# The marker producer lives in llvm-project/amd/sqtt-marker.  Keep the
# decoder-side tools and format documentation in this package independently
# of that plugin.
set(TTD_MARKER_SCRIPTS
    ${CMAKE_SOURCE_DIR}/scripts/sqtt_data.py ${CMAKE_SOURCE_DIR}/scripts/sqtt_decode_funcmap.py
    ${CMAKE_SOURCE_DIR}/scripts/sqtt_flamegraph.py ${CMAKE_SOURCE_DIR}/scripts/sqtt_memory_trace.py
    ${CMAKE_SOURCE_DIR}/scripts/sqtt_perfetto.py)

install(
    PROGRAMS ${TTD_MARKER_SCRIPTS}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT runtime)

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/docs/markers/
    DESTINATION ${CMAKE_INSTALL_DOCDIR}/markers
    COMPONENT devel)

if(BUILD_PYTHON)
    set(Python3_FIND_UNVERSIONED_NAMES LAST)
    if(BUILD_PYTHON STREQUAL "AUTO")
        find_package(
            Python3 3.8
            COMPONENTS Interpreter
            QUIET)
    else()
        find_package(
            Python3 3.8
            COMPONENTS Interpreter
            REQUIRED)
    endif()

    if(Python3_FOUND)
        set(ROCPROF_TRACE_DECODER_PYTHON_INSTALL_DIR
            "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}/site-packages"
            CACHE PATH "Install directory for the rocprof-trace-decoder Python API")

        install(
            DIRECTORY python/rocprof_trace_decoder
            DESTINATION ${ROCPROF_TRACE_DECODER_PYTHON_INSTALL_DIR}
            COMPONENT runtime
            PATTERN "__pycache__" EXCLUDE
            PATTERN "*.pyc" EXCLUDE)
    else()
        message(STATUS "Skipping Python API install: Python 3.8+ interpreter not found")
    endif()
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

include(packaging)

# Format target: runs clang-format, cmake-format, and autopep8.
find_program(CLANG_FORMAT NAMES clang-format-14)
find_program(CMAKE_FORMAT cmake-format)
find_program(
    AUTOPEP8
    NAMES autopep8
    DOC "Path to autopep8 Python formatter")

set(FORMAT_COMMANDS)

if(CLANG_FORMAT)
    file(
        GLOB_RECURSE
        ALL_CXX_SOURCES
        ${CMAKE_SOURCE_DIR}/source/*.cpp
        ${CMAKE_SOURCE_DIR}/source/*.hpp
        ${CMAKE_SOURCE_DIR}/source/*.h
        ${CMAKE_SOURCE_DIR}/include/*.hpp
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/test/*.cpp
        ${CMAKE_SOURCE_DIR}/test/*.hpp
        ${CMAKE_SOURCE_DIR}/test/*.h)
    list(FILTER ALL_CXX_SOURCES EXCLUDE REGEX ".*/test/att-tool/.*")
    list(FILTER ALL_CXX_SOURCES EXCLUDE REGEX ".*/build.*/.*")
    list(APPEND FORMAT_COMMANDS COMMAND ${CLANG_FORMAT} -i ${ALL_CXX_SOURCES})
endif()

if(CMAKE_FORMAT)
    set(ALL_CMAKE_FILES
        ${CMAKE_SOURCE_DIR}/CMakeLists.txt ${CMAKE_SOURCE_DIR}/source/CMakeLists.txt
        ${CMAKE_SOURCE_DIR}/test/CMakeLists.txt ${CMAKE_SOURCE_DIR}/test/unit/CMakeLists.txt)
    file(GLOB CMAKE_MODULES ${CMAKE_SOURCE_DIR}/cmake/*.cmake)
    list(APPEND ALL_CMAKE_FILES ${CMAKE_MODULES})
    list(APPEND FORMAT_COMMANDS COMMAND ${CMAKE_FORMAT} -i ${ALL_CMAKE_FILES})
endif()

if(AUTOPEP8)
    file(GLOB_RECURSE ALL_PYTHON_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/python/*.py
         ${CMAKE_SOURCE_DIR}/samples/*.py ${CMAKE_SOURCE_DIR}/test/*.py)
    list(FILTER ALL_PYTHON_SOURCES EXCLUDE REGEX ".*/build.*/.*")
    list(
        APPEND
        FORMAT_COMMANDS
        COMMAND
        ${AUTOPEP8}
        --in-place
        --max-line-length
        100
        --ignore
        E402
        ${ALL_PYTHON_SOURCES})
endif()

if(FORMAT_COMMANDS)
    add_custom_target(
        format
        ${FORMAT_COMMANDS}
        COMMENT "Running formatters on source files"
        VERBATIM)
endif()

# Documentation target: runs doxygen to generate API docs
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${CMAKE_SOURCE_DIR}/README.md)
    doxygen_add_docs(docs ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/source ${CMAKE_SOURCE_DIR}/README.md
                     COMMENT "Generating API documentation with Doxygen")
endif()

# Coverage
find_program(LCOV lcov)
find_program(GENHTML genhtml)
if(BUILD_TESTS
   AND LCOV
   AND GENHTML)
    set(TTD_COVERAGE_DEPS rocprof-trace-decoder)
    if(TARGET unit_tests)
        list(APPEND TTD_COVERAGE_DEPS unit_tests)
        list(APPEND TTD_COVERAGE_DEPS unit_tests_logging)
    endif()
    set(TTD_COVERAGE_SOURCES "${CMAKE_SOURCE_DIR}/source/*")

    add_custom_target(
        coverage
        # Remove stale profiling data
        COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
        # Run tests (skip sanitizer variants)
        COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR}/test -E "sanitize|ubsan|asan" -j8
                --output-on-failure
        # Capture coverage
        COMMAND ${LCOV} --quiet --capture --directory ${CMAKE_BINARY_DIR} --output-file
                ${CMAKE_BINARY_DIR}/coverage.info --gcov-tool gcov --ignore-errors mismatch,mismatch,gcov,empty
        # Extract only project sources (use full path to avoid matching
        # googletest-src etc.)
        COMMAND ${LCOV} --quiet --extract ${CMAKE_BINARY_DIR}/coverage.info ${TTD_COVERAGE_SOURCES} --output-file
                ${CMAKE_BINARY_DIR}/coverage_src.info --ignore-errors empty,unused
        # Generate HTML report
        COMMAND ${GENHTML} --quiet ${CMAKE_BINARY_DIR}/coverage_src.info --output-directory
                ${CMAKE_BINARY_DIR}/coverage_report --ignore-errors empty
        COMMAND ${CMAKE_COMMAND} -E echo "Coverage report: ${CMAKE_BINARY_DIR}/coverage_report/index.html"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating code coverage report"
        DEPENDS ${TTD_COVERAGE_DEPS}
        VERBATIM)

endif()
