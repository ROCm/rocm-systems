#
# Build Standaone Library for Advanced Thread Tracer
#
cmake_minimum_required(VERSION 3.16.0)

if(NOT DEFINED VERSION_MAJOR)
    set(VERSION_MAJOR 0)
endif()

if(NOT DEFINED VERSION_MINOR)
    set(VERSION_MINOR 1)
endif()

if(NOT DEFINED VERSION_PATCH)
    set(VERSION_PATCH 7)
endif()

set(VERSION_PATCH ${VERSION_PATCH})

# Set module name and project name.
project(
    rocprof-trace-decoder
    VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
    LANGUAGES CXX)
set(PROJECT_NAME ${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
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

add_subdirectory(source)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

include(packaging)

# Format target: runs clang-format and cmake-format
find_program(CLANG_FORMAT NAMES clang-format-14)
find_program(CMAKE_FORMAT cmake-format)

set(FORMAT_COMMANDS)

if(CLANG_FORMAT)
    file(
        GLOB_RECURSE
        ALL_CXX_SOURCES
        ${CMAKE_SOURCE_DIR}/source/*.cpp
        ${CMAKE_SOURCE_DIR}/source/*.hpp
        ${CMAKE_SOURCE_DIR}/source/*.h
        ${CMAKE_SOURCE_DIR}/test/*.cpp
        ${CMAKE_SOURCE_DIR}/test/*.hpp
        ${CMAKE_SOURCE_DIR}/test/*.h)
    list(FILTER ALL_CXX_SOURCES EXCLUDE REGEX ".*/test/att-tool/.*")
    list(FILTER ALL_CXX_SOURCES EXCLUDE REGEX ".*/build.*/.*")
    list(APPEND FORMAT_COMMANDS COMMAND ${CLANG_FORMAT} -i ${ALL_CXX_SOURCES})
endif()

if(CMAKE_FORMAT)
    set(ALL_CMAKE_FILES ${CMAKE_SOURCE_DIR}/CMakeLists.txt ${CMAKE_SOURCE_DIR}/source/CMakeLists.txt
                        ${CMAKE_SOURCE_DIR}/test/CMakeLists.txt ${CMAKE_SOURCE_DIR}/test/unit/CMakeLists.txt)
    file(GLOB CMAKE_MODULES ${CMAKE_SOURCE_DIR}/cmake/*.cmake)
    list(APPEND ALL_CMAKE_FILES ${CMAKE_MODULES})
    list(APPEND FORMAT_COMMANDS COMMAND ${CMAKE_FORMAT} -i ${ALL_CMAKE_FILES})
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
    add_custom_target(
        coverage
        # Remove stale profiling data
        COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
        # Run tests (skip sanitizer variants)
        COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR}/test -E sanitize -E ubsan -E asan -j8
                --output-on-failure
        # Capture coverage
        COMMAND ${LCOV} --quiet --capture --directory ${CMAKE_BINARY_DIR} --output-file
                ${CMAKE_BINARY_DIR}/coverage.info --gcov-tool gcov --ignore-errors mismatch,mismatch,gcov,empty
        # Extract only project sources (use full path to avoid matching
        # googletest-src etc.)
        COMMAND ${LCOV} --quiet --extract ${CMAKE_BINARY_DIR}/coverage.info "${CMAKE_SOURCE_DIR}/source/*" --output-file
                ${CMAKE_BINARY_DIR}/coverage_src.info --ignore-errors empty,unused
        # Generate HTML report
        COMMAND ${GENHTML} --quiet ${CMAKE_BINARY_DIR}/coverage_src.info --output-directory
                ${CMAKE_BINARY_DIR}/coverage_report --ignore-errors empty
        COMMAND ${CMAKE_COMMAND} -E echo "Coverage report: ${CMAKE_BINARY_DIR}/coverage_report/index.html"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating code coverage report"
        DEPENDS unit_tests rocprof-trace-decoder
        VERBATIM)
endif()
