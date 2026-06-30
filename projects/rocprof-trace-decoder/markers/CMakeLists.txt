cmake_minimum_required(VERSION 3.20)
project(SQTTInstrument LANGUAGES C CXX)

# Find ROCm LLVM (avoid system LLVM). Set CMAKE_PREFIX_PATH to choose a
# specific ROCm install, for example:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/rocm-7.13.0
set(SQTT_ROCM_PREFIX_HINTS ${CMAKE_PREFIX_PATH} /opt/rocm)
find_package(
    LLVM REQUIRED
    CONFIG
    PATHS ${SQTT_ROCM_PREFIX_HINTS}
    PATH_SUFFIXES lib/llvm/lib/cmake/llvm lib/cmake/llvm
    NO_DEFAULT_PATH)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Pass plugin shared library
add_library(
    SQTTInstrumentPass MODULE
    src/SQTTPass.cpp
    src/SQTTMarkers.cpp
    src/SQTTNamedMarkers.cpp
    src/SQTTBarriers.cpp
    src/SQTTMemory.cpp
    src/SQTTAddressTrace.cpp
    src/SQTTFunctions.cpp
    src/SQTTFuncMap.cpp)

target_include_directories(SQTTInstrumentPass PRIVATE ${LLVM_INCLUDE_DIRS} include src)

target_compile_definitions(SQTTInstrumentPass PRIVATE ${LLVM_DEFINITIONS})

target_compile_features(SQTTInstrumentPass PRIVATE cxx_std_17)

# Pass plugins must NOT link LLVM libraries -- they get symbols from the host process
set_target_properties(
    SQTTInstrumentPass PROPERTIES NO_SONAME ON PREFIX "" # produce SQTTInstrumentPass.so, not libSQTTInstrumentPass.so
)

target_compile_options(SQTTInstrumentPass PRIVATE -fno-rtti -fPIC)

# ----------------------------------------------------------------------------
# `format` target: rewrite C/C++ sources via clang-format and CMake files via
# cmake-format, using the .clang-format / .cmake-format.yaml at the repo root.
# Build with:  cmake --build <build-dir> --target format
# Missing formatters print a warning but don't fail the build.
# ----------------------------------------------------------------------------
file(
    GLOB_RECURSE
    SQTT_CXX_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.h"
    "${CMAKE_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_SOURCE_DIR}/include/*.h")

set(SQTT_CMAKE_FILES "${CMAKE_SOURCE_DIR}/CMakeLists.txt")

# Pinned to clang-format-14 -- other versions produce different output and
# would cause format churn across contributors. Override with
# -DCLANG_FORMAT_EXECUTABLE=... only if you've verified your version matches.
find_program(CLANG_FORMAT_EXECUTABLE NAMES clang-format-14)
find_program(CMAKE_FORMAT_EXECUTABLE NAMES cmake-format)

if(CLANG_FORMAT_EXECUTABLE)
    set(_fmt_cxx_cmd
        COMMAND
        ${CMAKE_COMMAND}
        -E
        echo
        "clang-format: ${CLANG_FORMAT_EXECUTABLE}"
        COMMAND
        ${CLANG_FORMAT_EXECUTABLE}
        -i
        --style=file
        ${SQTT_CXX_SOURCES})
else()
    set(_fmt_cxx_cmd COMMAND ${CMAKE_COMMAND} -E echo "warning: clang-format not found, skipping C++ formatting")
endif()

if(CMAKE_FORMAT_EXECUTABLE)
    set(_fmt_cmake_cmd
        COMMAND
        ${CMAKE_COMMAND}
        -E
        echo
        "cmake-format: ${CMAKE_FORMAT_EXECUTABLE}"
        COMMAND
        ${CMAKE_FORMAT_EXECUTABLE}
        -i
        ${SQTT_CMAKE_FILES})
else()
    set(_fmt_cmake_cmd COMMAND ${CMAKE_COMMAND} -E echo "warning: cmake-format not found, skipping CMake formatting")
endif()

add_custom_target(
    format
    ${_fmt_cxx_cmd} ${_fmt_cmake_cmd}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Formatting sources"
    VERBATIM)
