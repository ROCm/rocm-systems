cmake_minimum_required(VERSION 3.16.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(
        SQTTInstrument
        VERSION 0.0.0
        LANGUAGES C CXX)

    set(TTD_PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

    include(GNUInstallDirs)
    set(CMAKE_INSTALL_LIBDIR "lib")
elseif(NOT DEFINED TTD_PROJECT_SOURCE_DIR)
    set(TTD_PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

# Prefer ROCm LLVM so the plugin ABI matches the compiler process loading it.
# CMAKE_PREFIX_PATH can be used to choose a specific ROCm install, for example:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/rocm-7.13.0
set(SQTT_ROCM_PREFIX_HINTS ${CMAKE_PREFIX_PATH} /opt/rocm)
find_package(
    LLVM
    CONFIG
    QUIET
    PATHS
    ${SQTT_ROCM_PREFIX_HINTS}
    PATH_SUFFIXES
    lib/llvm/lib/cmake/llvm
    lib/cmake/llvm
    NO_DEFAULT_PATH)

if(NOT LLVM_FOUND)
    find_package(LLVM REQUIRED CONFIG)
    message(WARNING "ROCm LLVM was not found via CMAKE_PREFIX_PATH or /opt/rocm; using LLVM from ${LLVM_DIR}")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_library(
    SQTTInstrumentPass MODULE
    SQTTPass.cpp
    SQTTMarkers.cpp
    SQTTNamedMarkers.cpp
    SQTTAutomatic.cpp
    SQTTAddressTrace.cpp
    SQTTFunctions.cpp
    SQTTFuncMap.cpp)

target_include_directories(SQTTInstrumentPass PRIVATE ${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(SQTTInstrumentPass PRIVATE ${LLVM_DEFINITIONS})
target_compile_features(SQTTInstrumentPass PRIVATE cxx_std_17)

if(NOT MSVC)
    target_compile_options(SQTTInstrumentPass PRIVATE -fno-rtti)
endif()

# Pass plugins must not link LLVM libraries: they get symbols from the host compiler process.
set_target_properties(
    SQTTInstrumentPass
    PROPERTIES NO_SONAME ON
               PREFIX "lib"
               OUTPUT_NAME "sqttinstrumentpass"
               POSITION_INDEPENDENT_CODE ON
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
               ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

install(
    TARGETS SQTTInstrumentPass
    EXPORT rocprof-trace-decoder-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)

set(TTD_MARKER_SCRIPTS
    ${TTD_PROJECT_SOURCE_DIR}/scripts/sqtt_data.py ${TTD_PROJECT_SOURCE_DIR}/scripts/sqtt_decode_funcmap.py
    ${TTD_PROJECT_SOURCE_DIR}/scripts/sqtt_flamegraph.py ${TTD_PROJECT_SOURCE_DIR}/scripts/sqtt_memory_trace.py
    ${TTD_PROJECT_SOURCE_DIR}/scripts/sqtt_perfetto.py)

install(
    PROGRAMS ${TTD_MARKER_SCRIPTS}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT runtime)

install(
    DIRECTORY ${TTD_PROJECT_SOURCE_DIR}/docs/markers/
    DESTINATION ${CMAKE_INSTALL_DOCDIR}/markers
    COMPONENT devel)
