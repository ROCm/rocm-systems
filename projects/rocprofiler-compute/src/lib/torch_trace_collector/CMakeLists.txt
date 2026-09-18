## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

# Builds against the stub headers in torch_abi/, so no PyTorch install is
# needed. ATen and c10 symbols resolve against the workload's libtorch.
macro(_skip_collector reason)
    message(WARNING "torch_trace_collector not built: ${reason}")
    return()
endmacro()

find_package(rocprofiler-sdk-roctx CONFIG)
if(NOT rocprofiler-sdk-roctx_FOUND)
    _skip_collector("rocprofiler-sdk-roctx not found")
endif()

# Headers only; the interpreter that loads the module resolves Python symbols.
find_package(Python3 COMPONENTS Development.Module QUIET)
if(NOT Python3_Development.Module_FOUND)
    _skip_collector("Python 3 development headers not found")
endif()

# Skip rather than let target_compile_features fail the whole configure.
if(NOT cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    _skip_collector("compiler does not support C++20")
endif()

add_library(
    torch-trace-collector
    STATIC
    torch_trace_collector.cpp
    snapshot_store.cpp
    user_scope_kind.cpp
)
target_include_directories(torch-trace-collector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# PRIVATE so the tests, which need the real ATen headers, never see the stubs.
target_include_directories(
    torch-trace-collector
    SYSTEM
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/torch_abi
)
target_link_libraries(
    torch-trace-collector
    PUBLIC
        rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library
        synchronized
        gsl_assert
)
# The stub headers need a newer standard than the C++17 used across src/lib.
target_compile_features(torch-trace-collector PUBLIC cxx_std_20)
target_compile_options(torch-trace-collector PRIVATE -fvisibility=hidden)

add_library(torch-trace-collector-module MODULE torch_trace_collector_module.cpp)
target_link_libraries(torch-trace-collector-module PRIVATE torch-trace-collector)
target_include_directories(
    torch-trace-collector-module
    SYSTEM
    PRIVATE ${Python3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/torch_abi
)
target_compile_options(torch-trace-collector-module PRIVATE -fvisibility=hidden)
# ATen, c10 and the Python C API are all resolved by whoever loads the module.
target_link_options(
    torch-trace-collector-module
    PRIVATE LINKER:--unresolved-symbols=ignore-all
)
set_target_properties(
    torch-trace-collector-module
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "torch_trace_collector"
        SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
