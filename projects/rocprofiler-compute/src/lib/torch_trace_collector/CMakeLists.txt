## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

macro(_skip_collector reason)
    message(WARNING "torch_trace_collector not built: ${reason}")
    return()
endmacro()

find_package(rocprofiler-sdk-roctx CONFIG)
if(NOT rocprofiler-sdk-roctx_FOUND)
    _skip_collector("rocprofiler-sdk-roctx not found")
endif()

if(NOT cxx_std_17 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    _skip_collector("compiler does not support C++17")
endif()

# The collector compiles against the minimal declarations in torch_abi/, so
# neither PyTorch headers nor PyTorch libraries are build dependencies.
add_library(torch-trace-collector STATIC torch_trace_collector.cpp snapshot_store.cpp)
target_include_directories(torch-trace-collector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
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
target_compile_features(torch-trace-collector PUBLIC cxx_std_17)
target_compile_definitions(torch-trace-collector PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
set_target_properties(
    torch-trace-collector
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON
)

add_library(torch-trace-collector-module MODULE torch_trace_collector_module.cpp)
target_include_directories(
    torch-trace-collector-module
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/torch_abi
)
target_link_libraries(torch-trace-collector-module PRIVATE torch-trace-collector)
# Only the plain-C interface is public. PyTorch symbols intentionally remain
# unresolved until the workload loader promotes its real libtorch_cpu.so.
set(_torch_collector_exports "${CMAKE_CURRENT_SOURCE_DIR}/exports.map")
target_link_options(
    torch-trace-collector-module
    PRIVATE "LINKER:--version-script=${_torch_collector_exports}"
)
set_property(
    TARGET torch-trace-collector-module
    APPEND
    PROPERTY LINK_DEPENDS "${_torch_collector_exports}"
)
set_target_properties(
    torch-trace-collector-module
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        PREFIX ""
        OUTPUT_NAME torch_trace_collector
        SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
        SKIP_BUILD_RPATH ON
        INSTALL_RPATH ""
        INSTALL_RPATH_USE_LINK_PATH OFF
)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
