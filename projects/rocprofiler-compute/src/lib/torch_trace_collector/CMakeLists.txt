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

if(NOT cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    _skip_collector("compiler does not support C++20")
endif()

add_library(
    torch-trace-collector-module
    MODULE
    argument_capture.cpp
    torch_abi/runtime.cpp
    torch_trace_collector.cpp
)
target_include_directories(
    torch-trace-collector-module
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/torch_abi
)
target_link_libraries(
    torch-trace-collector-module
    PRIVATE ${CMAKE_DL_LIBS} rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library
)
# Torch imports remain unresolved until the Python loader promotes the
# workload's real libtorch_cpu.so into the process-wide symbol scope.
target_compile_features(torch-trace-collector-module PRIVATE cxx_std_20)
target_compile_definitions(torch-trace-collector-module PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

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
