
set(target rocprofiler-compute-tool)
add_library(${target} SHARED
    rocprofiler_compute_tool.h
    rocprofiler_compute_tool.cpp
    sdk_common.h
    sdk_callbacks.h
    sdk_callbacks.cpp
    helper.hpp
    helper.cpp
    input_parameters.h
    input_parameters.cpp
    sdk_wrapper.h
    sdk_wrapper.cpp
)

target_include_directories(${target} PUBLIC .)
target_link_libraries(${target} PUBLIC rocprofiler-sdk::rocprofiler-sdk)

add_subdirectory(test)