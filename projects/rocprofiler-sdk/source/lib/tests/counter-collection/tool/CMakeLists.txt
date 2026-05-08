#
# Counter collection tool shared library
#
set_source_files_properties(counters-tool.cpp PROPERTIES LANGUAGE HIP)

add_library(rocprofiler-sdk-counters-tool SHARED)
target_sources(rocprofiler-sdk-counters-tool PRIVATE counters-tool.cpp)
target_include_directories(rocprofiler-sdk-counters-tool
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    rocprofiler-sdk-counters-tool PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
                                          rocprofiler-sdk::rocprofiler-sdk-fmt)
set_target_properties(
    rocprofiler-sdk-counters-tool PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                             "${CMAKE_BINARY_DIR}/lib/rocprofiler-sdk")
