#
#   mock rocprofiler library
#
add_library(rocprofiler SHARED)
add_library(rocprofiler::rocprofiler ALIAS rocprofiler)
target_sources(rocprofiler PRIVATE rocprofiler.cpp)
target_include_directories(rocprofiler PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
set_target_properties(
    rocprofiler
    PROPERTIES OUTPUT_NAME rocprofiler64
               SOVERSION 2
               VERSION 2.0.0)
rocp_register_strip_target(rocprofiler)
