#
#   mock generic-tool library
#
add_library(generic-tool SHARED)
add_library(generic-tool::generic-tool ALIAS generic-tool)
target_sources(generic-tool PRIVATE generic-tool.cpp)
target_include_directories(
    generic-tool PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# target_link_libraries(generic-tool PRIVATE rocprofiler)
rocp_register_strip_target(generic-tool)
