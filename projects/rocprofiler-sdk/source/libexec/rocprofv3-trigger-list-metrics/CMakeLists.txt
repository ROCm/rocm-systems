#
#
#

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "tools")

add_executable(rocprofv3-trigger-list-metrics)
target_sources(rocprofv3-trigger-list-metrics PRIVATE rocprofv3_trigger_list_metrics.cpp)
target_link_libraries(rocprofv3-trigger-list-metrics
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk-hsa-runtime)
set_target_properties(
    rocprofv3-trigger-list-metrics
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBEXECDIR}
               BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(
    TARGETS rocprofv3-trigger-list-metrics
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}
    COMPONENT tools
    EXPORT rocprofiler-sdk-tool-targets)
