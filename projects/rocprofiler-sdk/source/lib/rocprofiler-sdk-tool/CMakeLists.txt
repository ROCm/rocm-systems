#
# Tool library used by rocprofiler
#

rocprofiler_activate_clang_tidy()

set(TOOL_HEADERS
    buffered_output.hpp
    config.hpp
    csv.hpp
    domain_type.hpp
    generateCSV.hpp
    generateJSON.hpp
    helper.hpp
    output_file.hpp
    statistics.hpp
    tmp_file_buffer.hpp
    tmp_file.hpp)

set(TOOL_SOURCES
    config.cpp
    domain_type.cpp
    generateCSV.cpp
    generateJSON.cpp
    helper.cpp
    main.c
    output_file.cpp
    tmp_file_buffer.cpp
    tmp_file.cpp
    tool.cpp)

add_library(rocprofiler-sdk-tool SHARED)
target_sources(rocprofiler-sdk-tool PRIVATE ${TOOL_SOURCES} ${TOOL_HEADERS})

add_subdirectory(plugins)

target_link_libraries(
    rocprofiler-sdk-tool
    PRIVATE rocprofiler::rocprofiler-shared-library rocprofiler::rocprofiler-headers
            rocprofiler::rocprofiler-build-flags rocprofiler::rocprofiler-memcheck
            rocprofiler::rocprofiler-common-library rocprofiler::rocprofiler-cereal)

set_target_properties(
    rocprofiler-sdk-tool
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               BUILD_RPATH "\$ORIGIN:\$ORIGIN/.."
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")

install(
    TARGETS rocprofiler-sdk-tool
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk
    COMPONENT tools
    EXPORT rocprofiler-sdk-tool-targets)

add_executable(rocprofv3-trigger-list-metrics)
target_sources(rocprofv3-trigger-list-metrics PRIVATE rocprofv3_trigger_list_metrics.cpp)
target_link_libraries(rocprofv3-trigger-list-metrics
                      PRIVATE rocprofiler::rocprofiler-hsa-runtime)
set_target_properties(
    rocprofv3-trigger-list-metrics
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk)

install(
    TARGETS rocprofv3-trigger-list-metrics
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk
    COMPONENT tools
    EXPORT rocprofiler-sdk-tool-targets)
