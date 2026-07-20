#
# Tool library used by rocprofiler
#

rocprofiler_activate_clang_tidy()

set(TOOL_HEADERS att_no_intercept.hpp att_no_intercept_impl.hpp config.hpp
                 execution_profile.hpp graph_stack.hpp helper.hpp stream_stack.hpp)

set(TOOL_SOURCES att_no_intercept.cpp config.cpp graph_stack.cpp main.c tool.cpp
                 stream_stack.cpp)

if(NOT ROCPROFILER_DISABLE_TRACE_DECODER)
    rocprofiler_sdk_find_trace_decoder()
    list(APPEND TOOL_SOURCES att_no_intercept_quick_scan.cpp)
endif()

add_library(rocprofiler-sdk-tool SHARED)
target_sources(rocprofiler-sdk-tool PRIVATE ${TOOL_SOURCES} ${TOOL_HEADERS})

if(ROCPROFILER_DISABLE_TRACE_DECODER)
    target_compile_definitions(rocprofiler-sdk-tool
                               PRIVATE ROCPROFILER_DISABLE_TRACE_DECODER=1)
endif()

target_link_libraries(
    rocprofiler-sdk-tool
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
            rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-output-library
            rocprofiler-sdk::rocprofiler-sdk-cereal
            rocprofiler-sdk::rocprofiler-sdk-perfetto
            rocprofiler-sdk::rocprofiler-sdk-otf2
            rocprofiler-sdk::rocprofiler-sdk-dw
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-sqlite3)

if(NOT ROCPROFILER_DISABLE_TRACE_DECODER)
    target_link_libraries(rocprofiler-sdk-tool
                          PRIVATE rocprofiler-sdk::rocprofiler-sdk-att-parser)
    target_compile_definitions(rocprofiler-sdk-tool
                               PRIVATE ROCPROFILER_ATT_QUICK_SCAN_ENABLED=1)
else()
    target_compile_definitions(rocprofiler-sdk-tool
                               PRIVATE ROCPROFILER_ATT_QUICK_SCAN_ENABLED=0)
endif()

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

add_subdirectory(kokkosp)
