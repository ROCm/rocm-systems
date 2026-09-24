#
# Tool library used by rocprofiler
#

rocprofiler_activate_clang_tidy()

set(TOOL_HEADERS
    att_no_intercept.hpp
    att_no_intercept_impl.hpp
    config.hpp
    execution_profile.hpp
    graph_stack.hpp
    helper.hpp
    kernel_filter_range.hpp
    kernel_iteration_filter.hpp
    stream_stack.hpp)

set(TOOL_SOURCES att_no_intercept.cpp config.cpp graph_stack.cpp kernel_filter_range.cpp
                 kernel_iteration_filter.cpp stream_stack.cpp tool.cpp)

# main.c hooks __libc_start_main so that the tool can wrap the application's main(). There
# is no equivalent on Windows; rocprofv3-launch calls rocprofiler_force_configure() and
# rocprofiler_finalize() explicitly instead.
if(NOT WIN32)
    list(APPEND TOOL_SOURCES main.c)
endif()

if(rocprof-trace-decoder_FOUND)
    list(APPEND TOOL_SOURCES att_no_intercept_quick_scan.cpp)
endif()

add_library(rocprofiler-sdk-tool SHARED)
target_sources(rocprofiler-sdk-tool PRIVATE ${TOOL_SOURCES} ${TOOL_HEADERS})

target_link_libraries(
    rocprofiler-sdk-tool
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
            rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-output-library
            rocprofiler-sdk::rocprofiler-sdk-cereal)

# These all back an output format or the ATT decoder, none of which are built on Windows.
if(NOT WIN32)
    target_link_libraries(
        rocprofiler-sdk-tool
        PRIVATE rocprofiler-sdk::rocprofiler-sdk-perfetto
                rocprofiler-sdk::rocprofiler-sdk-otf2
                rocprofiler-sdk::rocprofiler-sdk-dw
                rocprofiler-sdk::rocprofiler-sdk-amd-comgr
                rocprofiler-sdk::rocprofiler-sdk-att-parser
                rocprofiler-sdk::rocprofiler-sdk-sqlite3)
endif()

if(rocprof-trace-decoder_FOUND)
    target_link_libraries(rocprofiler-sdk-tool
                          PRIVATE rocprof-trace-decoder::rocprof-trace-decoder)
    target_compile_definitions(rocprofiler-sdk-tool
                               PRIVATE ROCPROFILER_ATT_QUICK_SCAN_ENABLED=1)
else()
    target_compile_definitions(rocprofiler-sdk-tool
                               PRIVATE ROCPROFILER_ATT_QUICK_SCAN_ENABLED=0)
endif()

if(WIN32)
    # DLLs are resolved from the loader search path, not an RPATH, and the version suffix
    # chain (libfoo.so.1.2.3) has no Windows counterpart.
    set_target_properties(rocprofiler-sdk-tool PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                          ${PROJECT_BINARY_DIR}/bin)

    install(
        TARGETS rocprofiler-sdk-tool
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT tools
        EXPORT rocprofiler-sdk-tool-targets)
else()
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
endif()
