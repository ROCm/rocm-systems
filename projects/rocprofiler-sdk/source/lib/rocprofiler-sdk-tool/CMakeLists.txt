#
# Tool library used by rocprofiler
#

rocprofiler_activate_clang_tidy()

set(TOOL_HEADERS
    att_no_intercept.hpp
    att_no_intercept_impl.hpp
    config.hpp
    execution_profile.hpp
    finalization_gate.hpp
    graph_stack.hpp
    helper.hpp
    kernel_filter_range.hpp
    kernel_iteration_filter.hpp
    stream_stack.hpp)

set(TOOL_SOURCES
    att_no_intercept.cpp
    config.cpp
    finalization_gate.cpp
    graph_stack.cpp
    kernel_filter_range.cpp
    kernel_iteration_filter.cpp
    main.c
    stream_stack.cpp
    tool.cpp)

if(rocprof-trace-decoder_FOUND)
    list(APPEND TOOL_SOURCES att_no_intercept_quick_scan.cpp)
    set(ROCPROFILER_TOOL_ATT_DEFINITION ROCPROFILER_ATT_QUICK_SCAN_ENABLED=1)
    set(ROCPROFILER_TOOL_ATT_LIBRARY rocprof-trace-decoder::rocprof-trace-decoder)
else()
    set(ROCPROFILER_TOOL_ATT_DEFINITION ROCPROFILER_ATT_QUICK_SCAN_ENABLED=0)
endif()

set(ROCPROFILER_TOOL_LINK_LIBRARIES
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
    rocprofiler-sdk::rocprofiler-sdk-att-parser
    rocprofiler-sdk::rocprofiler-sdk-sqlite3
    ${ROCPROFILER_TOOL_ATT_LIBRARY})

add_library(rocprofiler-sdk-tool SHARED)
target_sources(rocprofiler-sdk-tool PRIVATE ${TOOL_SOURCES} ${TOOL_HEADERS})
target_link_libraries(
    rocprofiler-sdk-tool PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
                                 ${ROCPROFILER_TOOL_LINK_LIBRARIES})
target_compile_definitions(rocprofiler-sdk-tool
                           PRIVATE ${ROCPROFILER_TOOL_ATT_DEFINITION})

set_target_properties(
    rocprofiler-sdk-tool
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               BUILD_RPATH "\$ORIGIN:\$ORIGIN/.."
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..:\$ORIGIN/../rocm_sysdeps/lib")

install(
    TARGETS rocprofiler-sdk-tool
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk
    COMPONENT tools
    EXPORT rocprofiler-sdk-tool-targets)

add_subdirectory(kokkosp)
add_subdirectory(list-avail)

if(ROCPROFILER_BUILD_TESTS)
    # Regression hooks are compiled only into this non-installed test variant.
    add_library(rocprofiler-sdk-tool-fork-test SHARED)
    target_sources(rocprofiler-sdk-tool-fork-test PRIVATE ${TOOL_SOURCES} ${TOOL_HEADERS})
    target_link_libraries(
        rocprofiler-sdk-tool-fork-test PRIVATE rocprofiler-sdk-fork-test-shared
                                               ${ROCPROFILER_TOOL_LINK_LIBRARIES})
    target_compile_definitions(
        rocprofiler-sdk-tool-fork-test PRIVATE ROCPROFILER_TOOL_TEST_HOOKS=1
                                               ${ROCPROFILER_TOOL_ATT_DEFINITION})

    set(_fork_test_root "${PROJECT_BINARY_DIR}/tests/fork-lifecycle-runtime")
    set_target_properties(
        rocprofiler-sdk-tool-fork-test
        PROPERTIES OUTPUT_NAME rocprofiler-sdk-tool
                   LIBRARY_OUTPUT_DIRECTORY "${_fork_test_root}/lib/rocprofiler-sdk"
                   SOVERSION ${PROJECT_VERSION_MAJOR}
                   VERSION ${PROJECT_VERSION}
                   BUILD_WITH_INSTALL_RPATH ON
                   INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")

    add_custom_target(
        rocprofiler-sdk-fork-test-runtime
        COMMAND ${CMAKE_COMMAND} -E make_directory
                "${_fork_test_root}/lib/rocprofiler-sdk"
        COMMAND ${CMAKE_COMMAND} -E rm -f
                "${_fork_test_root}/lib/librocprofiler-sdk-roctx.so"
        COMMAND
            ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE:rocprofiler-sdk-roctx-shared-library>
            "${_fork_test_root}/lib/librocprofiler-sdk-roctx.so"
        COMMAND
            ${CMAKE_COMMAND} -E rm -f
            "${_fork_test_root}/lib/rocprofiler-sdk/librocprofiler-sdk-tool-kokkosp.so"
        COMMAND
            ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:rocprofiler-sdk-tool-kokkosp>
            "${_fork_test_root}/lib/rocprofiler-sdk/librocprofiler-sdk-tool-kokkosp.so"
        COMMAND ${CMAKE_COMMAND} -E rm -f
                "${_fork_test_root}/lib/rocprofiler-sdk/librocprofv3-list-avail.so"
        COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:rocprofv3-list-avail>
                "${_fork_test_root}/lib/rocprofiler-sdk/librocprofv3-list-avail.so"
        DEPENDS rocprofiler-sdk-tool-fork-test rocprofiler-sdk-fork-test-shared
                rocprofiler-sdk-roctx-shared-library rocprofiler-sdk-tool-kokkosp
                rocprofv3-list-avail
        VERBATIM)
endif()
