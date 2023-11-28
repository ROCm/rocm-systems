#
# Tool library used by rocprofiler
#

add_library(rocprofiler-tool SHARED)
target_sources(rocprofiler-tool PRIVATE helper.hpp helper.cpp tool.cpp trace_buffer.hpp)
target_link_libraries(
    rocprofiler-tool
    PRIVATE rocprofiler::rocprofiler-shared-library
            rocprofiler::rocprofiler-hsa-runtime
            rocprofiler::rocprofiler-headers
            rocprofiler::rocprofiler-build-flags
            rocprofiler::rocprofiler-memcheck
            rocprofiler::rocprofiler-common-library
            rocprofiler::rocprofiler-cxx-filesystem
            atomic)

set_target_properties(
    rocprofiler-tool
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/rocprofiler
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               SKIP_BUILD_RPATH OFF
               BUILD_RPATH "\$ORIGIN:\$ORIGIN/.."
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")

install(
    TARGETS rocprofiler-tool
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
    COMPONENT tools
    EXPORT ${PROJECT_NAME}-library-targets)
