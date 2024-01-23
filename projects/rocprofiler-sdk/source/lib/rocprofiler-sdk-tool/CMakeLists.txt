#
# Tool library used by rocprofiler
#

add_library(rocprofiler-sdk-tool SHARED)
target_sources(
    rocprofiler-sdk-tool PRIVATE config.hpp config.cpp csv.hpp helper.hpp helper.cpp
                                 output_file.hpp output_file.cpp tool.cpp)

add_subdirectory(plugins)

target_link_libraries(
    rocprofiler-sdk-tool
    PRIVATE rocprofiler::rocprofiler-shared-library
            rocprofiler::rocprofiler-hsa-runtime
            rocprofiler::rocprofiler-headers
            rocprofiler::rocprofiler-build-flags
            rocprofiler::rocprofiler-memcheck
            rocprofiler::rocprofiler-common-library
            rocprofiler::rocprofiler-cxx-filesystem
            atomic)

set_target_properties(
    rocprofiler-sdk-tool
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               SKIP_BUILD_RPATH OFF
               BUILD_RPATH "\$ORIGIN:\$ORIGIN/.."
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")

install(
    TARGETS rocprofiler-sdk-tool
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocprofiler-sdk
    COMPONENT tools
    EXPORT rocprofiler-sdk-tool-targets)
