rocprofiler_activate_clang_tidy()

add_library(rocprofv3-list-avail SHARED)
target_sources(rocprofv3-list-avail PRIVATE rocprofv3_avail.cpp)

target_link_libraries(
    rocprofv3-list-avail
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-shared-library
            rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-cereal)

set_target_properties(
    rocprofv3-list-avail
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBEXECDIR}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               BUILD_RPATH "\$ORIGIN:\$ORIGIN/.."
               INSTALL_RPATH "\$ORIGIN:\$ORIGIN/..")

install(
    TARGETS rocprofv3-list-avail
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}
    COMPONENT tools
    EXPORT rocprofiler-sdk-tool-targets)
