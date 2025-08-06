#
# ROCTx Marker Library
#
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "roctx")

find_package(rocprofiler-register REQUIRED)

add_library(rocprofiler-sdk-roctx-shared-library SHARED)
add_library(rocprofiler-sdk::rocprofiler-sdk-roctx-shared-library ALIAS
            rocprofiler-sdk-roctx-shared-library)
add_library(rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library ALIAS
            rocprofiler-sdk-roctx-shared-library)

target_sources(rocprofiler-sdk-roctx-shared-library PRIVATE abi.cpp roctx.cpp)
target_include_directories(
    rocprofiler-sdk-roctx-shared-library
    INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/include>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(
    rocprofiler-sdk-roctx-shared-library
    PRIVATE rocprofiler-register::rocprofiler-register
            rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library)

set_target_properties(
    rocprofiler-sdk-roctx-shared-library
    PROPERTIES OUTPUT_NAME rocprofiler-sdk-roctx
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               SKIP_BUILD_RPATH OFF
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               DEFINE_SYMBOL roctx_EXPORTS)

install(
    TARGETS rocprofiler-sdk-roctx-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT roctx
    EXPORT rocprofiler-sdk-roctx-targets)
