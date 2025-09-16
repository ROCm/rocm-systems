#
# rocprofiler-sdk attach Library
#

find_package(rocprofiler-register REQUIRED)

add_library(rocprofiler-sdk-attach-shared-library SHARED)
add_library(rocprofiler-sdk::rocprofiler-sdk-attach-shared-library ALIAS
            rocprofiler-sdk-attach-shared-library)
add_library(rocprofiler-sdk-attach::rocprofiler-sdk-attach-shared-library ALIAS
            rocprofiler-sdk-attach-shared-library)

target_sources(
    rocprofiler-sdk-attach-shared-library
    PRIVATE queue_registration.cpp code_object_registration.cpp attach.cpp table.cpp)

target_include_directories(
    rocprofiler-sdk-attach-shared-library
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/projects/rocprofiler-sdk/source/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/projects/rocprofiler-sdk/source/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(
    rocprofiler-sdk-attach-shared-library
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-register::rocprofiler-register-headers)

set_target_properties(
    rocprofiler-sdk-attach-shared-library
    PROPERTIES OUTPUT_NAME rocprofiler-sdk-attach
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               SKIP_BUILD_RPATH OFF
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               DEFINE_SYMBOL attach_EXPORTS)

install(
    TARGETS rocprofiler-sdk-attach-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT core
    EXPORT rocprofiler-sdk-attach-targets)
