#
# rocprofiler-sdk Proxy Queue Library
#
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "queue")

add_library(rocprofiler-sdk-queue-shared-library SHARED)
add_library(rocprofiler-sdk::rocprofiler-sdk-queue-shared-library ALIAS
            rocprofiler-sdk-queue-shared-library)
add_library(rocprofiler-sdk-queue::rocprofiler-sdk-queue-shared-library ALIAS
            rocprofiler-sdk-queue-shared-library)

target_sources(rocprofiler-sdk-queue-shared-library
    PRIVATE queue_registration.cpp
            queue_registration.hpp
            queue_registration_controller.cpp
            queue_registration_controller.hpp)

target_include_directories(
    rocprofiler-sdk-queue-shared-library
    INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/projects/rocprofiler-sdk/source/include>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/projects/rocprofiler-sdk/source/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(
    rocprofiler-sdk-queue-shared-library
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library)

set_target_properties(
    rocprofiler-sdk-queue-shared-library
    PROPERTIES OUTPUT_NAME rocprofiler-sdk-queue
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               SKIP_BUILD_RPATH OFF
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               DEFINE_SYMBOL queue_EXPORTS)

install(
    TARGETS rocprofiler-sdk-queue-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT queue
    EXPORT rocprofiler-sdk-queue-targets)
