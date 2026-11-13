#
#   Out-of-process collector library (librocprofiler-sdk-collector)
#
rocprofiler_activate_clang_tidy()

# The implementation, linked into the shared library and the unit tests.
add_library(rocprofiler-sdk-collector-static-library STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-collector-static-library ALIAS
            rocprofiler-sdk-collector-static-library)

target_sources(rocprofiler-sdk-collector-static-library
               PRIVATE clock.cpp daemon.cpp join.cpp launch.cpp session.cpp)
add_subdirectory(ctf)

target_include_directories(
    rocprofiler-sdk-collector-static-library
    PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/include>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/include>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

target_link_libraries(
    rocprofiler-sdk-collector-static-library
    PUBLIC rocprofiler-sdk::rocprofiler-sdk-build-flags
           rocprofiler-sdk::rocprofiler-sdk-memcheck
           rocprofiler-sdk::rocprofiler-sdk-common-library
           rocprofiler-sdk::rocprofiler-sdk-lttng-ctl)

set_target_properties(
    rocprofiler-sdk-collector-static-library
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON)

add_library(rocprofiler-sdk-collector-shared-library SHARED)

foreach(_NAMESPACE rocprofiler-sdk rocprofiler-sdk-collector)
    foreach(_ALIAS library shared-library)
        add_library(${_NAMESPACE}::rocprofiler-sdk-collector-${_ALIAS} ALIAS
                    rocprofiler-sdk-collector-shared-library)
    endforeach()
endforeach()

target_sources(rocprofiler-sdk-collector-shared-library PRIVATE collector.cpp)

target_include_directories(
    rocprofiler-sdk-collector-shared-library
    PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/include>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>)

target_link_libraries(rocprofiler-sdk-collector-shared-library
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk-collector-static-library)

set_target_properties(
    rocprofiler-sdk-collector-shared-library
    PROPERTIES OUTPUT_NAME rocprofiler-sdk-collector
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
               SKIP_BUILD_RPATH OFF
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               DEFINE_SYMBOL ROCPROFILER_COLLECTOR_EXPORTS)

install(
    TARGETS rocprofiler-sdk-collector-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT core)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
