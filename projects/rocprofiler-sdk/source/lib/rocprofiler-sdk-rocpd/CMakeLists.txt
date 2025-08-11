#
# ROCm Profiling Data (rocpd) Library
#
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "rocpd")

add_library(rocprofiler-sdk-rocpd-shared-library SHARED)

foreach(_NAMESPACE rocprofiler-sdk rocprofiler-sdk-rocpd)
    foreach(_ALIAS library shared-library)
        add_library(${_NAMESPACE}::rocprofiler-sdk-rocpd-${_ALIAS} ALIAS
                    rocprofiler-sdk-rocpd-shared-library)
    endforeach()
endforeach()

target_sources(rocprofiler-sdk-rocpd-shared-library PRIVATE rocpd.cpp sql.cpp)

target_include_directories(
    rocprofiler-sdk-rocpd-shared-library
    INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/include>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(
    rocprofiler-sdk-rocpd-shared-library
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-build-flags
            rocprofiler-sdk::rocprofiler-sdk-memcheck
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-dl)

set_target_properties(
    rocprofiler-sdk-rocpd-shared-library
    PROPERTIES OUTPUT_NAME rocprofiler-sdk-rocpd
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION}
               SKIP_BUILD_RPATH OFF
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               DEFINE_SYMBOL rocpd_EXPORTS)

install(
    TARGETS rocprofiler-sdk-rocpd-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT rocpd
    EXPORT rocprofiler-sdk-rocpd-targets)
