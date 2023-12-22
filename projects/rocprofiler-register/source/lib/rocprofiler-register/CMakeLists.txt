#
#   builds the rocprofiler-register library
#
add_library(rocprofiler-register SHARED)
add_library(rocprofiler-register::rocprofiler-register ALIAS rocprofiler-register)

add_subdirectory(details)

target_sources(rocprofiler-register PRIVATE rocprofiler_register.cpp)

if(ROCPROFILER_REGISTER_BUILD_TESTS)
    # make sure header is C-compatible
    target_sources(rocprofiler-register PRIVATE rocprofiler_register.c)
endif()

target_include_directories(
    rocprofiler-register PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/source
                                 ${PROJECT_BINARY_DIR}/source)

target_link_libraries(
    rocprofiler-register
    PUBLIC rocprofiler-register::headers
    PRIVATE fmt::fmt glog::glog rocprofiler-register::build-flags
            rocprofiler-register::memcheck rocprofiler-register::stdcxxfs
            rocprofiler-register::dl)

set_target_properties(
    rocprofiler-register
    PROPERTIES OUTPUT_NAME rocprofiler-register
               SOVERSION ${PROJECT_VERSION_MAJOR}
               VERSION ${PROJECT_VERSION})

install(
    TARGETS rocprofiler-register
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT core
    EXPORT ${PROJECT_NAME}-library-targets)
