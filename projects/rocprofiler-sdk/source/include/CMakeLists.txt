#
#
#
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "development")

add_subdirectory(rocprofiler-sdk)
add_subdirectory(rocprofiler-sdk-rocattach)

if(ROCPROFILER_BUILD_COLLECTOR)
    add_subdirectory(rocprofiler-sdk-collector)
endif()
add_subdirectory(rocprofiler-sdk-roctx)
add_subdirectory(rocprofiler-sdk-rocpd)
