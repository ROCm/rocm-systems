#
#
#
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "core")
add_subdirectory(common)
add_subdirectory(rocprofiler-sdk)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "tools")
add_subdirectory(rocprofiler-sdk-tool)
add_subdirectory(rocprofiler-sdk-roctx)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
