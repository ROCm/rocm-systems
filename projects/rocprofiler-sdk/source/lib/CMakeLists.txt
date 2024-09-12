#
#
#
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "core")
add_subdirectory(common)
add_subdirectory(rocprofiler-sdk)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "tools")
add_subdirectory(rocprofiler-sdk-codeobj)
add_subdirectory(rocprofiler-sdk-roctx)
add_subdirectory(rocprofiler-sdk-tool)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
