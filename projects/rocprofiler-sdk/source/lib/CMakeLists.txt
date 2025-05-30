#
#
#
rocprofiler_activate_clang_tidy()

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "core")
add_subdirectory(common)
add_subdirectory(output)
add_subdirectory(rocprofiler-sdk)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "roctx")
add_subdirectory(rocprofiler-sdk-roctx)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "rocpd")
add_subdirectory(rocprofiler-sdk-rocpd)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "tools")
add_subdirectory(att-tool)
add_subdirectory(rocprofiler-sdk-tool)

add_subdirectory(python)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
