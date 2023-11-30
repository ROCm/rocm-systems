#
#
#
add_subdirectory(common)
add_subdirectory(rocprofiler-sdk)
add_subdirectory(rocprofiler-sdk-tool)
add_subdirectory(plugins)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
