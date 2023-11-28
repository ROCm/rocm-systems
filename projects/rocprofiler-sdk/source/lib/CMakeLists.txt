#
#
#
add_subdirectory(common)
add_subdirectory(rocprofiler)
add_subdirectory(rocprofiler-tool)
add_subdirectory(plugins)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
