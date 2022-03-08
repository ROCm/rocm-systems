add_subdirectory(omnitrace-avail)
add_subdirectory(omnitrace-critical-trace)
add_subdirectory(omnitrace)

if(OMNITRACE_BUILD_TESTING)
    add_subdirectory(tests)
endif()
