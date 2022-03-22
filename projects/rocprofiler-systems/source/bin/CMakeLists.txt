add_subdirectory(omnitrace-avail)
add_subdirectory(omnitrace-critical-trace)
add_subdirectory(omnitrace)

if(OMNITRACE_BUILD_TESTING OR "$ENV{OMNITRACE_CI}" MATCHES "[1-9]+|ON|on|y|yes")
    add_subdirectory(tests)
endif()
