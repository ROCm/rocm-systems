add_subdirectory(include)
if(AQLPROFILE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()