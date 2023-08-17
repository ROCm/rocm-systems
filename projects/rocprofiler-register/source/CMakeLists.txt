#
#
#
rocprofiler_register_activate_clang_tidy()

if(ROCPROFILER_REGISTER_BUILD_CODECOV)
    set(CMAKE_BUILD_TYPE "Coverage")
endif()

add_subdirectory(include)
add_subdirectory(lib)
