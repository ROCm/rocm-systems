#
#
#
if(ROCPROFILER_BUILD_CODECOV)
    set(CMAKE_BUILD_TYPE "Coverage")
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(docs)
