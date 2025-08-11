#
#
#
if(ROCPROFILER_BUILD_CODECOV)
    set(CMAKE_BUILD_TYPE "Coverage")
endif()

#
# by default, activate clang-tidy on all code in the source folder. unittest subfolders
# can add `rocprofiler_deactivate_clang_tidy()` to their CMakeLists.txt to disable
# clang-tidy linting
#
rocprofiler_activate_clang_tidy()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(libexec)
add_subdirectory(bin)
add_subdirectory(docs)
add_subdirectory(share)
