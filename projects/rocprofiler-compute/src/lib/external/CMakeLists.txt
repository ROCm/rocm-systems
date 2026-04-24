include(${CMAKE_CURRENT_LIST_DIR}/GitSubmodule.cmake)

# TheRock to introduces "hermetic" builds where submodules are checked out and then network access is disabled.
# This means that we can't rely on fetching submodules during build, ex via FetchContent.
# At the same time we can't rely on CI to always calling `submodule update`, therefore hybrid solution:
# dependencies are added as submodules but they are checked out on cmake as well.

if(ENABLE_TESTS)
    rocprofiler_compute_checkout_git_submodule(
        RELATIVE_PATH googletest
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        TEST_FILE CMakeLists.txt
        REPO_URL https://github.com/google/googletest.git
        REPO_BRANCH "v1.17.0"
    )

    set(BUILD_GMOCK ON CACHE BOOL "Build GMock" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Disable GTest installation" FORCE)
    add_subdirectory(googletest)
endif()

rocprofiler_compute_checkout_git_submodule(
    RELATIVE_PATH fmt
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    TEST_FILE CMakeLists.txt
    REPO_URL https://github.com/fmtlib/fmt.git
    REPO_BRANCH "12.1.0"
)

set(FMT_INSTALL OFF CACHE BOOL "Disable fmt installation" FORCE)
set(FMT_TEST OFF CACHE BOOL "Disable fmt tests" FORCE)
add_subdirectory(fmt)
