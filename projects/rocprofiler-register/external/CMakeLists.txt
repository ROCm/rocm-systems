set(BUILD_TESTING OFF)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME external)

if(ROCPROFILER_REGISTER_BUILD_GLOG)
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_register_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH glog
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        REPO_URL https://github.com/google/glog.git
        REPO_BRANCH "master")

    # May want to use GFLAGS in the future
    set(WITH_GFLAGS OFF)
    set(WITH_GTEST OFF)
    set(WITH_UNWIND "none")
    add_subdirectory(glog EXCLUDE_FROM_ALL)
else()
    find_package(glog REQUIRED GLOBAL)
endif()

if(ROCPROFILER_REGISTER_BUILD_FMT)
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_register_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH fmt
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        REPO_URL https://github.com/fmtlib/fmt.git
        REPO_BRANCH "master")

    set(FMT_TEST OFF)
    add_subdirectory(fmt EXCLUDE_FROM_ALL)
else()
    find_package(fmt REQUIRED GLOBAL)
endif()
