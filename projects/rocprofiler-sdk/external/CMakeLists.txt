#
#
#   External dependencies
#
#
set(BUILD_TESTING OFF)

if(ROCPROFILER_BUILD_TESTS)
    if(ROCPROFILER_BUILD_GTEST)
        set(INSTALL_GTEST
            OFF
            CACHE BOOL "")
        set(BUILD_GMOCK
            OFF
            CACHE BOOL "")

        add_subdirectory(googletest EXCLUDE_FROM_ALL)

        if(NOT TARGET GTest::gtest)
            message(FATAL_ERROR "missing GTest::gtest")
        endif()

        target_link_libraries(rocprofiler-gtest INTERFACE GTest::gtest)
        target_include_directories(
            rocprofiler-gtest SYSTEM
            INTERFACE ${CMAKE_CURRENT_LIST_DIR}/googletest/googletest/include)

        mark_as_advanced(INSTALL_GTEST)
        mark_as_advanced(BUILD_GMOCK)
        mark_as_advanced(GTEST_HAS_ABSL)
    else()
        find_package(GTest REQUIRED)
        target_link_libraries(rocprofiler-gtest INTERFACE GTest::gtest)
    endif()
endif()

if(ROCPROFILER_BUILD_GLOG)
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/glog
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/google/glog.git
        REPO_BRANCH "master")

    # May want to use GFLAGS in the future
    set(WITH_GFLAGS OFF)
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory(glog EXCLUDE_FROM_ALL)

    target_link_libraries(rocprofiler-glog INTERFACE $<BUILD_INTERFACE:glog::glog>)
    target_include_directories(
        rocprofiler-glog SYSTEM
        INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/glog>
                  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/glog/src>)
else()
    find_package(glog REQUIRED)
    target_link_libraries(rocprofiler-glog INTERFACE glog::glog)
endif()

if(ROCPROFILER_BUILD_FMT)
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_checkout_git_submodule(
        RECURSIVE
        RELATIVE_PATH external/fmt
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/fmtlib/fmt.git
        REPO_BRANCH "master")

    set(FMT_TEST OFF)
    add_subdirectory(fmt EXCLUDE_FROM_ALL)

    target_link_libraries(rocprofiler-fmt INTERFACE $<BUILD_INTERFACE:fmt::fmt>)
    target_include_directories(
        rocprofiler-fmt SYSTEM
        INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/fmt/include>)
else()
    find_package(fmt REQUIRED)
    target_link_libraries(rocprofiler-fmt INTERFACE fmt::fmt)
endif()
