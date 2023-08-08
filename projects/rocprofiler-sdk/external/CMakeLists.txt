#
#
#   External dependencies
#
#

if(ROCPROFILER_BUILD_TESTS)
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

    target_link_libraries(rocprofiler-googletest INTERFACE GTest::gtest)
    target_include_directories(
        rocprofiler-googletest SYSTEM
        INTERFACE ${CMAKE_CURRENT_LSIT_DIR}/googletest/googletest/include)

    mark_as_advanced(INSTALL_GTEST)
    mark_as_advanced(BUILD_GMOCK)
    mark_as_advanced(GTEST_HAS_ABSL)
endif()
