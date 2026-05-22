# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Race detection library: RaceDetector, WaveRaceState, and IntervalSet.
# Tracks in-flight memory events and detects synchronization hazards
# (missing s_waitcnt, s_barrier) at the register and LDS byte level.

# Standalone build support: when built outside the rocjitsu tree,
# this file acts as the top-level CMakeLists.txt.
if(NOT PROJECT_NAME)
    cmake_minimum_required(VERSION 3.22)
    project(race_detector_standalone CXX)
    set(CMAKE_CXX_STANDARD 20)
    include(CTest)

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_library(race_detector OBJECT src/RaceDetector.cpp src/WaveRaceState.cpp)
target_include_directories(race_detector PUBLIC include)
target_compile_features(race_detector PUBLIC cxx_std_20)

# Standalone tests for the race detection logic.
# These test the detection logic directly and do not depend on rocjitsu.
if(BUILD_TESTING)
    add_executable(
        race_detector_tests
        tests/race_detector_tests.cpp
        tests/interval_set_tests.cpp
    )
    target_link_libraries(
        race_detector_tests
        PRIVATE race_detector GTest::gtest_main
    )
    target_include_directories(race_detector_tests PRIVATE include)

    include(GoogleTest)
    gtest_discover_tests(race_detector_tests)
endif()
