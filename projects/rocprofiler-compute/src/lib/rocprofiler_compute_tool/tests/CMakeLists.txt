## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target test-rocprofiler-compute-tool)

add_executable(
    ${target}
    test_rocprofiler_compute_tool.h
    test_rocprofiler_compute_tool.cpp
    test_pc_sampling_feature.h
    test_pc_sampling_feature.cpp
    test_sdk_callbacks.h
    test_sdk_callbacks.cpp
    test_environ_cache.h
    test_environ_cache.cpp
    test_counters_writer.h
    test_counters_writer.cpp
    mocks.h
    mocks.cpp
    main.cpp
)

target_link_libraries(
    ${target}
    PRIVATE
        rocprofiler-compute-tool
        gtest_main
        gmock
        gsl_assert
        fmt::fmt
        asan_build_dependencies
)

# TheRock's post-hook overwrites INSTALL_RPATH with its own computed value, so
# this only takes effect in standalone TEST_FROM_INSTALL builds, where nothing
# else stamps an RPATH and librocprofiler-compute-tool.so is otherwise unfound.
set_target_properties(
    ${target}
    PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../../${CMAKE_INSTALL_LIBDIR}/${CMAKE_PROJECT_NAME}"
)

if(INSTALL_TESTS)
    install(
        TARGETS ${target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${CMAKE_PROJECT_NAME}/tests
            COMPONENT tests
    )
endif()
