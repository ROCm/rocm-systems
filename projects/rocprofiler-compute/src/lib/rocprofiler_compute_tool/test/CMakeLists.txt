## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target test_rocprofiler_compute)

add_executable(${target}
    test_rocprofiler_compute_tool.h
    test_rocprofiler_compute_tool.cpp
    test_sdk_callbacks.h
    test_sdk_callbacks.cpp
    mocks.h
    mocks.cpp
    main.cpp
    test_sdk_callbacks.h
)


target_link_libraries(${target} PRIVATE
    rocprofiler-compute-tool
    gtest_main
    gmock
    gsl_assert
    fmt::fmt
)
