## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

set(target test-torch-trace-collector)

add_executable(${target} test_torch_trace_collector.cpp roctx_range_intercept.cpp)

# Includes, torch libraries, and the DebugInfoKind flag are inherited from
# torch-trace-collector. Only the runtime search paths below are test-specific.
target_link_libraries(${target} PRIVATE torch-trace-collector gtest_main)

# torch's shared libraries cross-reference one another at load time.
target_link_options(
    ${target}
    PRIVATE LINKER:--allow-shlib-undefined LINKER:--wrap=roctxRangePushA
)

if(INSTALL_TESTS)
    install(
        TARGETS ${target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler-compute/tests
            COMPONENT tests
    )
endif()
