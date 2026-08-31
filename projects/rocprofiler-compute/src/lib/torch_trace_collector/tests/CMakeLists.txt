## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

set(target test-torch-trace-collector)

add_executable(${target} test_torch_trace_collector.cpp roctx_range_intercept.cpp)

target_link_libraries(${target} PRIVATE torch-trace-collector gtest_main nlohmann_json)
target_compile_definitions(
    ${target}
    PRIVATE
        MARKER_ARGS_CODEC_PATH="${PROJECT_SOURCE_DIR}/tests/marker_args_codec.json"
)
target_link_options(
    ${target}
    PRIVATE LINKER:--allow-shlib-undefined LINKER:--wrap=roctxRangePushA
)

if(INSTALL_TESTS)
    install(
        TARGETS ${target}
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${CMAKE_PROJECT_NAME}/tests
            COMPONENT tests
    )
endif()
