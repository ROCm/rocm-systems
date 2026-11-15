# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# rj_isa_diff: incremental ISA-to-ISA differential testing of changed SALU regions.
# See README.md, docs/plan.md and docs/implementation_map.md.

add_library(
    rj_isa_diff_core
    STATIC
    artifact.cpp
    boundary.cpp
    case_io.cpp
    contract.cpp
    diff.cpp
    differential.cpp
    effects.cpp
    input_policy.cpp
    json.cpp
    kernel_model.cpp
    location.cpp
    machine_state.cpp
    mutation.cpp
    region.cpp
    region_planner.cpp
    region_runner.cpp
    roles.cpp
)
target_include_directories(rj_isa_diff_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# The execution-capable ISA registry (via RJ_TOOL_OBJECT_LIBS) is required: the
# model-only registry leaves Instruction::execute null.
target_link_libraries(
    rj_isa_diff_core
    PUBLIC ${RJ_TOOL_OBJECT_LIBS} util flatbuffers Threads::Threads
)
target_compile_definitions(
    rj_isa_diff_core
    PRIVATE
        RJ_ISA_DIFF_RJ_REVISION="${ROCJITSU_GIT_REVISION}"
        RJ_ISA_DIFF_RJ_VERSION="${PROJECT_VERSION}"
)
rj_configure_target(rj_isa_diff_core PRIVATE_INCLUDES WARNINGS)

add_executable(rj_isa_diff rj_isa_diff_main.cpp)
target_link_libraries(rj_isa_diff PRIVATE rj_isa_diff_core)
rj_configure_target(rj_isa_diff TOOL)

if(BUILD_TESTING)
    include(GoogleTest)
    add_executable(
        rj_isa_diff_tests
        ${PROJECT_SOURCE_DIR}/tests/isa_diff/isa_diff_test.cpp
    )
    target_include_directories(
        rj_isa_diff_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/tests/isa_diff
    )
    target_link_libraries(
        rj_isa_diff_tests
        PRIVATE rj_isa_diff_core GTest::gtest_main
    )
    rj_configure_target(rj_isa_diff_tests TEST)
    rj_add_gtest_tests(rj_isa_diff_tests TEST_PREFIX "isa_diff.")
endif()
