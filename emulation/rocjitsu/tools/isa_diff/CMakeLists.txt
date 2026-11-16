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
    result_cache.cpp
    roles.cpp
    target.cpp
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

add_subdirectory(llvm)

add_executable(rj_isa_diff rj_isa_diff_main.cpp)
target_link_libraries(rj_isa_diff PRIVATE rj_isa_diff_core rj_isa_diff_llvm)
rj_configure_target(rj_isa_diff TOOL)

# One command for an LLVM change: the driver that tests the AMDGPU lit tests a patch
# updates (llvm/rj_isa_diff_pr.py) sits next to rj_isa_diff, which it runs.
configure_file(
    llvm/rj_isa_diff_pr.py
    ${CMAKE_CURRENT_BINARY_DIR}/rj_isa_diff_pr
    COPYONLY
)

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

    # LLVM plugin tests need an LLVM toolchain at test time; the core tests do not.
    find_program(
        RJ_ISA_DIFF_LLC
        llc
        HINTS $ENV{RJ_ISA_DIFF_LLVM_BIN} /opt/rocm/llvm/bin
        NO_DEFAULT_PATH
    )
    if(RJ_ISA_DIFF_LLC)
        get_filename_component(
            _rj_isa_diff_llvm_bin
            "${RJ_ISA_DIFF_LLC}"
            DIRECTORY
        )
        add_executable(
            rj_isa_diff_llvm_tests
            ${PROJECT_SOURCE_DIR}/tests/isa_diff/llvm_plugin_test.cpp
        )
        target_link_libraries(
            rj_isa_diff_llvm_tests
            PRIVATE rj_isa_diff_llvm GTest::gtest_main
        )
        target_compile_definitions(
            rj_isa_diff_llvm_tests
            PRIVATE RJ_ISA_DIFF_TEST_LLVM_BIN="${_rj_isa_diff_llvm_bin}"
        )
        rj_configure_target(rj_isa_diff_llvm_tests TEST)
        rj_add_gtest_tests(rj_isa_diff_llvm_tests TEST_PREFIX "isa_diff_llvm.")
    endif()

    # Drive the CLI end to end and validate every JSON it writes, plus the committed
    # samples, against docs/schema. Skipped (77) when python3-jsonschema is missing.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME isa_diff.schema_and_cli
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/docs/schema/validate_outputs.py
                --tool $<TARGET_FILE:rj_isa_diff>
                ${CMAKE_CURRENT_SOURCE_DIR}/docs/samples
        )
        set_tests_properties(
            isa_diff.schema_and_cli
            PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300
        )
        # rj_isa_diff_pr on a synthetic two-commit llvm-project history: NaN equivalence,
        # undefined values, a narrow return, poison triage and a real difference.
        if(RJ_ISA_DIFF_LLC)
            add_test(
                NAME isa_diff_llvm.pr_driver
                COMMAND
                    ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/llvm/test_rj_isa_diff_pr.py
                    --tool $<TARGET_FILE:rj_isa_diff> --driver
                    ${CMAKE_CURRENT_BINARY_DIR}/rj_isa_diff_pr --llvm-bin
                    ${_rj_isa_diff_llvm_bin}
            )
            set_tests_properties(isa_diff_llvm.pr_driver PROPERTIES TIMEOUT 600)
        endif()
    endif()
endif()
