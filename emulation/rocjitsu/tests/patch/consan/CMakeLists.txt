# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_executable(
    consan_input_test
    input_test.cpp
    prepare_test.cpp
    invariants_test.cpp
)
target_link_libraries(
    consan_input_test
    PRIVATE
        rocjitsu_consan_compiler
        rocjitsu_amdgpu_isa_model_registry
        GTest::gtest_main
)
rj_configure_target(consan_input_test PRIVATE_INCLUDES WARNINGS)

if(AMDCXX AND EXISTS "${ROCM_PATH}/lib/llvm/bin/ld.lld")
    set(_fixture_dir ${CMAKE_CURRENT_BINARY_DIR}/fixtures)
    file(MAKE_DIRECTORY ${_fixture_dir})
    set(_fixtures)
    set(_names
        race
        disjoint
        read_read
        collision
        single_lane
        alias
        barrier
        immediate
        wide
        four_ranges
        exhausted
        empty
        branched
    )
    set(_case 0)
    foreach(_name IN LISTS _names)
        math(EXPR _case "${_case} + 1")
        set(_object ${_fixture_dir}/${_name}.o)
        set(_image ${_fixture_dir}/${_name}.hsaco)
        add_custom_command(
            OUTPUT ${_object}
            COMMAND
                ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp
                -target amdgcn-amd-amdhsa -mcpu=gfx1100 -mcode-object-version=5
                -DCONSAN_CASE=${_case} -c
                ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/native_lds.s -o ${_object}
            DEPENDS fixtures/native_lds.s ${AMDCXX}
            VERBATIM
        )
        add_custom_command(
            OUTPUT ${_image}
            COMMAND
                ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared ${_object} -o ${_image}
            DEPENDS ${_object}
            VERBATIM
        )
        list(APPEND _fixtures ${_image})
    endforeach()
    add_custom_command(
        OUTPUT ${_fixture_dir}/renamed.o
        COMMAND
            ${CMAKE_CXX_COMPILER_LAUNCHER} ${AMDCXX} -x assembler-with-cpp
            -target amdgcn-amd-amdhsa -mcpu=gfx1100 -mcode-object-version=5
            -DCONSAN_CASE=1 -DCONSAN_KERNEL=renamed_input -c
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/native_lds.s -o
            ${_fixture_dir}/renamed.o
        DEPENDS fixtures/native_lds.s ${AMDCXX}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${_fixture_dir}/renamed.hsaco
        COMMAND
            ${ROCM_PATH}/lib/llvm/bin/ld.lld -shared ${_fixture_dir}/renamed.o
            -o ${_fixture_dir}/renamed.hsaco
        DEPENDS ${_fixture_dir}/renamed.o
        VERBATIM
    )
    list(APPEND _fixtures ${_fixture_dir}/renamed.hsaco)
    include(target_fixtures.cmake)
    target_sources(consan_input_test PRIVATE target_test.cpp)
    add_custom_target(consan_original_fixtures DEPENDS ${_fixtures})
    add_dependencies(consan_input_test consan_original_fixtures)
    add_dependencies(consan_input_test consan_target_fixtures)
    target_compile_definitions(
        consan_input_test
        PRIVATE CONSAN_FIXTURE_DIR="${_fixture_dir}"
    )
else()
    message(
        STATUS
        "ConSan original HSACO fixtures unavailable (need AMD LLVM assembler/linker)"
    )
endif()
rj_add_gtest_tests(consan_input_test)

set(RJ_CONSAN_TEST_ARTIFACT_ROOT
    "${PROJECT_BINARY_DIR}/Testing/ConSan"
    CACHE PATH
    "Directory for preserved ConSan integration evidence"
)
if(TARGET consan_original_fixtures)
    add_test(
        NAME ConSanPrepareCli.RejectNonExecutableOriginalBeforePublication
        COMMAND
            ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/prepare_cli_test.py
            $<TARGET_FILE:rj_consan_prepare> ${_fixture_dir}/race.hsaco
            ${RJ_CONSAN_TEST_ARTIFACT_ROOT}
    )
    add_test(
        NAME ConSanPrepareCli.RejectBoundedInputAndFilesystemFailures
        COMMAND
            ${Python3_EXECUTABLE} -B
            ${CMAKE_CURRENT_SOURCE_DIR}/prepare_cli_io_test.py
            $<TARGET_FILE:rj_consan_prepare> ${_fixture_dir}/race.hsaco
            ${RJ_CONSAN_TEST_ARTIFACT_ROOT}
    )
endif()

add_executable(consan_bootstrap_sim_test bootstrap_sim_test.cpp)
target_include_directories(
    consan_bootstrap_sim_test
    PRIVATE ${PROJECT_SOURCE_DIR}/tests
)
target_link_libraries(
    consan_bootstrap_sim_test
    PRIVATE
        rocjitsu_consan_compiler
        ${RJ_OBJECT_LIBS}
        flatbuffers
        GTest::gtest_main
)
rj_configure_target(consan_bootstrap_sim_test PRIVATE_INCLUDES GENERATED WARNINGS SCHEMAS)
if(TARGET consan_original_fixtures)
    add_dependencies(consan_bootstrap_sim_test consan_original_fixtures)
    add_dependencies(consan_bootstrap_sim_test consan_target_fixtures)
    target_compile_definitions(
        consan_bootstrap_sim_test
        PRIVATE CONSAN_FIXTURE_DIR="${_fixture_dir}"
    )
endif()
rj_add_gtest_tests(consan_bootstrap_sim_test)

if(TARGET consan_original_fixtures)
    set(_artifact ${CMAKE_CURRENT_BINARY_DIR}/prepared/race)
    add_custom_command(
        OUTPUT ${_artifact}/facts.bin
        BYPRODUCTS ${_artifact}/prepared.hsaco ${_artifact}/ready
        COMMAND
            ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/prepared
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${_artifact}
        COMMAND
            $<TARGET_FILE:rj_consan_prepare> ${_fixture_dir}/race.hsaco gfx1100
            lds_producer_consumer ${_artifact}
        DEPENDS rj_consan_prepare ${_fixture_dir}/race.hsaco
        VERBATIM
    )
    add_custom_target(consan_artifact_fixture DEPENDS ${_artifact}/facts.bin)
    add_executable(
        consan_image_test
        ${PROJECT_SOURCE_DIR}/tests/code/consan/artifact_test.cpp
    )
    target_link_libraries(
        consan_image_test
        PRIVATE rocjitsu_consan_image GTest::gtest_main
    )
    rj_configure_target(consan_image_test PRIVATE_INCLUDES WARNINGS)
    add_dependencies(consan_image_test consan_artifact_fixture)
    target_compile_definitions(
        consan_image_test
        PRIVATE
            CONSAN_ORIGINAL_FIXTURE="${_fixture_dir}/race.hsaco"
            CONSAN_ARTIFACT_FIXTURE="${_artifact}"
    )
    rj_add_gtest_tests(consan_image_test)
endif()
