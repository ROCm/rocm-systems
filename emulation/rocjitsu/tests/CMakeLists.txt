# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

# Device kernels (compiled from HIP sources for simulator testing).
add_subdirectory(kernels)

# Common object libraries linked by all test/tool executables.
set(RJ_OBJECT_LIBS
    rocjitsu_analysis
    rocjitsu_code
    rocjitsu_isa
    rocjitsu_isa_cdna1
    rocjitsu_isa_cdna2
    rocjitsu_isa_cdna3
    rocjitsu_isa_cdna4
    rocjitsu_isa_rdna1
    rocjitsu_isa_rdna2
    rocjitsu_isa_rdna3
    rocjitsu_isa_rdna3_5
    rocjitsu_isa_rdna4
    rocjitsu_isa_gfx1250
    rocjitsu_isa_risc_v
    simdojo
    rocjitsu_kmd
    rocjitsu_vm
    rocjitsu_vm_amdgpu
    rocjitsu_plugin_logging
    rocjitsu_plugin_race_detector
    rocjitsu_vm_risc_v
    rocjitsu_config
    rocjitsu_dbt_config
)

if(RJ_INSTALL_TESTS)
    set(RJ_INSTALL_TEST_DIR "${CMAKE_INSTALL_DATADIR}/rocjitsu/tests")
    set(RJ_INSTALL_TEST_BIN_DIR "${RJ_INSTALL_TEST_DIR}/bin")
    set(RJ_INSTALL_CTEST_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/install/CTestTestfile.cmake"
    )
    set(RJ_INSTALL_GTEST_GENERATOR
        "${PROJECT_SOURCE_DIR}/cmake/rj_generate_installed_gtest_ctest.cmake"
    )
    if(IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}")
        set(_rj_installed_bindir "${CMAKE_INSTALL_BINDIR}")
    else()
        set(_rj_installed_bindir "../../../${CMAKE_INSTALL_BINDIR}")
    endif()
    if(IS_ABSOLUTE "${CMAKE_INSTALL_DATADIR}")
        set(_rj_installed_datadir "${CMAKE_INSTALL_DATADIR}")
    else()
        set(_rj_installed_datadir "../..")
    endif()
    if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(_rj_installed_libdir "${CMAKE_INSTALL_LIBDIR}")
    else()
        set(_rj_installed_libdir "../../../${CMAKE_INSTALL_LIBDIR}")
    endif()
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/install")
    file(WRITE "${RJ_INSTALL_CTEST_FILE}" "# Installed rocjitsu CTest tests\n")
    file(
        APPEND "${RJ_INSTALL_CTEST_FILE}"
        "set(RJ_INSTALLED_BINDIR \"${_rj_installed_bindir}\")\n"
        "set(RJ_INSTALLED_CONFIG_DIR \"${_rj_installed_datadir}/rocjitsu/configs\")\n\n"
        "set(RJ_INSTALLED_KERNEL_DIR \"kernels\")\n\n"
        "set(RJ_INSTALLED_LIBDIR \"${_rj_installed_libdir}\")\n\n"
        "include(\"rocjitsu_tests.cmake\")\n\n"
    )

    # Install a compiled test/tool target into the installed CTest bin directory.
    # The installed CTest file references these binaries with paths relative to
    # ${RJ_INSTALL_TEST_DIR}, e.g. bin/rocjitsu_tests.
    function(rj_install_test_target target)
        install(
            TARGETS ${target}
            RUNTIME DESTINATION ${RJ_INSTALL_TEST_BIN_DIR}
        )
    endfunction()

    # Install a non-CMake executable produced by a custom command, such as a HIP
    # test binary compiled directly with hipcc.
    function(rj_install_test_program path)
        install(PROGRAMS ${path} DESTINATION ${RJ_INSTALL_TEST_BIN_DIR})
    endfunction()

    # Append one installed CTest entry to ${RJ_INSTALL_CTEST_FILE}. This mirrors
    # the subset of add_test()/set_tests_properties() used by rj_add_test, but
    # writes plain CMake code because the file is consumed later by
    # `ctest --test-dir <installed tests dir>`.
    function(rj_install_ctest name)
        set(oneValueArgs TIMEOUT SKIP_REGULAR_EXPRESSION)
        set(multiValueArgs COMMAND ENVIRONMENT LABELS)
        cmake_parse_arguments(
            ARG
            ""
            "${oneValueArgs}"
            "${multiValueArgs}"
            ${ARGN}
        )
        if(NOT ARG_COMMAND)
            message(FATAL_ERROR "rj_install_ctest(${name}) requires COMMAND")
        endif()

        set(_content "add_test(\"${name}\"")
        foreach(_arg IN LISTS ARG_COMMAND)
            string(APPEND _content " \"${_arg}\"")
        endforeach()
        string(APPEND _content ")\n")

        set(_props)
        if(ARG_TIMEOUT)
            string(APPEND _props " \"TIMEOUT\" \"${ARG_TIMEOUT}\"")
        endif()
        if(ARG_SKIP_REGULAR_EXPRESSION)
            string(
                APPEND _props
                " \"SKIP_REGULAR_EXPRESSION\" [=[${ARG_SKIP_REGULAR_EXPRESSION}]=]"
            )
        endif()
        if(ARG_LABELS)
            list(JOIN ARG_LABELS ";" _labels)
            string(APPEND _props " \"LABELS\" \"${_labels}\"")
        endif()
        if(ARG_ENVIRONMENT)
            list(JOIN ARG_ENVIRONMENT ";" _env)
            string(APPEND _props " \"ENVIRONMENT\" \"${_env}\"")
        endif()
        if(_props)
            string(
                APPEND _content
                "set_tests_properties(\"${name}\" PROPERTIES"
            )
            string(APPEND _content "${_props})\n")
        endif()

        file(APPEND "${RJ_INSTALL_CTEST_FILE}" "${_content}\n")
    endfunction()

    # Generate and install fine-grained CTest entries for every concrete
    # GoogleTest case in a target. The target executable must already exist, so
    # this is a build-time custom command that runs --gtest_list_tests and writes
    # an installed CTest fragment like rocjitsu_tests.cmake.
    function(rj_install_gtest_cases target)
        set(_fragment "${CMAKE_CURRENT_BINARY_DIR}/install/${target}.cmake")
        add_custom_command(
            OUTPUT "${_fragment}"
            COMMAND
                ${CMAKE_COMMAND} "-DTEST_EXECUTABLE=$<TARGET_FILE:${target}>"
                "-DINSTALLED_EXECUTABLE=bin/${target}"
                "-DOUTPUT_FILE=${_fragment}" -P "${RJ_INSTALL_GTEST_GENERATOR}"
            DEPENDS ${target} "${RJ_INSTALL_GTEST_GENERATOR}"
            COMMENT "Generating installed CTest entries for ${target}"
            VERBATIM
        )
        add_custom_target(${target}_installed_ctest ALL DEPENDS "${_fragment}")
        install(FILES "${_fragment}" DESTINATION ${RJ_INSTALL_TEST_DIR})
    endfunction()

    install(FILES "${RJ_INSTALL_CTEST_FILE}" DESTINATION ${RJ_INSTALL_TEST_DIR})
endif()

include(GoogleTest)

set(_rj_installed_test_bin_dir "${CMAKE_INSTALL_DATADIR}/rocjitsu/tests/bin")
function(_rj_path_from_installed_test_bin out install_dir file_name)
    if(IS_ABSOLUTE "${install_dir}")
        set(_path "${install_dir}/${file_name}")
    elseif(IS_ABSOLUTE "${_rj_installed_test_bin_dir}")
        set(_path "${CMAKE_INSTALL_PREFIX}/${install_dir}/${file_name}")
    else()
        file(
            RELATIVE_PATH
            _rel_dir
            "/${_rj_installed_test_bin_dir}"
            "/${install_dir}"
        )
        set(_path "${_rel_dir}/${file_name}")
    endif()
    set(${out} "${_path}" PARENT_SCOPE)
endfunction()

function(rj_test_runtime_dir_env out name)
    string(REGEX REPLACE "[^A-Za-z0-9_.-]" "_" _runtime_name "${name}")
    set(${out}
        "ROCJITSU_RUNTIME_DIR=${CMAKE_CURRENT_BINARY_DIR}/runtime/${_runtime_name}"
        PARENT_SCOPE
    )
endfunction()

function(rj_installed_test_runtime_dir_env out name)
    string(REGEX REPLACE "[^A-Za-z0-9_.-]" "_" _runtime_name "${name}")
    set(${out} "ROCJITSU_RUNTIME_DIR=runtime/${_runtime_name}" PARENT_SCOPE)
endfunction()

function(rj_environment_has_runtime_dir out)
    set(_has_runtime_dir FALSE)
    foreach(_entry IN LISTS ARGN)
        if(_entry MATCHES "^ROCJITSU_RUNTIME_DIR=")
            set(_has_runtime_dir TRUE)
        endif()
    endforeach()
    set(${out} "${_has_runtime_dir}" PARENT_SCOPE)
endfunction()

function(rj_set_default_runtime_dir name)
    get_property(_env TEST "${name}" PROPERTY ENVIRONMENT)
    rj_environment_has_runtime_dir(_has_runtime_dir ${_env})
    if(NOT _has_runtime_dir)
        rj_test_runtime_dir_env(_runtime_dir_env "${name}")
        list(APPEND _env "${_runtime_dir_env}")
        set_tests_properties("${name}" PROPERTIES ENVIRONMENT "${_env}")
    endif()
endfunction()

# Register a test for both the build tree and, when RJ_INSTALL_TESTS is ON, the
# installed CTest tree. COMMAND is the normal build-tree command passed to
# add_test(). INSTALL_COMMAND is optional and should use paths relative to the
# installed test directory, e.g. bin/my_test or ${RJ_INSTALLED_BINDIR}/rocjitsu.
# Shared properties TIMEOUT, SKIP_REGULAR_EXPRESSION, and LABELS are applied to
# both registrations. ENVIRONMENT applies only to the build-tree test;
# INSTALL_ENVIRONMENT adds to the default installed test environment, which
# points installed tests at packaged configs, kernels, binaries, and runtimes.
function(rj_add_test)
    set(oneValueArgs NAME WORKING_DIRECTORY TIMEOUT SKIP_REGULAR_EXPRESSION)
    set(multiValueArgs
        COMMAND
        ENVIRONMENT
        INSTALL_COMMAND
        INSTALL_ENVIRONMENT
        LABELS
    )
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(NOT ARG_NAME)
        message(FATAL_ERROR "rj_add_test requires NAME")
    endif()
    if(NOT ARG_COMMAND)
        message(FATAL_ERROR "rj_add_test(${ARG_NAME}) requires COMMAND")
    endif()

    rj_environment_has_runtime_dir(_has_runtime_dir ${ARG_ENVIRONMENT})
    if(NOT _has_runtime_dir)
        # Each launcher test writes the active config into its runtime dir for
        # the preloaded interposer. Use a per-test default so ctest -j runs
        # cannot overwrite another test's config or daemon socket.
        rj_test_runtime_dir_env(_runtime_dir_env "${ARG_NAME}")
        list(APPEND ARG_ENVIRONMENT "${_runtime_dir_env}")
    endif()

    add_test(NAME "${ARG_NAME}" COMMAND ${ARG_COMMAND})

    if(ARG_WORKING_DIRECTORY)
        set_tests_properties(
            "${ARG_NAME}"
            PROPERTIES WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}"
        )
    endif()
    if(ARG_TIMEOUT)
        set_tests_properties("${ARG_NAME}" PROPERTIES TIMEOUT "${ARG_TIMEOUT}")
    endif()
    if(ARG_SKIP_REGULAR_EXPRESSION)
        set_tests_properties(
            "${ARG_NAME}"
            PROPERTIES SKIP_REGULAR_EXPRESSION "${ARG_SKIP_REGULAR_EXPRESSION}"
        )
    endif()
    if(ARG_LABELS)
        list(JOIN ARG_LABELS ";" _labels)
        set_tests_properties("${ARG_NAME}" PROPERTIES LABELS "${_labels}")
    endif()
    if(ARG_ENVIRONMENT)
        list(JOIN ARG_ENVIRONMENT ";" _env)
        set_tests_properties("${ARG_NAME}" PROPERTIES ENVIRONMENT "${_env}")
    endif()

    if(RJ_INSTALL_TESTS AND ARG_INSTALL_COMMAND)
        set(_install_args COMMAND ${ARG_INSTALL_COMMAND})
        if(ARG_TIMEOUT)
            list(APPEND _install_args TIMEOUT "${ARG_TIMEOUT}")
        endif()
        if(ARG_SKIP_REGULAR_EXPRESSION)
            list(
                APPEND _install_args
                SKIP_REGULAR_EXPRESSION
                "${ARG_SKIP_REGULAR_EXPRESSION}"
            )
        endif()
        if(ARG_LABELS)
            list(APPEND _install_args LABELS ${ARG_LABELS})
        endif()
        set(_install_env
            "ROCJITSU_CONFIG_DIR=\${RJ_INSTALLED_CONFIG_DIR}"
            "ROCJITSU_KERNEL_DIR=\${RJ_INSTALLED_KERNEL_DIR}"
        )
        rj_environment_has_runtime_dir(
            _install_has_runtime_dir
            ${ARG_INSTALL_ENVIRONMENT}
        )
        if(NOT _install_has_runtime_dir)
            rj_installed_test_runtime_dir_env(
                _install_runtime_dir_env
                "${ARG_NAME}"
            )
            list(APPEND _install_env "${_install_runtime_dir_env}")
        endif()
        list(APPEND _install_env ${ARG_INSTALL_ENVIRONMENT})
        list(APPEND _install_args ENVIRONMENT ${_install_env})
        rj_install_ctest("${ARG_NAME}" ${_install_args})
    endif()
endfunction()

# Register a GoogleTest target with fine-grained tests in both the build tree
# and installed CTest tree. Build-tree discovery delegates to
# gtest_discover_tests(); installed discovery delegates to rj_install_gtest_cases
# so parameterized TEST_P cases are expanded from the built executable.
function(rj_add_gtest_tests target)
    gtest_discover_tests(${target} ${ARGN})
    if(RJ_INSTALL_TESTS)
        rj_install_test_target(${target})
        rj_install_gtest_cases(${target})
    endif()
endfunction()

# --- Unit / integration test suite ---

add_executable(
    rocjitsu_tests
    rvsim_test.cpp
    amdgpu_vm_test.cpp
    config_test.cpp
    matmul_test.cpp
    vector_add_test.cpp
    gfx1250_sim_test.cpp
    simdojo_sim_test.cpp
    simulated_kfd_test.cpp
    decode_smoke_test.cpp
    shared_infra_test.cpp
    transcendental_test.cpp
    analysis/liveness_test.cpp
    patch/spill_manager_test.cpp
    dbt/code_placement_test.cpp
    dbt/translate_test.cpp
    mtype_flags_test.cpp
    kfd_ioctl_test.cpp
    waitcnt_counter_test.cpp
    instruction_execution_harness_test.cpp
    decode_execute_benchmark.cpp
    execution_plugin_test.cpp
)
target_link_libraries(
    rocjitsu_tests
    PRIVATE
        ${RJ_OBJECT_LIBS}
        rocjitsu_kmd
        util
        flatbuffers
        Threads::Threads
        GTest::gtest_main
)
rj_configure_target(rocjitsu_tests TEST)

set(RJ_IREE_GFX1250_VALIDATION_DIR
    "${CMAKE_SOURCE_DIR}/../../../iree-gfx1250-validation"
    CACHE PATH
    "Path to the workspace IREE gfx1250 validation artifact bundle"
)
if(EXISTS "${RJ_IREE_GFX1250_VALIDATION_DIR}/instruction_audit.json")
    target_compile_definitions(
        rocjitsu_tests
        PRIVATE IREE_GFX1250_VALIDATION_DIR="${RJ_IREE_GFX1250_VALIDATION_DIR}"
    )
    message(
        STATUS
        "IREE gfx1250 artifact translation test enabled (${RJ_IREE_GFX1250_VALIDATION_DIR})"
    )
else()
    message(
        STATUS
        "IREE gfx1250 artifact translation test will skip "
        "(missing ${RJ_IREE_GFX1250_VALIDATION_DIR})"
    )
endif()

include(GoogleTest)
gtest_discover_tests(rocjitsu_tests)

add_executable(
    rj_dbt_translate_smoke_test
    tools/rj_dbt_translate_smoke_test.cpp
)
target_include_directories(
    rj_dbt_translate_smoke_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
        ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
)
target_link_libraries(rj_dbt_translate_smoke_test PRIVATE GTest::gtest)
rj_configure_target(rj_dbt_translate_smoke_test TEST)
add_dependencies(rj_dbt_translate_smoke_test rj_dbt_translate)
add_test(
    NAME "RjDbtTranslate.Smoke"
    COMMAND rj_dbt_translate_smoke_test $<TARGET_FILE:rj_dbt_translate>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# --- Scaling test (standalone executable) ---

add_executable(scaling_test scaling_test.cpp)
target_link_libraries(
    scaling_test
    PRIVATE ${RJ_OBJECT_LIBS} util flatbuffers Threads::Threads
)
target_compile_options(scaling_test PRIVATE -O2)
rj_configure_target(scaling_test TEST)

# --- Interposer tests run through the rocjitsu CLI launcher ---
# All tests that need the simulated GPU go through:
#   rocjitsu --config <cfg> -- <test_binary> [args]
# The CLI handles LD_PRELOAD and config discovery. rj_add_test gives launcher
# tests isolated runtime directories, so no RESOURCE_LOCK is needed.

set(RJ_CDNA4_KMD_CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx950_cdna4_kmd.json")
set(RJ_CDNA4_KMD_2GPU_CONFIG
    "${CMAKE_SOURCE_DIR}/configs/gfx950_cdna4_kmd_2gpu.json"
)
set(RJ_CDNA3_KMD_CONFIG "${CMAKE_SOURCE_DIR}/configs/gfx942_cdna3_kmd.json")
set(RJ_HIP_TEST_ARCH
    "gfx950"
    CACHE STRING
    "GPU ISA used to build basic HIP smoke tests"
)
set(RJ_HIP_TEST_CONFIG
    "${RJ_CDNA4_KMD_CONFIG}"
    CACHE FILEPATH
    "rocjitsu config used by basic HIP smoke tests"
)
set(RJ_HIP_TEST_2GPU_CONFIG
    "${RJ_CDNA4_KMD_2GPU_CONFIG}"
    CACHE FILEPATH
    "rocjitsu config used by basic two-GPU HIP tests"
)
get_filename_component(RJ_CDNA4_KMD_CONFIG_NAME "${RJ_CDNA4_KMD_CONFIG}" NAME)
get_filename_component(
    RJ_CDNA4_KMD_2GPU_CONFIG_NAME
    "${RJ_CDNA4_KMD_2GPU_CONFIG}"
    NAME
)
get_filename_component(RJ_HIP_TEST_CONFIG_NAME "${RJ_HIP_TEST_CONFIG}" NAME)
get_filename_component(
    RJ_HIP_TEST_2GPU_CONFIG_NAME
    "${RJ_HIP_TEST_2GPU_CONFIG}"
    NAME
)
set(RJ_CLI "$<TARGET_FILE:rocjitsu_bin>")

function(rj_detect_kfd_gfx_target_versions out_var)
    set(_versions)
    foreach(
        _nodes_dir
        IN
        ITEMS
            "/sys/devices/virtual/kfd/kfd/topology/nodes"
            "/sys/class/kfd/kfd/topology/nodes"
    )
        if(NOT IS_DIRECTORY "${_nodes_dir}")
            continue()
        endif()

        file(GLOB _node_dirs LIST_DIRECTORIES true "${_nodes_dir}/*")
        foreach(_node_dir IN LISTS _node_dirs)
            if(NOT IS_DIRECTORY "${_node_dir}")
                continue()
            endif()
            if(NOT EXISTS "${_node_dir}/properties")
                continue()
            endif()

            # DBT guest configs name a concrete host ISA. Read KFD topology at
            # configure time so hardware-backed tests are not registered on
            # hosts where /dev/kfd exists but the required ISA is absent.
            file(READ "${_node_dir}/properties" _properties)
            if(_properties MATCHES "(^|[\r\n])gfx_target_version[ \t]+([0-9]+)")
                list(APPEND _versions "${CMAKE_MATCH_2}")
            endif()
        endforeach()
    endforeach()
    if(_versions)
        list(REMOVE_DUPLICATES _versions)
    endif()
    set(${out_var} "${_versions}" PARENT_SCOPE)
endfunction()

rj_detect_kfd_gfx_target_versions(RJ_KFD_GFX_TARGET_VERSIONS)
set(RJ_HAS_KFD_DEVICE FALSE)
if(EXISTS "/dev/kfd")
    set(RJ_HAS_KFD_DEVICE TRUE)
endif()
set(RJ_HAS_GFX1201_GPU FALSE)
if("120001" IN_LIST RJ_KFD_GFX_TARGET_VERSIONS)
    set(RJ_HAS_GFX1201_GPU TRUE)
endif()

set(RJ_KMD_PRELOAD_ENV
    "LD_PRELOAD=$<TARGET_FILE:rocjitsu_shared>"
    "RJ_CONFIG=${CMAKE_SOURCE_DIR}/configs/gfx950_cdna4.json"
)

add_executable(guest_kfd_test guest_kfd_test.cpp)
target_include_directories(
    guest_kfd_test
    PRIVATE ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include ${HSA_INCLUDE_DIR}
)
target_link_libraries(guest_kfd_test PRIVATE Threads::Threads GTest::gtest_main)
rj_configure_target(guest_kfd_test TEST)
add_dependencies(guest_kfd_test rocjitsu_bin rocjitsu_shared rocjitsu_hooks)
if(RJ_HAS_GFX1201_GPU)
    add_test(
        NAME "GuestKfdMultiprocessTest.ForkedChildrenDoNotRemoveParentOverlay"
        COMMAND
            ${RJ_CLI} --config
            ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
            $<TARGET_FILE:guest_kfd_test>
            --gtest_filter=GuestKfdMultiprocessTest.ForkedChildrenDoNotRemoveParentOverlay
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME
            "GuestKfdMultiprocessTest.ForkedChildDropsInheritedGuestDriverBeforeReopen"
        COMMAND
            ${RJ_CLI} --config
            ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
            $<TARGET_FILE:guest_kfd_test>
            --gtest_filter=GuestKfdMultiprocessTest.ForkedChildDropsInheritedGuestDriverBeforeReopen
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "GuestKfdConcurrencyTest.ConcurrentOpenIoctlAndSysfsRedirect"
        COMMAND
            ${RJ_CLI} --config
            ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
            $<TARGET_FILE:guest_kfd_test>
            --gtest_filter=GuestKfdConcurrencyTest.ConcurrentOpenIoctlAndSysfsRedirect
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "GuestKfdMemoryTest.GuestAllocationMmapOffsetIsRejected"
        COMMAND
            ${RJ_CLI} --config
            ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
            $<TARGET_FILE:guest_kfd_test>
            --gtest_filter=GuestKfdMemoryTest.GuestAllocationMmapOffsetIsRejected
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(
        "GuestKfdMultiprocessTest.ForkedChildrenDoNotRemoveParentOverlay"
        "GuestKfdMultiprocessTest.ForkedChildDropsInheritedGuestDriverBeforeReopen"
        "GuestKfdConcurrencyTest.ConcurrentOpenIoctlAndSysfsRedirect"
        "GuestKfdMemoryTest.GuestAllocationMmapOffsetIsRejected"
        PROPERTIES TIMEOUT 60 SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
    )
    foreach(
        _guest_kfd_test
        IN
        ITEMS
            "GuestKfdMultiprocessTest.ForkedChildrenDoNotRemoveParentOverlay"
            "GuestKfdMultiprocessTest.ForkedChildDropsInheritedGuestDriverBeforeReopen"
            "GuestKfdConcurrencyTest.ConcurrentOpenIoctlAndSysfsRedirect"
            "GuestKfdMemoryTest.GuestAllocationMmapOffsetIsRejected"
    )
        rj_set_default_runtime_dir("${_guest_kfd_test}")
    endforeach()
    message(
        STATUS
        "GuestKfd DBT guest tests enabled (gfx1201 host GPU detected)"
    )
else()
    message(
        STATUS
        "GuestKfd DBT guest tests built but NOT registered with CTest "
        "(no gfx1201 KFD host GPU)"
    )
endif()

add_executable(hsa_hooks_unit_test dbt/hsa_hooks_unit_test.cpp)
target_compile_definitions(
    hsa_hooks_unit_test
    PRIVATE
        RJ_HOOK_UNIT_CONFIG_PATH="${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json"
)
target_include_directories(
    hsa_hooks_unit_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
        ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
        ${CMAKE_SOURCE_DIR}/lib/util/include
        ${GENERATED_DIR}
)
target_include_directories(
    hsa_hooks_unit_test
    SYSTEM
    PRIVATE ${HSA_INCLUDE_DIR}
)
target_link_libraries(
    hsa_hooks_unit_test
    PRIVATE rocjitsu_hooks Threads::Threads GTest::gtest_main
)
rj_configure_target(hsa_hooks_unit_test TEST)
gtest_discover_tests(hsa_hooks_unit_test)

# --- Interposer loader initialization regression tests ---

add_library(early_mmap_preload SHARED early_mmap_preload_test.cpp)
rj_configure_target(early_mmap_preload TEST)
add_dependencies(early_mmap_preload rocjitsu_shared)

add_executable(early_mmap_driver early_mmap_driver_test.cpp)
rj_configure_target(early_mmap_driver TEST)

if(RJ_INSTALL_TESTS)
    rj_install_test_target(early_mmap_driver)
    install(
        TARGETS early_mmap_preload
        LIBRARY DESTINATION ${RJ_INSTALL_TEST_BIN_DIR}
    )
endif()

# Keep rocjitsu first in LD_PRELOAD. The loader maps it before
# early_mmap_preload, but runs early_mmap_preload's constructor before
# init_interposer(). That constructor calls mmap/munmap early enough to exercise
# the pre-init fallback; verifying the exact path directly would require adding
# observable test state to the interposer.
rj_add_test(
    NAME "InterposerTest.EarlyMmapBeforeRocjitsuConstructor"
    COMMAND $<TARGET_FILE:early_mmap_driver>
    ENVIRONMENT
        "LD_PRELOAD=$<TARGET_FILE:rocjitsu_shared>:$<TARGET_FILE:early_mmap_preload>"
    INSTALL_COMMAND "bin/early_mmap_driver"
    INSTALL_ENVIRONMENT
        "LD_PRELOAD=\${RJ_INSTALLED_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}rocjitsu${CMAKE_SHARED_LIBRARY_SUFFIX}:bin/${CMAKE_SHARED_LIBRARY_PREFIX}early_mmap_preload${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

# --- rocminfo test (requires rocminfo binary + HSA runtime + CLI launcher) ---
# Spawns rocminfo via the rocjitsu CLI and validates output.

_rj_path_from_installed_test_bin(
    _rj_installed_rocjitsu_bin
    "${CMAKE_INSTALL_BINDIR}"
    "rocjitsu"
)
_rj_path_from_installed_test_bin(
    _rj_installed_cdna4_kmd_config
    "${CMAKE_INSTALL_DATADIR}/rocjitsu/configs"
    "${RJ_CDNA4_KMD_CONFIG_NAME}"
)
_rj_path_from_installed_test_bin(
    _rj_installed_dbt_guest_config
    "${CMAKE_INSTALL_DATADIR}/rocjitsu/configs"
    "guest_gfx950_on_gfx1201.json"
)

find_program(
    ROCMINFO_EXECUTABLE
    NAMES rocminfo
    HINTS ${CMAKE_SOURCE_DIR}/.venv
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES bin
    DOC "Path to the rocminfo executable"
)
if(ROCMINFO_EXECUTABLE AND RJ_HAS_KFD_DEVICE)
    add_executable(rocminfo_test rocminfo_test.cpp)
    target_compile_definitions(
        rocminfo_test
        PRIVATE
            ROCMINFO_PATH="${ROCMINFO_EXECUTABLE}"
            ROCJITSU_BIN="$<TARGET_FILE:rocjitsu_bin>"
            RJ_CONFIG_PATH="${RJ_CDNA4_KMD_CONFIG}"
    )
    target_link_libraries(rocminfo_test PRIVATE GTest::gtest_main)
    add_dependencies(rocminfo_test rocjitsu_bin rocjitsu_kmd_shim)

    add_test(
        NAME "RocminfoTest.ExitsSuccessfully"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.ExitsSuccessfully
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "RocminfoTest.InterposerActive"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.InterposerActive
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "RocminfoTest.HsaSystemAttributes"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.HsaSystemAttributes
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "RocminfoTest.DetectsGpuAgent"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.DetectsGpuAgent
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "RocminfoTest.ReportsGfx950"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.ReportsGfx950
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "RocminfoTest.ReportsWavefrontSize"
        COMMAND rocminfo_test --gtest_filter=RocminfoTest.ReportsWavefrontSize
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
        rj_add_test(
            NAME "RocminfoTest.${_rocminfo_test}"
            COMMAND
                rocminfo_test
                "--gtest_filter=RocminfoTest.${_rocminfo_test}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            ENVIRONMENT ${RJ_KMD_PRELOAD_ENV}
            INSTALL_COMMAND
                "bin/rocminfo_test"
                "--gtest_filter=RocminfoTest.${_rocminfo_test}"
            ${_rocminfo_install_environment_arg}
        )
    endforeach()

    add_executable(dbt_guest_rocminfo_test rocminfo_test.cpp)
    target_compile_definitions(
        dbt_guest_rocminfo_test
        PRIVATE
            ROCMINFO_PATH="${ROCMINFO_EXECUTABLE}"
            ROCJITSU_BIN="$<TARGET_FILE:rocjitsu_bin>"
            RJ_CONFIG_PATH="${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json"
            RJ_INSTALLED_ROCJITSU_BIN="${_rj_installed_rocjitsu_bin}"
            RJ_INSTALLED_CONFIG_PATH="${_rj_installed_dbt_guest_config}"
    )
    target_link_libraries(dbt_guest_rocminfo_test PRIVATE GTest::gtest_main)
    add_dependencies(
        dbt_guest_rocminfo_test
        rocjitsu_bin
        rocjitsu_shared
        rocjitsu_hooks
    )

    if(RJ_HAS_GFX1201_GPU)
        add_test(
            NAME "DbtGuestRocminfoTest.ExitsSuccessfully"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.ExitsSuccessfully
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME "DbtGuestRocminfoTest.InterposerActive"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.InterposerActive
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME "DbtGuestRocminfoTest.HsaSystemAttributes"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.HsaSystemAttributes
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME "DbtGuestRocminfoTest.DetectsGpuAgent"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.DetectsGpuAgent
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME "DbtGuestRocminfoTest.ReportsGfx950"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.ReportsGfx950
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME "DbtGuestRocminfoTest.ReportsWavefrontSize"
            COMMAND
                dbt_guest_rocminfo_test
                --gtest_filter=RocminfoTest.ReportsWavefrontSize
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        foreach(
            _dbt_guest_rocminfo_test
            IN
            ITEMS
                "DbtGuestRocminfoTest.ExitsSuccessfully"
                "DbtGuestRocminfoTest.InterposerActive"
                "DbtGuestRocminfoTest.HsaSystemAttributes"
                "DbtGuestRocminfoTest.DetectsGpuAgent"
                "DbtGuestRocminfoTest.ReportsGfx950"
                "DbtGuestRocminfoTest.ReportsWavefrontSize"
        )
            rj_set_default_runtime_dir("${_dbt_guest_rocminfo_test}")
        endforeach()
        message(
            STATUS
            "DBT guest rocminfo tests enabled (gfx1201 host GPU detected)"
        )
    else()
        message(
            STATUS
            "DBT guest rocminfo tests built but NOT registered with CTest "
            "(no gfx1201 KFD host GPU)"
        )
    endif()
    message(STATUS "rocminfo test enabled (found ${ROCMINFO_EXECUTABLE})")
elseif(ROCMINFO_EXECUTABLE)
    message(
        STATUS
        "rocminfo test disabled (/dev/kfd not found; rocminfo exits before "
        "simulated KFD discovery)"
    )
else()
    message(STATUS "rocminfo test disabled (rocminfo not found)")
endif()

# --- HSA init test (requires ROCm HSA runtime + CLI launcher) ---

find_library(
    HSA_RUNTIME64
    hsa-runtime64
    PATHS /opt/rocm/lib
    ENV ROCM_PATH
    PATH_SUFFIXES lib
)
if(HSA_RUNTIME64)
    get_filename_component(RJ_HSA_LIBRARY_DIR "${HSA_RUNTIME64}" DIRECTORY)
    get_filename_component(RJ_HSA_ROCM_ROOT "${RJ_HSA_LIBRARY_DIR}" DIRECTORY)
    set(RJ_HSA_INCLUDE_DIR "${RJ_HSA_ROCM_ROOT}/include")

    add_executable(hsa_init_test hsa_init_test.cpp)
    target_include_directories(
        hsa_init_test
        PRIVATE ${RJ_HSA_INCLUDE_DIR} ${HSA_INCLUDE_DIR}
    )
    target_link_libraries(hsa_init_test PRIVATE ${HSA_RUNTIME64} GTest::gtest)
    target_link_options(hsa_init_test PRIVATE -Wl,-rpath,${RJ_HSA_LIBRARY_DIR})
    add_dependencies(hsa_init_test rocjitsu_bin rocjitsu_kmd_shim)
    add_test(
        NAME "HsaTest.InitSucceeded"
        COMMAND
            ${RJ_CLI} --config ${RJ_CDNA4_KMD_CONFIG} --
            $<TARGET_FILE:hsa_init_test> --gtest_filter=HsaTest.InitSucceeded
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME "HsaTest.GpuAgentFound"
        COMMAND
            ${RJ_CLI} --config ${RJ_CDNA4_KMD_CONFIG} --
            $<TARGET_FILE:hsa_init_test> --gtest_filter=HsaTest.GpuAgentFound
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(
        "HsaTest.InitSucceeded"
        "HsaTest.GpuAgentFound"
        PROPERTIES SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]"
    )
    message(STATUS "HSA init test enabled (found ${HSA_RUNTIME64})")

    # --- HSA translate test: CDNA4→RDNA4 on-device verification ---
    # Requires both compiled device kernels AND an RDNA4 GPU on the host.
    if(HAS_DEVICE_KERNELS)
        # Detect supported DBT host GPUs at configure time via rocm_agent_enumerator.
        find_program(
            ROCM_AGENT_ENUM
            rocm_agent_enumerator
            HINTS
            ENV ROCM_PATH
            ${RJ_HSA_ROCM_ROOT}
            /opt/rocm
            PATH_SUFFIXES bin
        )
        set(HAS_DBT_VECTOR_ADD_GPU FALSE)
        set(HAS_CDNA3_GPU FALSE)
        set(HAS_RDNA4_GPU FALSE)
        set(HAS_GFX1201_GPU FALSE)
        if(ROCM_AGENT_ENUM)
            execute_process(
                COMMAND ${ROCM_AGENT_ENUM}
                OUTPUT_VARIABLE _agents
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _agent_rc
            )
            if(
                _agent_rc EQUAL 0
                AND _agents MATCHES "gfx94[012]|gfx1100|gfx120[01]"
            )
                set(HAS_DBT_VECTOR_ADD_GPU TRUE)
            endif()
            if(_agent_rc EQUAL 0 AND _agents MATCHES "gfx94[012]")
                set(HAS_CDNA3_GPU TRUE)
            endif()
            if(_agent_rc EQUAL 0 AND _agents MATCHES "gfx120[01]")
                set(HAS_RDNA4_GPU TRUE)
            endif()
            if(_agent_rc EQUAL 0 AND _agents MATCHES "gfx1201")
                set(HAS_GFX1201_GPU TRUE)
            endif()
        endif()

        add_executable(hsa_translate_test dbt/hsa_translate_test.cpp)
        target_include_directories(
            hsa_translate_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
                ${CMAKE_SOURCE_DIR}/lib/util/include
                ${RJ_HSA_INCLUDE_DIR}
        )
        target_link_libraries(
            hsa_translate_test
            PRIVATE
                ${RJ_OBJECT_LIBS}
                ${HSA_RUNTIME64}
                GTest::gtest
                GTest::gtest_main
        )
        target_link_options(
            hsa_translate_test
            PRIVATE -Wl,-rpath,${RJ_HSA_LIBRARY_DIR}
        )
        target_compile_definitions(
            hsa_translate_test
            PRIVATE HAS_HOST_AMDGPU=1 KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
        )
        add_dependencies(hsa_translate_test device_kernels)
        rj_configure_target(hsa_translate_test TEST)

        add_executable(hsa_hooks_test dbt/hsa_hooks_test.cpp)
        target_include_directories(
            hsa_hooks_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
                ${CMAKE_SOURCE_DIR}/lib/util/include
                ${RJ_HSA_INCLUDE_DIR}
        )
        target_link_libraries(
            hsa_hooks_test
            PRIVATE
                ${RJ_OBJECT_LIBS}
                ${HSA_RUNTIME64}
                GTest::gtest
                GTest::gtest_main
        )
        target_link_options(
            hsa_hooks_test
            PRIVATE -Wl,-rpath,${RJ_HSA_LIBRARY_DIR}
        )
        target_compile_definitions(
            hsa_hooks_test
            PRIVATE HAS_HOST_AMDGPU=1 KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
        )
        add_dependencies(
            hsa_hooks_test
            device_kernels
            rocjitsu_bin
            rocjitsu_hooks
        )
        rj_configure_target(hsa_hooks_test TEST)

        if(HAS_DBT_VECTOR_ADD_GPU)
            add_test(
                NAME "HsaTranslateTest.TranslateAndDispatchVectorAdd"
                COMMAND hsa_translate_test --gtest_filter=*VectorAdd*
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            message(
                STATUS
                "HSA translate test enabled (supported DBT GPU detected)"
            )
        else()
            message(
                STATUS
                "HSA translate vector_add test built but NOT registered with CTest "
                "(no supported DBT GPU)"
            )
        endif()

        if(HAS_GFX1201_GPU)
            rj_add_test(
                NAME "HsaHooksTest.TranslateGfx950Mfma16x16ThroughToolsLib"
                COMMAND
                    $<TARGET_FILE:rocjitsu_bin> --config
                    ${CMAKE_SOURCE_DIR}/configs/guest_gfx950_on_gfx1201.json --
                    $<TARGET_FILE:hsa_hooks_test>
                    --gtest_filter=HsaHooksTest.TranslateGfx950Mfma16x16ThroughToolsLib
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                TIMEOUT 60
                INSTALL_COMMAND
                    "\${RJ_INSTALLED_BINDIR}/rocjitsu"
                    "--config"
                    "\${RJ_INSTALLED_CONFIG_DIR}/guest_gfx950_on_gfx1201.json"
                    "--"
                    "bin/hsa_hooks_test"
                    "--gtest_filter=HsaHooksTest.TranslateGfx950Mfma16x16ThroughToolsLib"
            )
        endif()

        message(
            STATUS
            "HSA hooks test built but NOT registered with CTest "
            "(CDNA4 MFMA hook smoke needs MFMA scalar/inline operand lowering)"
        )

        if(HAS_RDNA4_GPU)
            add_test(
                NAME "HsaTranslateTest.TranslateAndDispatchMfma16x16"
                COMMAND hsa_translate_test --gtest_filter=*Mfma16x16*
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            message(STATUS "HSA translate test enabled (RDNA4 GPU detected)")
        else()
            message(
                STATUS
                "HSA translate test built but NOT registered with CTest (no RDNA4 GPU)"
            )
        endif()

        add_executable(
            cdna4_to_cdna3_dispatch_test
            dbt/cdna4_to_cdna3_dispatch_test.cpp
        )
        target_include_directories(
            cdna4_to_cdna3_dispatch_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/include
                ${CMAKE_SOURCE_DIR}/lib/rocjitsu/src
                ${CMAKE_SOURCE_DIR}/lib/util/include
                ${RJ_HSA_INCLUDE_DIR}
        )
        target_link_libraries(
            cdna4_to_cdna3_dispatch_test
            PRIVATE
                ${RJ_OBJECT_LIBS}
                ${HSA_RUNTIME64}
                GTest::gtest
                GTest::gtest_main
        )
        target_link_options(
            cdna4_to_cdna3_dispatch_test
            PRIVATE -Wl,-rpath,${RJ_HSA_LIBRARY_DIR}
        )
        target_compile_definitions(
            cdna4_to_cdna3_dispatch_test
            PRIVATE HAS_HOST_AMDGPU=1 KERNEL_DIR="${KERNEL_OUTPUT_DIR}"
        )
        add_dependencies(cdna4_to_cdna3_dispatch_test device_kernels)
        rj_configure_target(cdna4_to_cdna3_dispatch_test TEST)

        add_test(
            NAME "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulTranslates"
            COMMAND
                cdna4_to_cdna3_dispatch_test
                --gtest_filter=Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulTranslates
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        set_tests_properties(
            "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulTranslates"
            PROPERTIES TIMEOUT 120
        )

        if(HAS_CDNA3_GPU)
            add_test(
                NAME "Cdna4ToCdna3DispatchTest.DynamicCopyLoopDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.DynamicCopyLoopDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            add_test(
                NAME
                    "Cdna4ToCdna3DispatchTest.TritonDynamicMatmulDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.TritonDynamicMatmulDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            add_test(
                NAME
                    "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            add_test(
                NAME
                    "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulConservativeLivenessDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulConservativeLivenessDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            add_test(
                NAME
                    "Cdna4ToCdna3DispatchTest.TritonFlashAttentionNoAsyncDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.TritonFlashAttentionNoAsyncDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            add_test(
                NAME
                    "Cdna4ToCdna3DispatchTest.TritonFlashAttentionBufferAsyncDispatchAndRun"
                COMMAND
                    cdna4_to_cdna3_dispatch_test
                    --gtest_filter=Cdna4ToCdna3DispatchTest.TritonFlashAttentionBufferAsyncDispatchAndRun
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
            set_tests_properties(
                "Cdna4ToCdna3DispatchTest.DynamicCopyLoopDispatchAndRun"
                "Cdna4ToCdna3DispatchTest.TritonDynamicMatmulDispatchAndRun"
                "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulDispatchAndRun"
                "Cdna4ToCdna3DispatchTest.TritonBufferAsyncMatmulConservativeLivenessDispatchAndRun"
                "Cdna4ToCdna3DispatchTest.TritonFlashAttentionNoAsyncDispatchAndRun"
                "Cdna4ToCdna3DispatchTest.TritonFlashAttentionBufferAsyncDispatchAndRun"
                PROPERTIES TIMEOUT 120
            )
            message(
                STATUS
                "CDNA4->CDNA3 dispatch tests enabled (CDNA3 GPU detected)"
            )
        else()
            message(
                STATUS
                "CDNA4->CDNA3 dispatch tests built but NOT registered with CTest "
                "(no CDNA3 GPU)"
            )
        endif()
    endif()
else()
    message(STATUS "HSA init test disabled (libhsa-runtime64 not found)")
endif()

# --- HIP tests (requires hipcc + CLI launcher) ---

rj_find_amdcxx(AMDCXX)
set(RJ_ENABLE_HIP_TESTS ON)
find_library(HIP_RUNTIME64 amdhip64 PATHS "${ROCM_PATH}" PATH_SUFFIXES lib)
set(RJ_HIP_RUNTIME_SUPPORTS_TEST_ARCH FALSE)
set(RJ_HIP_RUNTIME_SUPPORTS_GFX950 FALSE)
set(RJ_HIP_RUNTIME_SUPPORTS_GFX942 FALSE)

function(rj_hip_runtime_supports_arch out_var arch)
    set(_supports FALSE)
    if(HIP_RUNTIME64)
        # Recent HIP runtimes reject agents whose PCI ID/ISA are not in their
        # device table, even when ROCR can enumerate the agent. Gate CTest
        # registration by the user-space runtime's advertised architectures so
        # unsupported targets do not fail before exercising rocjitsu.
        file(STRINGS "${HIP_RUNTIME64}" _rj_hip_arch_strings REGEX "${arch}")
        if(_rj_hip_arch_strings)
            set(_supports TRUE)
        endif()
    endif()
    set(${out_var} "${_supports}" PARENT_SCOPE)
endfunction()

if(NOT AMDCXX)
    set(RJ_ENABLE_HIP_TESTS OFF)
    message(STATUS "HIP tests disabled (amdclang++ not found)")
elseif(NOT HSA_RUNTIME64)
    set(RJ_ENABLE_HIP_TESTS OFF)
    message(STATUS "HIP tests disabled (libhsa-runtime64 not found)")
elseif(NOT HIP_RUNTIME64)
    set(RJ_ENABLE_HIP_TESTS OFF)
    message(STATUS "HIP tests disabled (libamdhip64 not found)")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND RJ_SANITIZER_LINK_OPTIONS)
    set(RJ_ENABLE_HIP_TESTS OFF)
    message(STATUS "HIP tests disabled (GNU host compiler with sanitizers)")
endif()

if(RJ_ENABLE_HIP_TESTS)
    rj_hip_runtime_supports_arch(
        RJ_HIP_RUNTIME_SUPPORTS_TEST_ARCH
        "${RJ_HIP_TEST_ARCH}"
    )
    rj_hip_runtime_supports_arch(RJ_HIP_RUNTIME_SUPPORTS_GFX950 "gfx950")
    rj_hip_runtime_supports_arch(RJ_HIP_RUNTIME_SUPPORTS_GFX942 "gfx942")
endif()

if(RJ_ENABLE_HIP_TESTS)
    set(GTEST_INC ${googletest_SOURCE_DIR}/googletest/include)
    set(GTEST_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

    # Helper: build a HIP test binary with hipcc + gtest.
    function(rj_add_hip_test TEST_NAME)
        cmake_parse_arguments(ARG "" "ARCH;SOURCE" "EXTRA_LIBS" ${ARGN})
        if(NOT ARG_ARCH)
            set(ARG_ARCH ${RJ_HIP_TEST_ARCH})
        endif()
        if(ARG_SOURCE)
            set(HIP_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SOURCE})
        else()
            set(HIP_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp)
        endif()
        set(HIP_TEST_BIN ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
        add_custom_command(
            OUTPUT ${HIP_TEST_BIN}
            COMMAND
                ${HIPCC_EXECUTABLE} --offload-arch=${ARG_ARCH} -std=c++20
                -isystem ${GTEST_INC} -I${CMAKE_CURRENT_SOURCE_DIR}
                -L${GTEST_LIB_DIR} -lgtest -lpthread ${ARG_EXTRA_LIBS}
                -Wl,-rpath,${GTEST_LIB_DIR} -o ${HIP_TEST_BIN} ${HIP_TEST_SRC}
            DEPENDS ${HIP_TEST_SRC} GTest::gtest GTest::gtest_main
            COMMENT "Building ${TEST_NAME} with hipcc (${ARG_ARCH})"
            VERBATIM
        )
        add_custom_target(${TEST_NAME}_target ALL DEPENDS ${HIP_TEST_BIN})
        add_dependencies(${TEST_NAME}_target rocjitsu_bin rocjitsu_kmd_shim)
    endfunction()

    # Helper: register a HIP test case run through the CLI launcher.
    function(rj_add_hip_test_case TEST_BIN TEST_CASE)
        cmake_parse_arguments(ARG "" "CONFIG;NAME" "" ${ARGN})
        if(NOT ARG_CONFIG)
            set(ARG_CONFIG ${RJ_CDNA4_KMD_CONFIG})
        endif()
        if(NOT ARG_NAME)
            set(ARG_NAME ${TEST_CASE})
        endif()
        add_test(
            NAME ${ARG_NAME}
            COMMAND
                ${RJ_CLI} --config ${ARG_CONFIG} --
                ${CMAKE_CURRENT_BINARY_DIR}/${TEST_BIN}
                --gtest_filter=${TEST_CASE}
        )
    endfunction()

    if(RJ_HIP_RUNTIME_SUPPORTS_TEST_ARCH)
        rj_add_hip_test(hip_memcpy_test ARCH ${RJ_HIP_TEST_ARCH})
        rj_add_hip_test_case(
            hip_memcpy_test
            "HipMemcpyTest.RoundTripFloat"
            CONFIG ${RJ_HIP_TEST_CONFIG}
        )
        rj_add_hip_test_case(
            hip_memcpy_test
            "HipMemcpyTest.RoundTripInt"
            CONFIG ${RJ_HIP_TEST_CONFIG}
        )
        rj_add_hip_test_case(
            hip_memcpy_test
            "HipMemcpyTest.DeviceToDevice"
            CONFIG ${RJ_HIP_TEST_CONFIG}
        )
        rj_add_hip_test(hip_vector_add_test ARCH ${RJ_HIP_TEST_ARCH})
        rj_add_hip_test_case(
            hip_vector_add_test
            "HipVectorAddTest.CorrectResult"
            CONFIG ${RJ_HIP_TEST_CONFIG}
        )

        # --- Daemon RPC tests (gtest-driven) ---
        # A gtest fixture manages the daemon lifecycle: fork+exec the daemon in
        # SetUp, run the HIP test binary as a subprocess, tear down in TearDown.
        _rj_path_from_installed_test_bin(
            _rj_installed_daemon_config
            "${CMAKE_INSTALL_DATADIR}/rocjitsu/configs"
            "${RJ_HIP_TEST_CONFIG_NAME}"
        )
        _rj_path_from_installed_test_bin(
            _rj_installed_daemon_config_2gpu
            "${CMAKE_INSTALL_DATADIR}/rocjitsu/configs"
            "${RJ_HIP_TEST_2GPU_CONFIG_NAME}"
        )
        _rj_path_from_installed_test_bin(
            _rj_installed_preload_lib
            "${CMAKE_INSTALL_LIBDIR}"
            "${CMAKE_SHARED_LIBRARY_PREFIX}rocjitsu${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )

        add_executable(daemon_test daemon_test.cpp)
        target_compile_definitions(
            daemon_test
            PRIVATE
                RJ_DAEMON_BIN="$<TARGET_FILE:rocjitsu_bin>"
                RJ_DAEMON_CONFIG="${RJ_HIP_TEST_CONFIG}"
                RJ_DAEMON_CONFIG_2GPU="${RJ_HIP_TEST_2GPU_CONFIG}"
                RJ_PRELOAD_LIB="$<TARGET_FILE:rocjitsu_shared>"
                RJ_HIP_VECTOR_ADD_BIN="${CMAKE_CURRENT_BINARY_DIR}/hip_vector_add_test"
                RJ_HIP_MEMCPY_BIN="${CMAKE_CURRENT_BINARY_DIR}/hip_memcpy_test"
                RJ_HIP_RCCL_BIN="${CMAKE_CURRENT_BINARY_DIR}/hip_rccl_test"
                RJ_INSTALLED_DAEMON_BIN="${_rj_installed_rocjitsu_bin}"
                RJ_INSTALLED_DAEMON_CONFIG="${_rj_installed_daemon_config}"
                RJ_INSTALLED_DAEMON_CONFIG_2GPU="${_rj_installed_daemon_config_2gpu}"
                RJ_INSTALLED_PRELOAD_LIB="${_rj_installed_preload_lib}"
                RJ_INSTALLED_HIP_VECTOR_ADD_BIN="hip_vector_add_test"
                RJ_INSTALLED_HIP_MEMCPY_BIN="hip_memcpy_test"
                RJ_INSTALLED_HIP_RCCL_BIN="hip_rccl_test"
        )
        target_link_libraries(daemon_test PRIVATE GTest::gtest_main)
        add_dependencies(
            daemon_test
            rocjitsu_bin
            hip_vector_add_test_target
            hip_memcpy_test_target
        )
        if(RJ_INSTALL_TESTS)
            rj_install_test_target(daemon_test)
        endif()
        set(RJ_DAEMON_TESTS
            DaemonTest.HipVectorAdd
            DaemonTest.HipMemcpyRoundTripFloat
            DaemonTest.HipMemcpyRoundTripInt
            DaemonTest.HipMemcpyDeviceToDevice
            DaemonTest.TwoIndependentClients
        )
        foreach(TC IN LISTS RJ_DAEMON_TESTS)
            rj_add_test(
                NAME ${TC}
                COMMAND daemon_test --gtest_filter=${TC}
                TIMEOUT 120
                LABELS daemon
                INSTALL_COMMAND
                    "bin/daemon_test"
                    "--gtest_filter=${TC}"
            )
        endforeach()

        # --- RCCL collective test ---
        find_library(RCCL_LIB rccl PATHS "${ROCM_PATH}" PATH_SUFFIXES lib)
        if(RCCL_LIB AND RJ_HIP_RUNTIME_SUPPORTS_GFX950)
            rj_add_hip_test(hip_rccl_test ARCH gfx950 EXTRA_LIBS ${RCCL_LIB})
            add_dependencies(daemon_test hip_rccl_test_target)
            set(RJ_RCCL_TESTS
                RcclDaemonTest.AllReduce
                RcclDaemonTest.Broadcast
                RcclDaemonTest.AllGather
                RcclDaemonTest.ReduceScatter
                RcclDaemonTest.SendRecv
            )
            foreach(TC IN LISTS RJ_RCCL_TESTS)
                rj_add_test(
                    NAME ${TC}
                    COMMAND daemon_test --gtest_filter=${TC}
                    TIMEOUT 180
                    LABELS daemon rccl
                    INSTALL_COMMAND
                        "bin/daemon_test"
                        "--gtest_filter=${TC}"
                )
            endforeach()
            message(STATUS "RCCL collective test enabled (found ${RCCL_LIB})")
        elseif(RCCL_LIB)
            message(
                STATUS
                "RCCL collective test disabled (libamdhip64 does not advertise gfx950 support)"
            )
        else()
            message(STATUS "RCCL collective test disabled (librccl not found)")
        endif()
    else()
        message(
            STATUS
            "HIP smoke/daemon tests disabled (libamdhip64 does not advertise ${RJ_HIP_TEST_ARCH} support)"
        )
    endif()

    if(RJ_HIP_RUNTIME_SUPPORTS_GFX950)
        # This test exercises CDNA ACCVGPR/DS transpose instructions directly
        # in inline assembly, so it must stay on a CDNA target even when the
        # generic HIP smoke tests are built for a local RDNA host such as
        # gfx1201.
        rj_add_hip_test(hip_ds_transpose_acc_test ARCH gfx950)

        rj_add_hip_test(hip_bfe_2d_test ARCH gfx950)
        rj_add_hip_test_case(hip_bfe_2d_test "BfeRegressionTest.ThreadIdxY")

        # --- Logging plugin end-to-end test ---
        rj_add_hip_test(
            hip_logging_test
            ARCH gfx950
            SOURCE logging/hip_logging_test.hip
        )
        set(LOGGING_SINK_DIR ${CMAKE_CURRENT_BINARY_DIR}/logging_test_output)
        set(LOGGING_RUNTIME_DIR ${LOGGING_SINK_DIR}/runtime)
        file(MAKE_DIRECTORY ${LOGGING_SINK_DIR})
        file(MAKE_DIRECTORY ${LOGGING_RUNTIME_DIR})
        rj_add_hip_test_case(
            hip_logging_test "LoggingTest.dispatch_logged"
            TIMEOUT 30
            ENVIRONMENT
                "ROCJITSU_RUNTIME_DIR=${LOGGING_RUNTIME_DIR}"
                "RJ_LOG=1"
                "RJ_SINKS=file"
                "RJ_SINK_DIR=${LOGGING_SINK_DIR}"
        )

        # --- Profiled plugin group end-to-end test ---
        rj_add_hip_test(
            hip_profiled_plugin_test
            ARCH gfx950
            SOURCE logging/hip_profiled_plugin_test.hip
        )
        set(PROFILED_SINK_DIR ${CMAKE_CURRENT_BINARY_DIR}/profiled_test_output)
        set(PROFILED_RUNTIME_DIR ${PROFILED_SINK_DIR}/runtime)
        file(MAKE_DIRECTORY ${PROFILED_SINK_DIR})
        file(MAKE_DIRECTORY ${PROFILED_RUNTIME_DIR})
        rj_add_hip_test_case(
            hip_profiled_plugin_test "ProfiledPluginTest.hook_profile_output"
            TIMEOUT 30
            ENVIRONMENT
                "ROCJITSU_RUNTIME_DIR=${PROFILED_RUNTIME_DIR}"
                "RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1"
                "RJ_SINKS=file"
                "RJ_SINK_DIR=${PROFILED_SINK_DIR}"
        )

        add_subdirectory(race-detector)
    else()
        message(
            STATUS
            "CDNA4 HIP tests disabled (libamdhip64 does not advertise gfx950 support)"
        )
    endif()

    # --- rocBLAS GEMM test ---
    find_library(
        ROCBLAS_LIB
        rocblas
        PATHS /opt/rocm/lib
        ENV ROCM_PATH
        PATH_SUFFIXES lib
    )
    if(ROCBLAS_LIB)
        if(RJ_HIP_RUNTIME_SUPPORTS_GFX950)
            rj_add_hip_test(hip_gemm_test ARCH gfx950 EXTRA_LIBS ${ROCBLAS_LIB})

            foreach(
                TC
                IN
                ITEMS
                    Tiny_2x2x3
                    Square_4x4
                    Square_8x8
                    Square_16x16
                    Square_32x32
                    Square_64x64
                    Rect_16x32x8
                    Rect_1x64x1
                    Rect_64x1x64
                    Rect_7x11x13
                    AlphaBeta
                    BetaZero
                    Large_2048x2048
            )
                if(TC STREQUAL "Large_2048x2048")
                    rj_add_hip_test_case(
                        hip_gemm_test
                        "RocblasGemmTest.${TC}"
                        TIMEOUT 300
                    )
                else()
                    rj_add_hip_test_case(hip_gemm_test "RocblasGemmTest.${TC}")
                endif()
            endforeach()
            foreach(I RANGE 0 4)
                rj_add_hip_test_case(hip_gemm_test "RocblasGemmFuzz.Iter${I}")
            endforeach()
            message(
                STATUS
                "rocBLAS GEMM gfx950 tests enabled (found ${ROCBLAS_LIB})"
            )
        endif()

        if(RJ_HIP_RUNTIME_SUPPORTS_GFX942)
            rj_add_hip_test(
                hip_gemm_test_cdna3
                ARCH gfx942
                SOURCE hip_gemm_test.cpp
                EXTRA_LIBS ${ROCBLAS_LIB}
            )

            foreach(
                TC
                IN
                ITEMS Tiny_2x2x3 Square_16x16 Square_32x32 Rect_7x11x13
            )
                rj_add_hip_test_case(
                    hip_gemm_test_cdna3
                    "RocblasGemmTest.${TC}"
                    CONFIG ${RJ_CDNA3_KMD_CONFIG}
                    NAME "RocblasGemmCdna3.${TC}"
                )
            endforeach()
            foreach(I RANGE 0 2)
                rj_add_hip_test_case(
                    hip_gemm_test_cdna3
                    "RocblasGemmFuzz.Iter${I}"
                    CONFIG ${RJ_CDNA3_KMD_CONFIG}
                    NAME "RocblasGemmCdna3.Fuzz_Iter${I}"
                )
            endforeach()
            message(
                STATUS
                "rocBLAS GEMM gfx942 tests enabled (found ${ROCBLAS_LIB})"
            )
        endif()

        if(
            NOT RJ_HIP_RUNTIME_SUPPORTS_GFX950
            AND NOT RJ_HIP_RUNTIME_SUPPORTS_GFX942
        )
            message(
                STATUS
                "rocBLAS GEMM test disabled (libamdhip64 does not advertise gfx950 or gfx942 support)"
            )
        endif()
    else()
        message(STATUS "rocBLAS GEMM test disabled (librocblas not found)")
    endif()

    message(STATUS "HIP tests enabled (found ${AMDCXX})")
endif()

# --- Narrow conversion HIP CTS tests ---
# Uses the same HIP compiler path as the other CDNA4 HIP tests.
if(RJ_ENABLE_HIP_TESTS AND RJ_HIP_RUNTIME_SUPPORTS_GFX950)
    set(CVT_NARROW_BIN ${CMAKE_CURRENT_BINARY_DIR}/hip_cvt_narrow_test)
    add_custom_command(
        OUTPUT ${CVT_NARROW_BIN}
        COMMAND
            ${AMDCXX} -x hip --offload-arch=gfx950 --rocm-path=${ROCM_PATH}
            -std=c++20 ${RJ_HIP_SANITIZER_FLAGS}
            -I${CMAKE_SOURCE_DIR}/lib/util/include -o ${CVT_NARROW_BIN}
            ${CMAKE_CURRENT_SOURCE_DIR}/hip_cvt_narrow_test.cpp
            ${RJ_HIP_SANITIZER_LINK_FLAGS}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hip_cvt_narrow_test.cpp
        COMMENT "Building hip_cvt_narrow_test with amdclang++ -x hip (gfx950)"
        VERBATIM
    )
    add_custom_target(hip_cvt_narrow_test_target ALL DEPENDS ${CVT_NARROW_BIN})
    add_dependencies(hip_cvt_narrow_test_target rocjitsu_bin rocjitsu_shared)
    if(RJ_INSTALL_TESTS)
        rj_install_test_program(${CVT_NARROW_BIN})
    endif()

    rj_add_test(
        NAME "CvtNarrow.AllTests"
        COMMAND ${RJ_CLI} --config ${RJ_CDNA4_KMD_CONFIG} -- ${CVT_NARROW_BIN}
        TIMEOUT 300
        INSTALL_COMMAND
            "\${RJ_INSTALLED_BINDIR}/rocjitsu"
            "--config"
            "\${RJ_INSTALLED_CONFIG_DIR}/${RJ_CDNA4_KMD_CONFIG_NAME}"
            "--"
            "bin/hip_cvt_narrow_test"
    )
endif()
