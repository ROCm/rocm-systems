# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

project(ROCmRuntimeHSA LANGUAGES C)

if(BUILD_TESTING)
    add_executable(hsa_smoke tests/abi/native_c_smoke.c)
    target_compile_features(hsa_smoke PRIVATE c_std_11)
    target_link_libraries(hsa_smoke PRIVATE rocm_runtime::hsa_shared)
    add_test(NAME hsa.smoke COMMAND hsa_smoke)
    set_tests_properties(hsa.smoke PROPERTIES LABELS "host;abi")

    add_executable(hsa_fork_smoke tests/abi/fork_smoke.c)
    target_compile_features(hsa_fork_smoke PRIVATE c_std_11)
    target_link_libraries(hsa_fork_smoke PRIVATE rocm_runtime::hsa_shared)
    add_test(NAME hsa.fork_smoke COMMAND hsa_fork_smoke)
    set_tests_properties(
        hsa.fork_smoke
        PROPERTIES
            LABELS "host;abi"
            SKIP_RETURN_CODE 77
            ENVIRONMENT_MODIFICATION "HSA_TOOLS_LIB=unset:"
    )

    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        add_library(hsa_fork_tool MODULE tests/abi/fork_tool.cpp)
        target_compile_features(hsa_fork_tool PRIVATE cxx_std_17)
        target_link_libraries(hsa_fork_tool PRIVATE rocm_runtime::runtime_headers)
        add_dependencies(hsa_fork_smoke hsa_fork_tool)
        add_test(NAME hsa.fork_tool_smoke COMMAND hsa_fork_smoke tool)
        set_tests_properties(
            hsa.fork_tool_smoke
            PROPERTIES
                LABELS "host;abi"
                SKIP_RETURN_CODE 77
                ENVIRONMENT "HSA_TOOLS_LIB=$<TARGET_FILE:hsa_fork_tool>"
        )
    endif()
endif()
