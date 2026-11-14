# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

project(ROCmRuntimeHSA LANGUAGES C)

if(BUILD_TESTING)
    add_executable(hsa_smoke tests/abi/native_c_smoke.c)
    target_compile_features(hsa_smoke PRIVATE c_std_11)
    target_link_libraries(hsa_smoke PRIVATE rocm_runtime::hsa_shared)
    add_test(NAME hsa.smoke COMMAND hsa_smoke)
    set_tests_properties(hsa.smoke PROPERTIES LABELS "host;abi")
    if(WIN32)
        set_tests_properties(hsa.smoke PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${_runtime_library_dir}")
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(hsa_loader_extensions tests/abi/loader_extensions.c)
        target_compile_features(hsa_loader_extensions PRIVATE c_std_11)
        target_link_libraries(hsa_loader_extensions PRIVATE rocm_runtime::hsa_shared)
        add_test(NAME hsa.loader_extensions COMMAND hsa_loader_extensions)
        set_tests_properties(hsa.loader_extensions PROPERTIES
            LABELS "host;abi" SKIP_RETURN_CODE 77 TIMEOUT 30
            ENVIRONMENT "ROCR_VISIBLE_DEVICES="
            ENVIRONMENT_MODIFICATION "HSA_TOOLS_LIB=unset:")
        add_executable(hsa_cpu_signals tests/abi/cpu_signals.c)
        target_compile_features(hsa_cpu_signals PRIVATE c_std_11)
        target_link_libraries(hsa_cpu_signals PRIVATE rocm_runtime::hsa_shared)
        add_test(NAME hsa.cpu_signals COMMAND hsa_cpu_signals)
        set_tests_properties(hsa.cpu_signals PROPERTIES
            LABELS "host;abi" SKIP_RETURN_CODE 77 TIMEOUT 30
            ENVIRONMENT "ROCR_VISIBLE_DEVICES="
            ENVIRONMENT_MODIFICATION "HSA_TOOLS_LIB=unset:")
        add_test(NAME hsa.profiler_publication
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/abi/check_profiler.py"
                "$<TARGET_FILE:runtime_shared_image>" --cxx "${CMAKE_CXX_COMPILER}")
        set_tests_properties(hsa.profiler_publication PROPERTIES
            LABELS "host;abi" SKIP_RETURN_CODE 77 TIMEOUT 60)
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

        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|aarch64|arm64)$")
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
endif()
