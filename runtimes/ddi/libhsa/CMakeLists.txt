# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

project(ROCmRuntimeHSA LANGUAGES C)

if(BUILD_TESTING)
    add_executable(hsa_smoke tests/abi/native_c_smoke.c)
    target_compile_features(hsa_smoke PRIVATE c_std_11)
    target_link_libraries(hsa_smoke PRIVATE rocm_runtime::hsa_shared)
    add_test(NAME hsa.smoke COMMAND hsa_smoke)
    set_tests_properties(hsa.smoke PROPERTIES LABELS "host;abi")
endif()
