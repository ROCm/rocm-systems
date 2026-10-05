# Native ABI version follows the ROCr baseline, independently of Cargo's version.
project(ROCmRuntimeHSA VERSION 1.21.0 LANGUAGES C)

runtime_rust_library(runtime_hsa_shared
  PACKAGE libhsa LIBRARY hsa_runtime64 TYPE SHARED OUTPUT_NAME hsa-runtime64
  VERSION "${PROJECT_VERSION}" SOVERSION "${PROJECT_VERSION_MAJOR}"
  SONAME_ENV ROCM_RUNTIME_HSA_SONAME
)
add_library(rocm_runtime::hsa_shared ALIAS runtime_hsa_shared)
target_link_libraries(
    runtime_hsa_shared
    INTERFACE rocm_runtime::runtime_headers
)

if(BUILD_TESTING)
    add_executable(hsa_smoke tests/abi/native_c_smoke.c)
    target_compile_features(hsa_smoke PRIVATE c_std_11)
    target_link_libraries(hsa_smoke PRIVATE rocm_runtime::hsa_shared)
    add_test(NAME hsa.smoke COMMAND hsa_smoke)
    set_tests_properties(hsa.smoke PROPERTIES LABELS "host;abi")
endif()
