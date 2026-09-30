# Native filename follows the current ROCr 1.21.0 baseline, independently of
# the Rust package version. Cargo's underscore-named artifact stays internal.
runtime_rust_library(runtime_hsa_shared libhsa hsa_runtime64 SHARED libhsa-runtime64.so.1.21.0)
set_target_properties(runtime_hsa_shared PROPERTIES IMPORTED_SONAME libhsa-runtime64.so.1)
add_library(ROCm::hsa_shared ALIAS runtime_hsa_shared)
target_link_libraries(runtime_hsa_shared INTERFACE ROCm::runtime_headers)

if(BUILD_TESTING)
  add_executable(hsa_smoke tests/abi/native_c_smoke.c)
  target_compile_features(hsa_smoke PRIVATE c_std_11)
  target_link_libraries(hsa_smoke PRIVATE ROCm::hsa_shared)
  add_test(NAME hsa.smoke COMMAND hsa_smoke)
  set_tests_properties(hsa.smoke PROPERTIES LABELS "host;abi")
endif()
