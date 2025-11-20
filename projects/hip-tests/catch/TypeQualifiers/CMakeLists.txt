# Common Tests
set(TEST_SRC
    hipManagedKeyword.cc
)

hip_add_exe_to_target(NAME TypeQualifiers
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
if(HIP_PLATFORM MATCHES "nvidia")
  set_target_properties(TypeQualifiers PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(TypeQualifiers cuda)
  endif()
