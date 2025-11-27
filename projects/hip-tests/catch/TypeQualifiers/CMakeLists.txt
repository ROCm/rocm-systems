# Common Tests
set(TEST_SRC
    hipManagedKeyword.cc
)

hip_add_exe_to_target(NAME TypeQualifiers
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
set_target_properties(TypeQualifiers PROPERTIES LINKER_LANGUAGE ${GPGPU_LANGUAGE})
target_link_libraries(TypeQualifiers ${GPGPU_LINKER_LIBRARIES})
