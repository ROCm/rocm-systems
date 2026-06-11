set(TEST_SRC
  hipGreenCtxCreate.cc
  hipExecutionCtxStreamCreate.cc
  hipExecutionCtxRecordEvent.cc
  hipExecutionCtxResourceSplit.cc
)

hip_add_exe_to_target(
  NAME ExecutionContext
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests
)

set_target_properties(ExecutionContext PROPERTIES LINKER_LANGUAGE ${GPGPU_LANGUAGE})
target_link_libraries(ExecutionContext ${GPGPU_LINKER_LIBRARIES})
