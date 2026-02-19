set(TEST_SRC
  hipGreenCtxSanity.cc
)

hip_add_exe_to_target(
  NAME GreenContext
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests
)
