# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipModuleLoadUnload.cc
)

hip_add_exe_to_target(NAME module_stress
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME stress_test
                      LINKER_LIBS hiprtc)
