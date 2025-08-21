set(TEST_SRC
    loadlib.cc
)

hip_add_exe_to_target(NAME LibraryTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc)