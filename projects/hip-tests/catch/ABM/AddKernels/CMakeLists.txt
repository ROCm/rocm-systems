# Common Tests - Test independent of all platforms
set(TEST_SRC
    add.cc
)

hip_add_exe_to_target(NAME ABMAddKernels
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

set_target_properties(ABMAddKernels PROPERTIES LINKER_LANGUAGE ${GPGPU_LANGUAGE})
target_link_libraries(ABMAddKernels ${GPGPU_LINKER_LIBRARIES})
