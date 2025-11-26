# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipMemsetAsyncMultiThread.cc
  hipMultiThreadDevice.cc
  hipMultiThreadStreams1.cc
  hipMultiThreadStreams2.cc
)

hip_add_exe_to_target(NAME MultiThreadTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
set_target_properties(MultiThreadTest PROPERTIES LINKER_LANGUAGE ${GPGPU_LANGUAGE})
target_link_libraries(MultiThreadTest ${GPGPU_LINKER_LIBRARIES})
