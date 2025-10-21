set(TEST_SRC
    loadlib_rtc.cc
    loadlib_co.cc
    library_negative.cc
)

hip_add_device_code(
  TARGET library_code_load
  SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/library_code_load.cc)

if(HIP_PLATFORM MATCHES "amd")
    hip_add_exe_to_target(NAME LibraryTests
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS hiprtc)
else()
    hip_add_exe_to_target(NAME LibraryTests
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS nvrtc)
endif()

add_dependencies(LibraryTests library_code_load)
