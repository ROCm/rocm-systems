# Common Tests - Test independent of all platforms
# moved hipDeviceGetP2PAttribute.cc from /catch/unit/device to
# /catch/unit/p2p folder and its dependent files.
set(TEST_SRC
    hipDeviceGetP2PAttribute.cc
)

# only for AMD
if(HIP_PLATFORM MATCHES "amd")
  set(AMD_SRC
    hipP2pLinkTypeAndHopFunc.cc
  )
  set(TEST_SRC ${TEST_SRC} ${AMD_SRC})
endif()

set_source_files_properties(hipDeviceGetP2PAttribute.cc PROPERTIES COMPILE_FLAGS -std=c++17)

hip_add_executable(NAME hipDeviceGetP2PAttribute_exe
                   SOURCE hipDeviceGetP2PAttribute_exe.cc
                   ADD_TO_TARGET build_tests)

hip_add_exe_to_target(NAME p2pTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
