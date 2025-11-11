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

add_executable(hipDeviceGetP2PAttribute_exe EXCLUDE_FROM_ALL hipDeviceGetP2PAttribute_exe.cc)
set_source_files_properties(hipDeviceGetP2PAttribute_exe.cc PROPERTIES LANGUAGE HIP)
set_target_properties(hipDeviceGetP2PAttribute_exe PROPERTIES LINKER_LANGUAGE HIP)
target_link_libraries(hipDeviceGetP2PAttribute_exe hip::host hip::device)

set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipDeviceGetP2PAttribute_exe)

hip_add_exe_to_target(NAME p2pTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

add_dependencies(build_tests hipDeviceGetP2PAttribute_exe)
