# Common Tests
set(TEST_SRC
  childMalloc.cc
  hipDeviceComputeCapabilityMproc.cc
  hipDeviceGetPCIBusIdMproc.cc
  hipDeviceTotalMemMproc.cc
  hipGetDeviceAttributeMproc.cc
  hipGetDeviceCountMproc.cc
  hipGetDevicePropertiesMproc.cc
  hipSetGetDeviceMproc.cc
  hipIpcMemAccessTest.cc
  hipMallocConcurrencyMproc.cc
  hipMemCoherencyTstMProc.cc
  hipIpcEventHandle.cc
  deviceAllocationMproc.cc
  hipNoGpuTsts.cc
  hipMemGetInfoMProc.cc
)

if(UNIX)
  hip_add_device_code(
    TARGET dummy_kernel
    SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/dummy_kernel.cpp)

  add_dependencies(build_tests dummy_kernel)
endif()

# the last argument linker libraries is required for this test but optional to the function
if(HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME MultiProc
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS nvrtc)
  set_target_properties(MultiProc PROPERTIES COMPILE_FLAGS -arch=sm_70)
elseif(HIP_PLATFORM MATCHES "amd")
  hip_add_exe_to_target(NAME MultiProc
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc)
endif()
