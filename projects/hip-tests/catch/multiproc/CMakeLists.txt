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
  add_custom_target(dummy_kernel.code
                      COMMAND ${CMAKE_CXX_COMPILER}
                      --genco ${CMAKE_CURRENT_SOURCE_DIR}/dummy_kernel.cpp
                      -o ${CMAKE_CURRENT_BINARY_DIR}/../multiproc/dummy_kernel.code
                      -I${HIP_PATH}/include/ -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                      --hip-path=${HIP_PATH})
    set_property(GLOBAL APPEND PROPERTY
                 G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/dummy_kernel.code)
endif()

# the last argument linker libraries is required for this test but optional to the function
if(HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME MultiProc
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS nvrtc)
  # sm_70 is not supported in CUDA 13.0. On CUDA 13.0+ it would default to sm_75.
  if (CUDA_VERSION VERSION_LESS 13.0)
    set_target_properties(MultiProc PROPERTIES COMPILE_FLAGS -arch=sm_70)
  endif()
elseif(HIP_PLATFORM MATCHES "amd")
  hip_add_exe_to_target(NAME MultiProc
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc)
endif()

if(UNIX)
  add_dependencies(build_tests dummy_kernel.code)
endif()
