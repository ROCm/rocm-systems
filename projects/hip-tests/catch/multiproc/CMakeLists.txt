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

string(TOUPPER "${CMAKE_BUILD_TYPE}" _HIP_BUILD_TYPE_UPPER)
separate_arguments(HIP_DEVICE_BUILD_FLAGS_NO_ARCH NATIVE_COMMAND
  "${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_${_HIP_BUILD_TYPE_UPPER}}")
list(FILTER HIP_DEVICE_BUILD_FLAGS_NO_ARCH EXCLUDE REGEX "--offload-arch=")

if(UNIX)
  if (HIP_PLATFORM MATCHES "amd")
    add_custom_target(dummy_kernel.code
                      COMMAND ${CMAKE_HIP_COMPILER} ${HIP_DEVICE_BUILD_FLAGS_NO_ARCH}
                      --cuda-device-only -x hip ${CMAKE_CURRENT_SOURCE_DIR}/dummy_kernel.cpp
                      -o ${CMAKE_CURRENT_BINARY_DIR}/../multiproc/dummy_kernel.code
                      -I${HIP_INCLUDE_DIR}
                      -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)
  else()
    add_custom_target(dummy_kernel.code
                      COMMAND ${CMAKE_HIP_COMPILER}
                      --fatbin -x cu ${CMAKE_CURRENT_SOURCE_DIR}/dummy_kernel.cpp
                      -o ${CMAKE_CURRENT_BINARY_DIR}/../multiproc/dummy_kernel.code
                      -I${HIP_INCLUDE_DIR}
                      -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

  endif()
endif()
set_property(GLOBAL APPEND PROPERTY
              G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/dummy_kernel.code)

# the last argument linker libraries is required for this test but optional to the function
if(HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME MultiProc
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS nvrtc)
  set_target_properties(MultiProc PROPERTIES COMPILE_FLAGS -arch=compute_75)
elseif(HIP_PLATFORM MATCHES "amd")
  hip_add_exe_to_target(NAME MultiProc
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc::hiprtc)
endif()

if(UNIX)
  add_dependencies(build_tests dummy_kernel.code)
endif()
