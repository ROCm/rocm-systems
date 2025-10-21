# Common Tests - Test independent of all platforms
set(TEST_SRC
    hipChooseDevice.cc
    hipDeviceComputeCapability.cc
    hipDeviceGetByPCIBusId.cc
    hipDeviceGetLimit_old.cc
    hipDeviceGetName.cc
    hipDeviceGetPCIBusId.cc
    hipDeviceSetGetCacheConfig.cc
    hipDeviceSynchronize.cc
    hipDeviceTotalMem.cc
    hipGetDeviceAttribute.cc
    hipGetDeviceCount.cc
    hipGetDeviceProperties.cc
    hipRuntimeGetVersion.cc
    hipGetSetDeviceFlags.cc
    hipSetGetDevice.cc
    hipDeviceGetUuid.cc
    hipDeviceAPUCheck.cc
    hipDeviceGetDefaultMemPool.cc
    hipDeviceCanAccessPeer.cc
    hipDeviceEnableDisablePeerAccess.cc
    hipExtGetLinkTypeAndHopCount.cc
    hipDeviceSetLimit_old.cc
    hipDeviceSetGetLimit.cc
    hipDeviceSetGetSharedMemConfig.cc
    hipDeviceReset.cc
    hipDeviceSetGetMemPool.cc
    hipInit.cc
    hipDriverGetVersion.cc
    hipSetValidDevices.cc
)

if(HIP_PLATFORM MATCHES "amd" AND BUILD_SHARED_LIBS)
  set(AMD_SRC
      hipGetProcAddressDevMgmt.cc
      hipGetDriverEntryPoint.cc)
  if(UNIX)
    set(AMD_SRC ${AMD_SRC} hipGetProcAddressIpcApis.cc)
  endif()
  set(TEST_SRC ${TEST_SRC} ${AMD_SRC})
endif()

if(UNIX)
    set(TEST_SRC ${TEST_SRC}
                 hipIpcOpenMemHandle.cc
                 hipIpcGetMemHandle.cc
                 hipIpcCloseMemHandle.cc
       )
endif()

set(DEVICE_STANDALONE_EXES
    getDeviceCount
    chkUUIDFrmChildProc
    chkUUIDInGrandChild
    setuuidGetDevCount
    multipleUUID
    setEnvInChildProc
    uuidList
)

if(UNIX)
  list(APPEND DEVICE_STANDALONE_EXES getUUIDfrmRocinfo)
endif()

# Disabled executables due to defect ticket SWDEV-467665
if(0)
  list(APPEND DEVICE_STANDALONE_EXES
      passUUIDToGrandChild
      ResetUUIDInChild
  )
endif()

if(HIP_PLATFORM MATCHES "amd")
  list(APPEND DEVICE_STANDALONE_EXES hipDeviceSetGetScratch)
endif()

foreach(EXE_NAME ${DEVICE_STANDALONE_EXES})
  hip_add_executable(NAME ${EXE_NAME}
                     SOURCE ${EXE_NAME}.cc
                     ADD_TO_TARGET build_tests)
endforeach()

hip_add_exe_to_target(NAME DeviceTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
