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

add_executable(getDeviceCount EXCLUDE_FROM_ALL getDeviceCount_exe.cc)
set_source_files_properties(getDeviceCount_exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS getDeviceCount)
target_link_libraries(getDeviceCount ${GPGPU_LINKER_LIBRARIES})

add_executable(chkUUIDFrmChildProc_Exe EXCLUDE_FROM_ALL chkUUIDFrmChildProc_Exe.cc)
add_executable(chkUUIDInGrandChild_Exe EXCLUDE_FROM_ALL chkUUIDInGrandChild_Exe.cc)
add_executable(setuuidGetDevCount EXCLUDE_FROM_ALL setuuidGetDevCount_Exe.cc)

set_source_files_properties(chkUUIDFrmChildProc_Exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
set_source_files_properties(chkUUIDInGrandChild_Exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
set_source_files_properties(setuuidGetDevCount_Exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
target_link_libraries(chkUUIDFrmChildProc_Exe ${GPGPU_LINKER_LIBRARIES})
target_link_libraries(chkUUIDInGrandChild_Exe ${GPGPU_LINKER_LIBRARIES})
target_link_libraries(setuuidGetDevCount ${GPGPU_LINKER_LIBRARIES})

set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS
             chkUUIDFrmChildProc_Exe
             chkUUIDInGrandChild_Exe
             setuuidGetDevCount)

if(UNIX)
  add_executable(getUUIDfrmRocinfo EXCLUDE_FROM_ALL getUUIDfrmRocinfo_Exe.cc)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS getUUIDfrmRocinfo)
  add_dependencies(build_tests getUUIDfrmRocinfo)
  set_source_files_properties(getUUIDfrmRocinfo_Exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
endif()
add_executable(multipleUUID EXCLUDE_FROM_ALL multipleUUID_Exe.cc)
add_executable(setEnvInChildProc EXCLUDE_FROM_ALL setEnvInChildProc_Exe.cc)
add_executable(uuidList EXCLUDE_FROM_ALL uuidList.cc)

set_source_files_properties(multipleUUID_Exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
set_source_files_properties(setEnvInChildProc_Exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
set_source_files_properties(uuidList.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
target_link_libraries(multipleUUID ${GPGPU_LINKER_LIBRARIES})
target_link_libraries(setEnvInChildProc ${GPGPU_LINKER_LIBRARIES})
target_link_libraries(uuidList ${GPGPU_LINKER_LIBRARIES})


set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS
             multipleUUID
             setEnvInChildProc
             uuidList)
#Disabled below two executable due to the defect ticket SWDEV-467665
if(0)
add_executable(passUUIDToGrandChild_Exe EXCLUDE_FROM_ALL passUUIDToGrandChild_Exe.cc)
add_executable(ResetUUIDInChild_Exe EXCLUDE_FROM_ALL ResetUUIDInChild_Exe.cc)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS
             passUUIDToGrandChild_Exe
             ResetUUIDInChild_Exe)
endif()
hip_add_exe_to_target(NAME DeviceTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
add_dependencies(build_tests getDeviceCount chkUUIDFrmChildProc_Exe chkUUIDInGrandChild_Exe setuuidGetDevCount multipleUUID setEnvInChildProc uuidList)
#Disabled below two executable due to the defect ticket SWDEV-467665
if(0)
add_dependencies(build_tests passUUIDToGrandChild_Exe ResetUUIDInChild_Exe)
endif()

if(HIP_PLATFORM MATCHES "amd")
  add_executable(hipDeviceSetGetScratchExe EXCLUDE_FROM_ALL hipDeviceSetGetScratchExe.cc)
  add_dependencies(DeviceTest hipDeviceSetGetScratchExe)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipDeviceSetGetScratchExe)
  set_source_files_properties(hipDeviceSetGetScratchExe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
endif()

