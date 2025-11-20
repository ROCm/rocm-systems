# Common Tests - Test independent of all platforms
set(TEST_SRC
    error_handling_common.cc
    hipGetErrorName.cc
    hipGetErrorString.cc
    hipDrvGetErrorName.cc
    hipDrvGetErrorString.cc
    hipGetLastError.cc
    hipPeekAtLastError.cc
)

if(UNIX)
  set(TEST_SRC
      ${TEST_SRC}
      hipGetLastErrorOnAbort.cc)
endif()

add_executable(hipGetLastErrorEnv_Exe EXCLUDE_FROM_ALL hipGetLastErrorEnv_Exe.cc)
add_executable(hipPeekAtLastErrorEnv_Exe EXCLUDE_FROM_ALL hipPeekAtLastErrorEnv_Exe.cc)

<<<<<<< HEAD
set_source_files_properties(hipGetLastErrorEnv_Exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
set_source_files_properties(hipPeekAtLastErrorEnv_Exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
set_target_properties(hipGetLastErrorEnv_Exe PROPERTIES LINKER_LANGUAGE ${GPGPU_LANGUAGE})
set_target_properties(hipPeekAtLastErrorEnv_Exe PROPERTIES LINKER_LANGUAGE ${GPGPU_LANGUAGE})
=======
set_source_files_properties(hipGetLastErrorEnv_Exe.cc PROPERTIES LANGUAGE HIP)
set_source_files_properties(hipPeekAtLastErrorEnv_Exe.cc PROPERTIES LANGUAGE HIP)
set_target_properties(hipGetLastErrorEnv_Exe PROPERTIES LINKER_LANGUAGE HIP)
set_target_properties(hipPeekAtLastErrorEnv_Exe PROPERTIES LINKER_LANGUAGE HIP)
>>>>>>> b0d0373c081b48173e5c7ae8709f66e38c7045ec
target_link_libraries(hipGetLastErrorEnv_Exe hip::host hip::device)
target_link_libraries(hipPeekAtLastErrorEnv_Exe hip::host hip::device)

if(HIP_PLATFORM MATCHES "amd")
  set(AMD_SRC
      hipExtGetLastError.cc
  )
  set(TEST_SRC ${TEST_SRC} ${AMD_SRC})
endif()
hip_add_exe_to_target(NAME ErrorHandlingTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
<<<<<<< HEAD
if (HIP_PLATFORM MATCHES "nvidia")
	set_target_properties(ErrorHandlingTest PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(ErrorHandlingTest cuda)
endif()
=======
>>>>>>> b0d0373c081b48173e5c7ae8709f66e38c7045ec

add_dependencies(ErrorHandlingTest hipGetLastErrorEnv_Exe hipPeekAtLastErrorEnv_Exe)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipGetLastErrorEnv_Exe hipPeekAtLastErrorEnv_Exe)
