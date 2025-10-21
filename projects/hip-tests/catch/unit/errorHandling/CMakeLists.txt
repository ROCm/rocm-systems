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

if(HIP_PLATFORM MATCHES "amd")
  set(AMD_SRC
      hipExtGetLastError.cc
  )
  set(TEST_SRC ${TEST_SRC} ${AMD_SRC})
endif()

hip_add_executable(NAME hipGetLastErrorEnv_Exe
                   SOURCE hipGetLastErrorEnv_Exe.cc)

hip_add_executable(NAME hipPeekAtLastErrorEnv_Exe
                   SOURCE hipPeekAtLastErrorEnv_Exe.cc)

hip_add_exe_to_target(NAME ErrorHandlingTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++17)

add_dependencies(ErrorHandlingTest hipGetLastErrorEnv_Exe hipPeekAtLastErrorEnv_Exe)
