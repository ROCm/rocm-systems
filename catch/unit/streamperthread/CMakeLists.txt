# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipStreamPerThread_Basic.cc
  hipStreamPerThread_Event.cc
  hipStreamPerThread_MultiThread.cc
  hipStreamPerThread_DeviceReset.cc
  hipStreamPerThrdTsts.cc
  hipStreamPerThrdCompilerOptn.cc
)

if(HIP_PLATFORM MATCHES "amd")
   set(TEST_SRC ${TEST_SRC}
       hipGetProcAddressSptApis.cc)
endif()

if(HIP_PLATFORM MATCHES "amd")
        set_source_files_properties(hipStreamPerThrdCompilerOptn.cc PROPERTIES COMPILE_OPTIONS "-fgpu-default-stream=per-thread")
endif()

if(HIP_PLATFORM MATCHES "nvidia")
        set_source_files_properties(hipStreamPerThrdCompilerOptn.cc PROPERTIES COMPILE_OPTIONS "--default-stream=per-thread")
endif()

hip_add_exe_to_target(NAME StreamPerThreadTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
