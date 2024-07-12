set(TEST_SRC
    execution_control_common.cc
    hipFuncGetAttributes.cc
    hipLaunchKernel.cc
    hipLaunchCooperativeKernel.cc
    hipLaunchCooperativeKernelMultiDevice.cc
)

if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC ${TEST_SRC}
        hipExtLaunchKernel.cc
        hipExtLaunchMultiKernelMultiDevice.cc
        launch_api.cc
        hipGetProcAddressLaunchCbExecCtrlApis.cc
    )
else()
    # These functions are currently unimplemented on AMD
    set(TEST_SRC ${TEST_SRC}
        hipFuncSetCacheConfig.cc
        hipFuncSetSharedMemConfig.cc
        hipFuncSetAttribute.cc
    )
endif()

hip_add_exe_to_target(NAME ExecutionControlTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      COMPILE_OPTIONS -std=c++17)
