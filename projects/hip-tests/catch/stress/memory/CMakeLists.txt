# Common Tests - Test independent of all platforms
set(TEST_SRC
    memcpy.cc
    hipMemcpyMThreadMSize.cc
    hipMallocManagedStress.cc
    hipMemPrftchAsyncStressTst.cc
    hipHostMallocStress.cc
    hipHostRegisterStress.cc
)

if(UNIX)
  set(TEST_SRC ${TEST_SRC}
               hipHmmOvrSubscriptionTst.cc)
  hip_add_executable(NAME hold_memory
                     SOURCE hold_memory.cc
                     ADD_TO_TARGET stress_test)
endif()

hip_add_exe_to_target(NAME memory_stress
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME stress_test)
