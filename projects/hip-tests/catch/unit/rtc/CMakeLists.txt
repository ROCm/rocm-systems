# Common Tests - Test independent of all platforms
set(TEST_SRC
    saxpy.cc
    warpsize.cc
    hipRtcFunctional.cc
    hipStreamCaptureRtc.cc
    includepath.cc
    hipRTCDeviceMalloc.cc
    hiprtcGetLoweredName.cc
)

# AMD only tests
set(AMD_TEST_SRC
    customOptions.cc
    hipRtcBfloat16.cc
    linker.cc
    shfl.cc
    shfl_sync.cc
    rtc_reduce.cc
    stdheaders.cc
    hiprtc_MathConstants_HeaderTst.cc
    hiprtc_VectorTypes_HeaderTst.cc
    hiprtc_MathFunctions_HeaderTst.cc
    hiprtc_fp16_HeaderTst.cc
    hiprtc_TextureTypes_HeaderTst.cc
    hipRtcComplexHeader.cc
    hipRtcPtrdiff_t.cc
    hiprtc_Bitcode_UndefinedFn.cc
    hiprtc_bfloat16_HeaderTst.cc
)

add_custom_target(copyRtcHeaders ALL
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/saxpy.h
  ${CMAKE_CURRENT_BINARY_DIR}/saxpy.h
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/headers
  ${CMAKE_CURRENT_BINARY_DIR}/headers)

file(GLOB INSTALL_HEADER_FILES "./headers/*")
set_property(GLOBAL APPEND PROPERTY G_INSTALL_HEADER_FILES ${INSTALL_HEADER_FILES})
set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/saxpy.h)

if(UNIX)
   set(AMD_TEST_SRC ${AMD_TEST_SRC}
       RtcFunctions.cpp
       RtcUtility.cpp
       hiprtcComplrOptnTesting.cc)
endif()

if(HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME RTC
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
  hip_add_exe_to_target(NAME RTC
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc)
endif()

if(HIP_PLATFORM MATCHES "nvidia")
  set(RTC_LIB nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(RTC_LIB hiprtc)
endif()

hip_add_executable(NAME ChkPtrdiff_t_Exe
                   SOURCE ChkPtrdiff_t_Exe.cc
                   ADD_TO_TARGET build_tests
                   LINKER_LIBS ${RTC_LIB})

add_dependencies(build_tests copyRtcHeaders)
