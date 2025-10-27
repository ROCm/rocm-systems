# AMD Tests
set(TEST_SRC
    printfFlags.cc
    printfLength.cc
    printfSpecifiers.cc
    printfFlagsNonHost.cc
    printfSpecifiersNonHost.cc
    printfHost.cc)

if(HIP_PLATFORM MATCHES "nvidia")
  set(LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(LINKER_LIBS hiprtc)
endif()

if(UNIX)
  set(AMD_TEST_SRC
    printfNonHost.cc
    hipPrintfManyDevices.cc
    hipPrintfStar.cc
    hipPrintfManyWaves.cc
    hipPrintfWidthPrecision.cc
    hipPrintfBasic.cc
    hipPrintfAltForms.cc)
endif()

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC ${TEST_SRC} ${AMD_TEST_SRC})
endif()

if(HIP_PLATFORM MATCHES "amd")
    set_source_files_properties(printfFlagsNonHost.cc PROPERTIES COMPILE_OPTIONS "-mprintf-kind=buffered")
    set_source_files_properties(printfSpecifiersNonHost.cc PROPERTIES COMPILE_OPTIONS "-mprintf-kind=buffered")
    set_source_files_properties(printfNonHost.cc PROPERTIES COMPILE_OPTIONS "-mprintf-kind=buffered")
endif()

if(HIP_PLATFORM MATCHES "amd")
  hip_add_exe_to_target(NAME PrintfTest
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    LINKER_LIBS ${LINKER_LIBS}
                    PROPERTY CXX_STANDARD 17)
elseif (HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME PrintfTest
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    LINKER_LIBS ${LINKER_LIBS}
                    COMPILE_OPTIONS -std=c++17)
endif()

# Standalone exes
foreach(exe
    printfFlags
    printfLength
    printfSpecifiers
    printfFlagsNonHost
    printfSpecifiersNonHost
)
  hip_add_executable(
    NAME ${exe}_exe
    SOURCE ${exe}_exe.cc
    ADD_TO_TARGET build_tests)
endforeach()
