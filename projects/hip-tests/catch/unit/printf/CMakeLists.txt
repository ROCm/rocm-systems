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
  set(LINKER_LIBS hiprtc::hiprtc)
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
                    LINKER_LIBS ${LINKER_LIBS})
endif()

if(UNIX)
file(GLOB NEGATIVE_TEST_SRC
      printf_negative_kernels.cc)

file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})

#add_test(NAME Unit_Printf_Negative
#         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
#         ./src ${HIP_PLATFORM} ${HIP_PATH}
#         printf_negative_kernels.cc 11)
endif()

# Standalone exes
add_executable(printfFlags_exe EXCLUDE_FROM_ALL printfFlags_exe.cc)
add_executable(printfLength_exe EXCLUDE_FROM_ALL printfLength_exe.cc)
add_executable(printfSpecifiers_exe EXCLUDE_FROM_ALL printfSpecifiers_exe.cc)
add_executable(printfFlagsNonHost_exe EXCLUDE_FROM_ALL printfFlagsNonHost_exe.cc)
add_executable(printfSpecifiersNonHost_exe EXCLUDE_FROM_ALL printfSpecifiersNonHost_exe.cc)

set_source_files_properties(printfFlags_exe.cc PROPERTIES LANGUAGE HIP)
set_source_files_properties(printfLength_exe.cc PROPERTIES LANGUAGE HIP)
set_source_files_properties(printfSpecifiers_exe.cc PROPERTIES LANGUAGE HIP)
set_source_files_properties(printfFlagsNonHost_exe.cc PROPERTIES LANGUAGE HIP)
set_source_files_properties(printfSpecifiersNonHost_exe.cc PROPERTIES LANGUAGE HIP)
set_target_properties(printfFlags_exe PROPERTIES LINKER_LANGUAGE HIP)
set_target_properties(printfLength_exe PROPERTIES LINKER_LANGUAGE HIP)
set_target_properties(printfSpecifiers_exe PROPERTIES LINKER_LANGUAGE HIP)
set_target_properties(printfFlagsNonHost_exe PROPERTIES LINKER_LANGUAGE HIP)
set_target_properties(printfSpecifiersNonHost_exe PROPERTIES LINKER_LANGUAGE HIP)
target_link_libraries(printfFlags_exe hip::host hip::device)
target_link_libraries(printfLength_exe hip::host hip::device)
target_link_libraries(printfSpecifiers_exe hip::host hip::device)
target_link_libraries(printfFlagsNonHost_exe hip::host hip::device)
target_link_libraries(printfSpecifiersNonHost_exe hip::host hip::device)

set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS
             printfFlags_exe
             printfLength_exe
             printfSpecifiers_exe
             printfFlagsNonHost_exe
             printfSpecifiersNonHost_exe)

add_dependencies(build_tests printfFlags_exe)
add_dependencies(build_tests printfLength_exe)
add_dependencies(build_tests printfSpecifiers_exe)
add_dependencies(build_tests printfFlagsNonHost_exe)
add_dependencies(build_tests printfSpecifiersNonHost_exe)

