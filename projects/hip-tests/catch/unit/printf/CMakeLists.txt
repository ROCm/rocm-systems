# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

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
    hostcallStackSize.cc
    hostcallShutdown.cc
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

hip_add_exe_to_target(NAME PrintfTest
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    LINKER_LIBS ${LINKER_LIBS}
                    PROPERTY CXX_STANDARD 17)

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

set_source_files_properties(printfFlags_exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
set_source_files_properties(printfLength_exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
set_source_files_properties(printfSpecifiers_exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
set_source_files_properties(printfFlagsNonHost_exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
set_source_files_properties(printfSpecifiersNonHost_exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})

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

# Needs the static TLS block in a process of its own, so it cannot join the shared test binary.
if(UNIX AND HIP_PLATFORM MATCHES "amd")
  add_executable(hostcallStackSize_exe EXCLUDE_FROM_ALL hostcallStackSize_exe.cc)
  set_source_files_properties(hostcallStackSize_exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hostcallStackSize_exe)
  add_dependencies(build_tests hostcallStackSize_exe)
endif()

# Exercises process teardown itself, so it needs a process that is allowed to finish exiting.
if(UNIX AND HIP_PLATFORM MATCHES "amd")
  add_executable(hostcallShutdown_exe EXCLUDE_FROM_ALL hostcallShutdown_exe.cc)
  set_source_files_properties(hostcallShutdown_exe.cc PROPERTIES LANGUAGE ${GPGPU_LANGUAGE})
  target_link_libraries(hostcallShutdown_exe pthread)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hostcallShutdown_exe)
  add_dependencies(build_tests hostcallShutdown_exe)
endif()

