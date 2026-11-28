# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipClassKernel.cc
)

hip_add_exe_to_target(NAME CompilerTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC
    hipSquare.cc
  )

  set_source_files_properties(hipSquare.cc PROPERTIES COMPILE_FLAGS "--offload-compress")

  hip_add_exe_to_target(NAME SimpleCompressedCodeObjectTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

  include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/hip-tests-generic-targets.cmake")
  hip_tests_get_generic_target_options(OFFLOAD_ARCH_GENERIC_STR GENERIC_TARGETS)
  list(JOIN GENERIC_TARGETS "," _generic_target_ids)
  # Pass the same image inventory to every consumer, including standalone tests.
  add_compile_definitions(HIP_TESTS_GENERIC_TARGETS="${_generic_target_ids}")

  # ASAN is only supported on xnack+ offload archs; clang skips it on the other
  # generic targets and emits an -Woption-ignored warning. Suppress it under device
  # ASAN so -Werror does not fail the build.
  set(ASAN_GENERIC_TARGET_FLAGS)
  if(ENABLE_SANITIZER STREQUAL "ASAN")
    set(ASAN_GENERIC_TARGET_FLAGS -Wno-option-ignored)
  endif()

  set(DISABLE_GENERIC_TARGET_ONLY)

  # Clear the architecture flags inherited from the common HIP build before
  # adding generic targets, so an exact device target cannot mask generic
  # code-object selection in these helper executables.
  if(BUILD_SHARED_LIBS) # Build hipGenericTargetOnly for shared libs only
    set(GENERIC_TARGET_ONLY_EXE hipSquareGenericTargetOnly)
    set(GENERIC_TARGET_ONLY_COMPRESSED_EXE hipSquareGenericTargetOnlyCompressed)

    set(LIBFS)
    if(NOT WIN32)
      set(LIBFS -lstdc++fs)
    endif()

    set_source_files_properties(hipSquareGenericTarget.cc PROPERTIES LANGUAGE HIP)
    add_executable(${GENERIC_TARGET_ONLY_EXE}
      hipSquareGenericTarget.cc
      $<TARGET_OBJECTS:Main_Object>)
    set_target_properties(${GENERIC_TARGET_ONLY_EXE} PROPERTIES HIP_ARCHITECTURES OFF)
    target_compile_options(${GENERIC_TARGET_ONLY_EXE} PUBLIC
      -w --no-offload-arch=all ${OFFLOAD_ARCH_GENERIC_STR})
    target_link_libraries(${GENERIC_TARGET_ONLY_EXE} hip::host hip::device)
    target_compile_definitions(${GENERIC_TARGET_ONLY_EXE} PRIVATE NO_GENERIC_TARGET_ONLY_TEST)
    if(NOT USE_PREBUILT_CATCH)
      target_link_libraries(${GENERIC_TARGET_ONLY_EXE} Catch2::Catch2WithMain)
    endif()

    add_executable(${GENERIC_TARGET_ONLY_COMPRESSED_EXE}
      hipSquareGenericTarget.cc
      $<TARGET_OBJECTS:Main_Object>)
    target_compile_definitions(${GENERIC_TARGET_ONLY_COMPRESSED_EXE} PRIVATE GENERIC_COMPRESSED NO_GENERIC_TARGET_ONLY_TEST)
    set_target_properties(${GENERIC_TARGET_ONLY_COMPRESSED_EXE} PROPERTIES HIP_ARCHITECTURES OFF)
    target_compile_options(${GENERIC_TARGET_ONLY_COMPRESSED_EXE} PUBLIC
      --offload-compress -w --no-offload-arch=all ${OFFLOAD_ARCH_GENERIC_STR})
    target_link_libraries(${GENERIC_TARGET_ONLY_COMPRESSED_EXE} hip::host hip::device)
    if(NOT USE_PREBUILT_CATCH)
      target_link_libraries(${GENERIC_TARGET_ONLY_COMPRESSED_EXE} Catch2::Catch2WithMain)
    endif()
    if (WIN32)
      set(GENERIC_TARGET_ONLY_EXE ${GENERIC_TARGET_ONLY_EXE}.exe)
      set(GENERIC_TARGET_ONLY_COMPRESSED_EXE ${GENERIC_TARGET_ONLY_COMPRESSED_EXE}.exe)
    endif()
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/${GENERIC_TARGET_ONLY_EXE})
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/${GENERIC_TARGET_ONLY_COMPRESSED_EXE})
  else()
    set(DISABLE_GENERIC_TARGET_ONLY "-DNO_GENERIC_TARGET_ONLY_TEST")
  endif()

  # Build hipSquareGenericTarget to cover generic targets and the specific target
  set(TEST_SRC
    hipSquareGenericTarget.cc
  )
  hip_add_exe_to_target(NAME hipSquareGenericTargetDefault
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
  set_target_properties(hipSquareGenericTargetDefault PROPERTIES COMPILE_FLAGS "-mcode-object-version=6 ${DISABLE_GENERIC_TARGET_ONLY}")
  target_compile_options(hipSquareGenericTargetDefault PUBLIC ${ASAN_GENERIC_TARGET_FLAGS} ${OFFLOAD_ARCH_GENERIC_STR})

  hip_add_exe_to_target(NAME hipSquareGenericTargetCompressed
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
  set_target_properties(hipSquareGenericTargetCompressed PROPERTIES COMPILE_FLAGS " -DGENERIC_COMPRESSED ${DISABLE_GENERIC_TARGET_ONLY} -mcode-object-version=6 --offload-compress")
  target_compile_options(hipSquareGenericTargetCompressed PUBLIC ${ASAN_GENERIC_TARGET_FLAGS} ${OFFLOAD_ARCH_GENERIC_STR})

  add_dependencies(hipSquareGenericTargetDefault hipSquareGenericTargetCompressed)
  if(BUILD_SHARED_LIBS)
    add_dependencies(hipSquareGenericTargetDefault hipSquareGenericTargetOnly)
    add_dependencies(hipSquareGenericTargetDefault hipSquareGenericTargetOnlyCompressed)
  endif()

  # Build hipSpirvTest for amdgcnspirv only
  set(SPIRV_TEST_SRC
    hipSpirvTest.cc
    ../../hipTestMain/hip_test_context.cc
    ../../hipTestMain/main.cc
    ../../hipTestMain/hip_test_level.cc)
  set_source_files_properties(${SPIRV_TEST_SRC} PROPERTIES LANGUAGE HIP)
  add_executable(hipSpirvTest EXCLUDE_FROM_ALL ${SPIRV_TEST_SRC})
  set_target_properties(hipSpirvTest PROPERTIES LINKER_LANGUAGE HIP)
  target_compile_options(hipSpirvTest PRIVATE --no-offload-arch=all --offload-arch=amdgcnspirv)
  target_link_options(hipSpirvTest PRIVATE --no-offload-arch=all --offload-arch=amdgcnspirv)
  target_link_libraries(hipSpirvTest PRIVATE Catch2::Catch2 hip::host)
  target_include_directories(hipSpirvTest PRIVATE ${CATCH_INCLUDE_BINARY_DIR})
  add_dependencies(hipSpirvTest hip_tests_config)
  add_dependencies(build_tests hipSpirvTest)
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS hipSpirvTest)
  catch_discover_tests(hipSpirvTest
    DISCOVERY_MODE PRE_TEST
    ADD_TAGS_AS_LABELS
    PROPERTIES SKIP_RETURN_CODE 4)
  file(GLOB SPIRV_TEST_CTEST_FILES "${CMAKE_CURRENT_BINARY_DIR}/hipSpirvTest-*_include.cmake")
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_CTEST_INCLUDE_FILES ${SPIRV_TEST_CTEST_FILES})
endif()
