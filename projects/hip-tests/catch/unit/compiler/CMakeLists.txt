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

  set(OFFLOAD_ARCH_GENERIC_STR
    --offload-arch=gfx9-generic
    --offload-arch=gfx9-4-generic:sramecc+:xnack-
    --offload-arch=gfx9-4-generic:sramecc-:xnack-
    --offload-arch=gfx9-4-generic:xnack+
    --offload-arch=gfx10-1-generic
    --offload-arch=gfx10-3-generic
    --offload-arch=gfx11-generic
    --offload-arch=gfx12-generic)

  set(DISABLE_GENERIC_TARGET_ONLY)

  # Build hipSquareGenericTargetOnly to cover generic targets only
  # Because default catch2 build will reference CMAKE_CXX_FLAGS that contains specific targets which will hijack generic
  # target in hip-rt, we have to use custom build to contain generic targets only.
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
    target_compile_options(${GENERIC_TARGET_ONLY_EXE} PUBLIC -w ${OFFLOAD_ARCH_GENERIC_STR})
    target_link_libraries(${GENERIC_TARGET_ONLY_EXE} hip::host hip::device)
    target_compile_definitions(${GENERIC_TARGET_ONLY_EXE} PRIVATE NO_GENERIC_TARGET_ONLY_TEST)
    if(NOT USE_PREBUILT_CATCH)
      target_link_libraries(${GENERIC_TARGET_ONLY_EXE} Catch2::Catch2WithMain)
    endif()

    add_executable(${GENERIC_TARGET_ONLY_COMPRESSED_EXE}
      hipSquareGenericTarget.cc
      $<TARGET_OBJECTS:Main_Object>)
    target_compile_definitions(${GENERIC_TARGET_ONLY_COMPRESSED_EXE} PRIVATE GENERIC_COMPRESSED NO_GENERIC_TARGET_ONLY_TEST)
    target_compile_options(${GENERIC_TARGET_ONLY_COMPRESSED_EXE} PUBLIC --offload-compress -w ${OFFLOAD_ARCH_GENERIC_STR})
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
  target_compile_options(hipSquareGenericTargetDefault PUBLIC ${OFFLOAD_ARCH_GENERIC_STR})

  hip_add_exe_to_target(NAME hipSquareGenericTargetCompressed
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
  set_target_properties(hipSquareGenericTargetCompressed PROPERTIES COMPILE_FLAGS " -DGENERIC_COMPRESSED ${DISABLE_GENERIC_TARGET_ONLY} -mcode-object-version=6 --offload-compress")
  target_compile_options(hipSquareGenericTargetCompressed PUBLIC ${OFFLOAD_ARCH_GENERIC_STR})

  add_dependencies(hipSquareGenericTargetDefault hipSquareGenericTargetCompressed)
  if(BUILD_SHARED_LIBS)
    add_dependencies(hipSquareGenericTargetDefault hipSquareGenericTargetOnly)
    add_dependencies(hipSquareGenericTargetDefault hipSquareGenericTargetOnlyCompressed)
  endif()

  # SWDEV-548807 skip building hipSpirvTest
  if(false)
    add_custom_target(hipSpirvTest ALL
                    COMMAND ${CMAKE_HIP_COMPILER} -x hip ${CMAKE_CURRENT_SOURCE_DIR}/hipSpirvTest.cc
                    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/../../hipTestMain/hip_test_context.cc
                    -x hip ${CMAKE_CURRENT_SOURCE_DIR}/../../hipTestMain/main.cc
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    -I${Catch2_SOURCE_DIR}/src
                    -I${Catch2_BINARY_DIR}/generated-includes
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../external/picojson
                    --offload-arch=amdgcnspirv
                    -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/compiler/hipSpirvTest)
    add_dependencies(CompilerTest hipSpirvTest)
  endif()
endif()
