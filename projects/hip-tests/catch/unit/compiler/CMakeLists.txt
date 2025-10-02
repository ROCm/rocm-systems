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

  set(OFFLOAD_ARCH_GENERIC_STR "--offload-arch=gfx9-generic --offload-arch=gfx9-4-generic:sramecc+:xnack- --offload-arch=gfx9-4-generic:sramecc-:xnack- --offload-arch=gfx9-4-generic:xnack+ --offload-arch=gfx10-1-generic --offload-arch=gfx10-3-generic --offload-arch=gfx11-generic --offload-arch=gfx12-generic")

  set(DISABLE_GENERIC_TARGET_ONLY)

  # Build hipSquareGenericTargetOnly to cover generic targets only
  # Because default catch2 build will reference CMAKE_CXX_FLAGS that contains specific targets which will hijack generic
  # target in hip-rt, we have to use custom build to contain generic targets only.
  if(BUILD_SHARED_LIBS) # Build hipGenericTargetOnly for shared libs only
    set(GENERIC_TARGET_ONLY_EXE hipSquareGenericTargetOnly)
    set(GENERIC_TARGET_ONLY_COMPRESSED_EXE hipSquareGenericTargetOnlyCompressed)

    set(LIBFS)
    if(WIN32)
      set(GENERIC_TARGET_ONLY_EXE ${GENERIC_TARGET_ONLY_EXE}.exe)
      set(GENERIC_TARGET_ONLY_COMPRESSED_EXE ${GENERIC_TARGET_ONLY_COMPRESSED_EXE}.exe)
    else()
        set(LIBFS -lstdc++fs)
    endif()

    add_custom_target(hipSquareGenericTargetOnly ALL
                    COMMAND ${CMAKE_CXX_COMPILER} -DNO_GENERIC_TARGET_ONLY_TEST --std=c++17 -mcode-object-version=6 -w "${OFFLOAD_ARCH_GENERIC_STR}"
                    ${CMAKE_CURRENT_SOURCE_DIR}/hipSquareGenericTarget.cc
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../hipTestMain/hip_test_context.cc
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../hipTestMain/hip_test_features.cc
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../hipTestMain/main.cc
                    -o ${CMAKE_CURRENT_BINARY_DIR}/${GENERIC_TARGET_ONLY_EXE}
                    -I${HIP_PATH}/include/ --hip-path=${HIP_PATH}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../external/Catch2
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../external/picojson ${LIBFS})
    add_custom_target(hipSquareGenericTargetOnlyCompressed ALL
                    COMMAND ${CMAKE_CXX_COMPILER} -DNO_GENERIC_TARGET_ONLY_TEST -DGENERIC_COMPRESSED --std=c++17 -mcode-object-version=6 --offload-compress -w "${OFFLOAD_ARCH_GENERIC_STR}"
                    ${CMAKE_CURRENT_SOURCE_DIR}/hipSquareGenericTarget.cc
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../hipTestMain/hip_test_context.cc
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../hipTestMain/hip_test_features.cc
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../hipTestMain/main.cc
                    -o ${CMAKE_CURRENT_BINARY_DIR}/${GENERIC_TARGET_ONLY_COMPRESSED_EXE}
                    -I${HIP_PATH}/include/ --hip-path=${HIP_PATH}
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../external/Catch2
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../external/picojson ${LIBFS})
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/${GENERIC_TARGET_ONLY_EXE})
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/${GENERIC_TARGET_ONLY_COMPRESSED_EXE})
  else()
    set(DISABLE_GENERIC_TARGET_ONLY "-DNO_GENERIC_TARGET_ONLY_TEST")
  endif()

  # Build hipSquareGenericTarget to cover generic targets and the specific target
  set(TEST_SRC
    hipSquareGenericTarget.cc
  )
  hip_add_exe_to_target(NAME hipSquareGenericTarget
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
  set_target_properties(hipSquareGenericTarget PROPERTIES COMPILE_FLAGS "-mcode-object-version=6 ${DISABLE_GENERIC_TARGET_ONLY} -w ${OFFLOAD_ARCH_GENERIC_STR}")

  hip_add_exe_to_target(NAME hipSquareGenericTargetCompressed
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
  set_target_properties(hipSquareGenericTargetCompressed PROPERTIES COMPILE_FLAGS " -DGENERIC_COMPRESSED ${DISABLE_GENERIC_TARGET_ONLY} -mcode-object-version=6 --offload-compress -w ${OFFLOAD_ARCH_GENERIC_STR}")

  add_dependencies(hipSquareGenericTarget hipSquareGenericTargetCompressed)
  if(BUILD_SHARED_LIBS)
    add_dependencies(hipSquareGenericTarget hipSquareGenericTargetOnly)
    add_dependencies(hipSquareGenericTarget hipSquareGenericTargetOnlyCompressed)
  endif()

  # SWDEV-548807 skip building hipSpirvTest
  if(false)
    add_custom_target(hipSpirvTest ALL
                    COMMAND ${CMAKE_CXX_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/hipSpirvTest.cc
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../hipTestMain/hip_test_context.cc
                    ${CMAKE_CURRENT_SOURCE_DIR}/../../hipTestMain/main.cc
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../external/Catch2
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../external/picojson
                    -I${HIP_PATH}/include/ --hip-path=${HIP_PATH} --offload-arch=amdgcnspirv
                    -o ${CMAKE_CURRENT_BINARY_DIR}/../../unit/compiler/hipSpirvTest)
    add_dependencies(CompilerTest hipSpirvTest)
  endif()

endif()
