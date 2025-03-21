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

  set(OFFLOAD_ARCH_GENERIC_STR "--offload-arch=gfx9-generic --offload-arch=gfx10-1-generic --offload-arch=gfx10-3-generic --offload-arch=gfx11-generic --offload-arch=gfx12-generic")

  # Build hipSquareGenericTargetOnly to cover generic targets only
  # Because default catch2 build will reference CMAKE_CXX_FLAGS that contains specific targets which will hijack generic
  # target in hip-rt, we have to use custom build to contain generic targets only.
  set(GENERIC_TARGET_ONLY_EXE hipSquareGenericTargetOnly)
  set(LIBFS)
  if(WIN32)
    set(GENERIC_TARGET_ONLY_EXE ${GENERIC_TARGET_ONLY_EXE}.exe)
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
                  -I${HIP_PATH}/include/ --rocm-path=${ROCM_PATH}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../external/Catch2
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../external/picojson ${LIBFS})
  set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS ${CMAKE_CURRENT_BINARY_DIR}/${GENERIC_TARGET_ONLY_EXE})

  # Build hipSquareGenericTarget to cover generic targets and the specific target
  set(TEST_SRC
    hipSquareGenericTarget.cc
  )
  hip_add_exe_to_target(NAME hipSquareGenericTarget
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
  set_source_files_properties(hipSquareGenericTarget.cc
                      PROPERTIES COMPILE_FLAGS "-mcode-object-version=6 -w ${OFFLOAD_ARCH_GENERIC_STR}")
  add_dependencies(hipSquareGenericTarget hipSquareGenericTargetOnly)
endif()
