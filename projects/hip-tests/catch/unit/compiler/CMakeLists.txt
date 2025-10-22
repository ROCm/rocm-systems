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

  set(OFFLOAD_ARCH_GENERIC_LIST
    --offload-arch=gfx9-generic
    --offload-arch=gfx9-4-generic:sramecc+:xnack-
    --offload-arch=gfx9-4-generic:sramecc-:xnack-
    --offload-arch=gfx9-4-generic:xnack+
    --offload-arch=gfx10-1-generic
    --offload-arch=gfx10-3-generic
    --offload-arch=gfx11-generic
    --offload-arch=gfx12-generic
  )

  # Build hipSquareGenericTargetOnly to cover generic targets only
  # Because default catch2 build will reference CMAKE_CXX_FLAGS that contains specific targets which will hijack generic
  # target in hip-rt, we have to use custom build to contain generic targets only.
  if(BUILD_SHARED_LIBS) # Build hipGenericTargetOnly for shared libs only
    set(GENERIC_SOURCES
      hipSquareGenericTarget.cc
      ../../hipTestMain/hip_test_context.cc
      ../../hipTestMain/hip_test_features.cc
      ../../hipTestMain/main.cc
    )

    set(GENERIC_INCLUDE_DIRECTORIES
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${CMAKE_CURRENT_SOURCE_DIR}/../../external/Catch2
      ${CMAKE_CURRENT_SOURCE_DIR}/../../external/picojson
    )

    hip_add_executable(
      NAME hipSquareGenericTargetOnly
      SOURCE ${GENERIC_SOURCES}
      COMPILE_FLAGS -DNO_GENERIC_TARGET_ONLY_TEST -mcode-object-version=6 -w ${OFFLOAD_ARCH_GENERIC_LIST}
      INCLUDE_DIRECTORIES ${GENERIC_INCLUDE_DIRECTORIES}
      REPLACE_COMPILE_FLAGS
      ADD_TO_TARGET build_tests
    )

    hip_add_executable(
      NAME hipSquareGenericTargetOnlyCompressed
      SOURCE ${GENERIC_SOURCES}
      COMPILE_FLAGS -DNO_GENERIC_TARGET_ONLY_TEST -DGENERIC_COMPRESSED -mcode-object-version=6 --offload-compress -w ${OFFLOAD_ARCH_GENERIC_LIST}
      INCLUDE_DIRECTORIES ${GENERIC_INCLUDE_DIRECTORIES}
      REPLACE_COMPILE_FLAGS
      ADD_TO_TARGET build_tests
    )

    if(UNIX)
      target_link_libraries(hipSquareGenericTargetOnly PRIVATE stdc++fs)
      target_link_libraries(hipSquareGenericTargetOnlyCompressed PRIVATE stdc++fs)
    endif()
  endif()

  # Build hipSquareGenericTarget to cover generic targets and the specific target
  set(TEST_SRC
    hipSquareGenericTarget.cc
  )
  hip_add_exe_to_target(NAME hipSquareGenericTarget
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
  target_compile_options(hipSquareGenericTarget PRIVATE -mcode-object-version=6 -w ${OFFLOAD_ARCH_GENERIC_LIST})
  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(hipSquareGenericTarget PRIVATE NO_GENERIC_TARGET_ONLY_TEST)
  endif()

  hip_add_exe_to_target(NAME hipSquareGenericTargetCompressed
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
  target_compile_definitions(hipSquareGenericTargetCompressed PRIVATE GENERIC_COMPRESSED)
  target_compile_options(hipSquareGenericTargetCompressed PRIVATE -mcode-object-version=6 --offload-compress -w ${OFFLOAD_ARCH_GENERIC_LIST})
  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(hipSquareGenericTargetCompressed PRIVATE NO_GENERIC_TARGET_ONLY_TEST)
  endif()

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
