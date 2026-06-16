cmake_minimum_required ( VERSION 3.16.0 )
# set ( CMAKE_VERBOSE_MAKEFILE TRUE CACHE BOOL "Verbose Output" FORCE )

set ( EXE_NAME "ctrl" )

if ( NOT DEFINED TEST_DIR )
  set ( TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
  set ( TEST_BINARY_DIR ${CMAKE_BINARY_DIR} )
  project ( ${EXE_NAME} )
  list ( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake_modules" )
  ## Set build environment
  include ( env )
  enable_testing()
  include(CTest)
endif ()

## Util sources
file( GLOB UTIL_SRC "${TEST_DIR}/util/*.cpp" )

## Test control sources
set ( CTRL_SRC
  ${TEST_DIR}/app/test.cpp
  ${TEST_DIR}/ctrl/test_hsa.cpp
  ${TEST_DIR}/pgen/test_pmgr.cpp
)

## Test kernels sources
set ( TEST_NAME simple_convolution )
set ( KERN_SRC ${TEST_DIR}/${TEST_NAME}/${TEST_NAME}.cpp )

find_package(Clang REQUIRED CONFIG
             PATHS "${ROCM_PATH}"
             PATH_SUFFIXES "llvm/lib/cmake/clang")

## Building test executable
add_executable ( ${EXE_NAME} ${KERN_SRC} ${CTRL_SRC} ${UTIL_SRC} )
target_include_directories ( ${EXE_NAME}
  PRIVATE
    ${TEST_DIR}
    ${API_PATH}
    ${ROCM_ROOT_DIR}/include
    ${TEST_DIR}/parser/
    ${TEST_DIR}/../src/core/include
    ${TEST_DIR}/../src
)
target_link_libraries( ${EXE_NAME}
  PRIVATE
      pthread
      aqlprofile::headers
      hsa-runtime64::hsa-runtime64
      dl )
install(TARGETS ${EXE_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME} COMPONENT tests)

## Build the hsa (standalone) simple_convolution test
function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
  separate_arguments(CLANG_ARG_LIST UNIX_COMMAND
    "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0 -Wl,--build-id=sha1
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${OUTPUT_FILE} ${INPUT_FILE}")
  add_custom_command(OUTPUT ${TEST_BINARY_DIR}/${OUTPUT_FILE}
    COMMAND clang ${CLANG_ARG_LIST}
    WORKING_DIRECTORY ${TEST_BINARY_DIR}
    DEPENDS ${INPUT_FILE} clang
    COMMENT "Building ${OUTPUT_FILE}..."
    VERBATIM)
  install(FILES ${TEST_BINARY_DIR}/${OUTPUT_FILE} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME} COMPONENT tests)
  set(HSACO_TARGET_LIST ${HSACO_TARGET_LIST} ${TEST_BINARY_DIR}/${OUTPUT_FILE} PARENT_SCOPE)
endfunction(generate_hsaco)

separate_arguments(GPU_TARGETS)
list(LENGTH GPU_TARGETS list_count)
if (${list_count} LESS_EQUAL 1)
  string(REPLACE " " ";" GPU_LIST "${GPU_TARGETS}")
  string(REPLACE "," ";" GPU_LIST "${GPU_TARGETS}")
else()
  set(GPU_LIST ${GPU_TARGETS})
endif()

foreach(target_id ${GPU_LIST})
  ## generate kernel bitcodes
  generate_hsaco(${target_id} ${TEST_DIR}/${TEST_NAME}/${TEST_NAME}.cl ${target_id}_${TEST_NAME}.hsaco)
endforeach(target_id)
# add_custom_target(test DEPENDS ${HSACO_TARGET_LIST})
add_custom_target(mytest ALL DEPENDS ${TARGET_NAME} ${HSACO_TARGET_LIST})

## Deploying test run scripts
execute_process ( COMMAND sh -xc "cp --remove-destination ${TEST_DIR}/da_16b.py ${TEST_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "cp --remove-destination ${TEST_DIR}/text_sample_ex.py ${TEST_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "cp --remove-destination ${TEST_DIR}/binary_sample_ex.py ${TEST_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "cp --remove-destination ${TEST_DIR}/run.sh ${TEST_DIR}/run_priv.sh ${TEST_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "if [ ! -e run.sh ] ; then ln -s ${TEST_BINARY_DIR}/run.sh; fi" )
execute_process ( COMMAND sh -xc "if [ ! -e run_v.sh ] ; then ln -s ${TEST_BINARY_DIR}/run.sh run_v.sh; fi" )
execute_process ( COMMAND sh -xc "if [ ! -e run_priv.sh ] ; then ln -s ${TEST_BINARY_DIR}/run_priv.sh; fi" )
execute_process ( COMMAND sh -xc "if [ ! -e run_priv_v.sh ] ; then ln -s ${TEST_BINARY_DIR}/run_priv.sh run_priv_v.sh; fi" )
install(FILES ${TEST_DIR}/run_install.sh RENAME run_tests.sh
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME} COMPONENT tests)

#add_subdirectory(parser)

add_test(NAME legacy-tests COMMAND "${TEST_BINARY_DIR}/run.sh")
set_tests_properties(legacy-tests PROPERTIES TIMEOUT 600 LABELS "unittests")

# --- Test-filter standardization (TSF) ---
# Source of truth for categories is test_categories.yaml. When building inside the
# rocm-systems monorepo (shared/ctest present), apply category labels to the build
# tree AND emit a relocatable install-time CTestTestfile.cmake that test_runner.py
# runs via `ctest -L <category>` from the installed share/hsa-amd-aqlprofile/ dir.
# Guarded so standalone / sparse checkouts without shared/ctest still build.
set(AQLPROFILE_CTEST_INSTALL_FILE "")
if(DEFINED ROCM_SYSTEMS_ROOT
   AND EXISTS "${ROCM_SYSTEMS_ROOT}/shared/ctest/TestCategories.cmake"
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
  include("${ROCM_SYSTEMS_ROOT}/shared/ctest/TestCategories.cmake")

  # Relocatable install file: relative ./ path so ctest runs run_tests.sh from
  # the installed share/hsa-amd-aqlprofile/ directory. For now only the legacy
  # ctrl wrapper is registered; more entries (gtest/integration) can be appended
  # here later without any TheRock-side change.
  set(AQLPROFILE_CTEST_INSTALL_FILE
      "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
  file(WRITE "${AQLPROFILE_CTEST_INSTALL_FILE}"
[=[
# Generated by aqlprofile for install-time CTest (relative paths).
add_test(legacy-tests "./run_tests.sh")
set_tests_properties(legacy-tests PROPERTIES TIMEOUT 600 LABELS "unittests")
]=]
  )

  # Apply quick/standard/comprehensive/full labels from the YAML to both the
  # build-tree tests in this directory and the install-time file (2nd arg).
  apply_ctest_category_labels(
    "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
    "${AQLPROFILE_CTEST_INSTALL_FILE}"
  )
endif()

if(AQLPROFILE_CTEST_INSTALL_FILE)
  install(
    FILES "${AQLPROFILE_CTEST_INSTALL_FILE}"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}"
    COMPONENT tests
    RENAME "CTestTestfile.cmake"
  )
endif()

option(AQLPROFILE_BUILD_INTEGERATION_TESTS "Build integration tests" ON)

if(AQLPROFILE_BUILD_INTEGERATION_TESTS AND NOT WIN32)
    add_subdirectory(integration)
endif()
