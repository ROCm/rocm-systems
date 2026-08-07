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
if ( WIN32 )
  # test/util/hsa_rsrc_factory.cpp (and its .h / reg_offsets.h) are git symlinks into
  # ../../src/util. With core.symlinks=false, Windows checks them out as text stubs, so
  # the glob would compile a bogus source. Drop the stub and compile the real source.
  list( REMOVE_ITEM UTIL_SRC "${TEST_DIR}/util/hsa_rsrc_factory.cpp" )
  list( APPEND UTIL_SRC "${TEST_DIR}/../src/util/hsa_rsrc_factory.cpp" )
endif()

## Test control sources
set ( CTRL_SRC
  ${TEST_DIR}/app/test.cpp
  ${TEST_DIR}/ctrl/test_hsa.cpp
  ${TEST_DIR}/pgen/test_pmgr.cpp
)

## Test kernels sources
set ( TEST_NAME simple_convolution )
set ( KERN_SRC ${TEST_DIR}/${TEST_NAME}/${TEST_NAME}.cpp )

## Locate a clang capable of building the amdgcn kernel bitcode (.hsaco).
## On Linux this comes from the ROCm Clang CMake package (imported `clang` target).
## On Windows that package is usually absent, so fall back to locating clang.exe from
## the HIP SDK (AMDHIP64_LIB_DIR/../bin or HIP_PATH) and drive the custom command with it.
if ( WIN32 )
  find_package(Clang QUIET CONFIG
               PATHS "${ROCM_PATH}"
               PATH_SUFFIXES "llvm/lib/cmake/clang")
  if ( TARGET clang )
    set ( CLANG_EXECUTABLE clang )
  else ()
    get_filename_component ( _HIP_BIN_HINT "${AMDHIP64_LIB_DIR}/../bin" ABSOLUTE )
    find_program ( CLANG_EXECUTABLE
      NAMES clang clang.exe
      HINTS "${_HIP_BIN_HINT}" "$ENV{HIP_PATH}/bin" "${ROCM_PATH}/llvm/bin" "${ROCM_PATH}/bin" )
    if ( NOT CLANG_EXECUTABLE )
      message ( WARNING
        "clang not found for building kernel .hsaco on Windows. The ctrl.exe test binary "
        "will still build, but you must supply a prebuilt <target>_${TEST_NAME}.hsaco next "
        "to ctrl.exe at runtime. Set HIP_PATH or AMDHIP64_LIB_DIR to enable kernel builds." )
    endif ()
  endif ()
else ()
  find_package(Clang REQUIRED CONFIG
               PATHS "${ROCM_PATH}"
               PATH_SUFFIXES "llvm/lib/cmake/clang")
  set ( CLANG_EXECUTABLE clang )
endif ()

## On Windows the test calls core HSA APIs (hsa_memory_copy, hsa_queue_destroy,
## hsa_executable_* etc.) directly.
##
## Two HSA link modes are supported on Windows:
##  * default: link amdhip64.lib for the bulk of the HSA API and additionally link
##    the rocr import library hsa-runtime64.lib for the ~4 core symbols amdhip64 does
##    not re-export. NOTE: amdhip64 embeds its own private rocr, so this mixes two HSA
##    runtimes and cross-runtime handles (e.g. hsa_executable_symbol_get_info on an
##    executable created via amdhip64) fail at runtime.
##  * AQLPROFILE_WIN_LINK_ROCR=ON: link ONLY the standalone rocr hsa-runtime64.lib,
##    which exports the full HSA API. The whole stack (ctrl.exe + aqlprofile.dll, when
##    aqlprofile is also built with AQLPROFILE_WIN_LINK_ROCR=ON) then shares a single
##    HSA runtime and rocr auto-loads hsa-amd-aqlprofile64.dll from the exe directory.
## Point HSA_RUNTIME64_LIB_DIR at the directory containing hsa-runtime64.lib.
if ( WIN32 )
  option ( AQLPROFILE_WIN_LINK_ROCR
           "Link only the standalone rocr hsa-runtime64 import lib (single runtime) instead of amdhip64" OFF )
  find_library ( HSA_RUNTIME64_LIB
    NAMES hsa-runtime64
    HINTS "${HSA_RUNTIME64_LIB_DIR}" "${CMAKE_INSTALL_PREFIX}/lib" )
  if ( NOT HSA_RUNTIME64_LIB OR HSA_RUNTIME64_LIB STREQUAL "HSA_RUNTIME64_LIB-NOTFOUND" )
    if ( AQLPROFILE_WIN_LINK_ROCR )
      message ( FATAL_ERROR
        "AQLPROFILE_WIN_LINK_ROCR=ON but hsa-runtime64.lib not found. "
        "Set HSA_RUNTIME64_LIB_DIR to the directory containing hsa-runtime64.lib." )
    endif ()
    message ( WARNING
      "hsa-runtime64 import library not found; ctrl.exe may fail to link core HSA symbols. "
      "Set HSA_RUNTIME64_LIB_DIR to the directory containing hsa-runtime64.lib." )
    set ( HSA_RUNTIME64_LIB "" )
  else ()
    message ( STATUS "---HSA_RUNTIME64_LIB: ${HSA_RUNTIME64_LIB}" )
  endif ()
endif ()

## Building test executable
add_executable ( ${EXE_NAME} ${KERN_SRC} ${CTRL_SRC} ${UTIL_SRC} )
target_include_directories ( ${EXE_NAME}
  PRIVATE
    # On Windows, put src ahead of test so the real util/*.h are found instead of the
    # symlink text stubs under test/util (see UTIL_SRC note above).
    $<$<BOOL:${WIN32}>:${TEST_DIR}/../src>
    ${TEST_DIR}
    ${API_PATH}
    ${ROCM_ROOT_DIR}/include
    ${TEST_DIR}/parser/
    ${TEST_DIR}/../src/core/include
    ${TEST_DIR}/../src
)
## Windows HSA link source: with AQLPROFILE_WIN_LINK_ROCR link only rocr (single
## runtime); otherwise link amdhip64 + rocr (mixed, for the ~4 core symbols).
if ( WIN32 )
  if ( AQLPROFILE_WIN_LINK_ROCR )
    set ( CTRL_WIN_HSA_LIBS "${HSA_RUNTIME64_LIB}" )
  else ()
    set ( CTRL_WIN_HSA_LIBS "${AMDHIP64_LIB}" "${HSA_RUNTIME64_LIB}" )
  endif ()
endif ()
target_link_libraries( ${EXE_NAME}
  PRIVATE
      aqlprofile::headers
      $<$<NOT:$<BOOL:${WIN32}>>:pthread>
      $<$<NOT:$<BOOL:${WIN32}>>:hsa-runtime64::hsa-runtime64>
      $<$<NOT:$<BOOL:${WIN32}>>:dl>
      $<$<BOOL:${WIN32}>:${CTRL_WIN_HSA_LIBS}> )
install(TARGETS ${EXE_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME} COMPONENT tests)

## Build the hsa (standalone) simple_convolution test
function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
  separate_arguments(CLANG_ARG_LIST UNIX_COMMAND
    "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0 -Wl,--build-id=sha1
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${OUTPUT_FILE} ${INPUT_FILE}")
  ## Depend on the imported `clang` target when available (Linux/ROCm Clang package),
  ## otherwise just depend on the input file (Windows clang.exe located via find_program).
  if ( TARGET clang )
    set ( _CLANG_DEP clang )
  else ()
    set ( _CLANG_DEP "" )
  endif ()
  add_custom_command(OUTPUT ${TEST_BINARY_DIR}/${OUTPUT_FILE}
    COMMAND ${CLANG_EXECUTABLE} ${CLANG_ARG_LIST}
    WORKING_DIRECTORY ${TEST_BINARY_DIR}
    DEPENDS ${INPUT_FILE} ${_CLANG_DEP}
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

## Only generate kernel bitcodes when a clang was located. On Windows without clang the
## ctrl.exe target still builds; a prebuilt .hsaco can be dropped in beside it.
if ( CLANG_EXECUTABLE )
  foreach(target_id ${GPU_LIST})
    ## generate kernel bitcodes
    generate_hsaco(${target_id} ${TEST_DIR}/${TEST_NAME}/${TEST_NAME}.cl ${target_id}_${TEST_NAME}.hsaco)
  endforeach(target_id)
endif ()
# add_custom_target(test DEPENDS ${HSACO_TARGET_LIST})
add_custom_target(mytest ALL DEPENDS ${TARGET_NAME} ${HSACO_TARGET_LIST})

## Deploying test run scripts (POSIX shell helpers; not applicable on Windows)
if ( NOT WIN32 )
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
endif ()

option(AQLPROFILE_BUILD_INTEGERATION_TESTS "Build integration tests" ON)

if(AQLPROFILE_BUILD_INTEGERATION_TESTS AND NOT WIN32)
    add_subdirectory(integration)
endif()
