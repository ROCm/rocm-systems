macro(dashboard_submit)
  ctest_submit()
endmacro()

set(CTEST_PROJECT_NAME "aqlprofile")
set(CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE "cdash.rocprofiler.amd.com")
set(CTEST_DROP_LOCATION "/submit.php?project=${CTEST_PROJECT_NAME}")
set(CTEST_DROP_SITE_CDASH TRUE)

set(CTEST_UPDATE_TYPE git)
set(CTEST_UPDATE_VERSION_ONLY TRUE)
set(CTEST_GIT_COMMAND git)
set(CTEST_GIT_INIT_SUBMODULES FALSE)

set(CTEST_OUTPUT_ON_FAILURE TRUE)
set(CTEST_USE_LAUNCHERS TRUE)
set(CTEST_VERBOSE ON)

set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS "100")
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS "100")
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "51200")

if(NOT DEFINED CTEST_SOURCE_DIRECTORY)
  set(CTEST_SOURCE_DIRECTORY ".")
endif()

if(NOT DEFINED CTEST_BINARY_DIRECTORY)
  set(CTEST_BINARY_DIRECTORY "./build")
endif()

if(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm")
endif()

if(NOT DEFINED AQLPROFILE_EXTRA_CONFIGURE_ARGS)
  set(AQLPROFILE_EXTRA_CONFIGURE_ARGS "")
endif()

if(NOT DEFINED AQLPROFILE_BUILD_NUM_JOBS)
  set(AQLPROFILE_BUILD_NUM_JOBS "16")
endif()

set(CTEST_CONFIGURE_COMMAND "cmake -B ${CTEST_BINARY_DIRECTORY} -DCMAKE_BUILD_TYPE='RelWithDebInfo' -DCMAKE_PREFIX_PATH=/opt/rocm -DAQLPROFILE_BUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm -DCPACK_GENERATOR='DEB;RPM;STGZ' -DGPU_TARGETS='gfx906,gfx90a,gfx942,gfx1101,gfx1201' ${AQLPROFILE_EXTRA_CONFIGURE_ARGS} ${CTEST_SOURCE_DIRECTORY}")
set(CTEST_BUILD_COMMAND "cmake --build \"${CTEST_BINARY_DIRECTORY}\" -- -j ${AQLPROFILE_BUILD_NUM_JOBS} all mytest")

if(NOT DEFINED CTEST_SITE)
  set(CTEST_SITE "${HOSTNAME}")
endif()

if(NOT DEFINED CTEST_BUILD_NAME)
  set(CTEST_BUILD_NAME "aqlprofile-amd-staging-ubuntu-${RUNNER_HOSTNAME}-core")
endif()

macro(handle_error _message _ret)
  if(NOT ${${_ret}} EQUAL 0)
    dashboard_submit(PARTS Done RETURN_VALUE _submit_ret)
    message(FATAL_ERROR "${_message} failed: ${${_ret}}")
  endif()
endmacro()

ctest_start(Continuous)

set(CTEST_SOURCE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}/../../")

ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}" BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE _update_ret)
handle_error("Update" _update_ret)

ctest_configure(SOURCE "${CTEST_SOURCE_DIRECTORY}" BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE _configure_ret)
dashboard_submit(PARTS Start Update Configure RETURN_VALUE _submit_ret)

handle_error("Configure" _configure_ret)

ctest_build(SOURCE "${CTEST_SOURCE_DIRECTORY}" BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE _build_ret)
dashboard_submit(PARTS Build RETURN_VALUE _submit_ret)

handle_error("Build" _build_ret)

ctest_test(SOURCE "${CTEST_SOURCE_DIRECTORY}" BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE _test_ret)
dashboard_submit(PARTS Test RETURN_VALUE _submit_ret)

handle_error("Testing" _test_ret)

dashboard_submit(PARTS Done RETURN_VALUE _submit_ret)
