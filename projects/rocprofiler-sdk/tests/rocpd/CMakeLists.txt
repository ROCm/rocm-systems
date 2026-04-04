#
# rocpd command-line tests
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocpd
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(rocpd-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

#########################################################################################
#
# Test the --help flag works
#
#########################################################################################

foreach(_SUBPARSER "" "-convert" "-query" "-summary")
    string(REPLACE "-" "" _CMD "${_SUBPARSER}")

    rocprofiler_add_integration_execute_test(
        rocpd${_SUBPARSER}-help
        COMMAND ${Python3_EXECUTABLE} -m rocpd ${_CMD} --help
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 120
        LABELS "integration-tests;rocpd"
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${rocpd-env}")
endforeach()

foreach(_MODULE "csv" "pftrace" "otf2" "query" "summary")
    rocprofiler_add_integration_execute_test(
        rocpd-module-${_MODULE}-help
        COMMAND ${Python3_EXECUTABLE} -m rocpd.${_MODULE} --help
        DEPENDS rocprofiler-sdk::rocprofv3
        TIMEOUT 120
        LABELS "integration-tests;rocpd"
        PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
        ENVIRONMENT "${rocpd-env}")
endforeach()
