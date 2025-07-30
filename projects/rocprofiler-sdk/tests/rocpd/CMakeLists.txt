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
    "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

#########################################################################################
#
# Test the --help flag works
#
#########################################################################################

foreach(_SUBPARSER "" "-convert")
    string(REPLACE "-" "" _CMD "${_SUBPARSER}")
    add_test(NAME rocpd${_SUBPARSER}-help COMMAND ${Python3_EXECUTABLE} -m rocpd ${_CMD}
                                                  --help)

    set_tests_properties(
        rocpd${_SUBPARSER}-help
        PROPERTIES TIMEOUT 120 LABELS "integration-tests;rocpd" ENVIRONMENT
                   "${rocpd-env}" FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endforeach()

foreach(_MODULE "csv" "pftrace" "otf2")
    add_test(NAME rocpd-module-${_MODULE}-help COMMAND ${Python3_EXECUTABLE} -m
                                                       rocpd.${_MODULE} --help)

    set_tests_properties(
        rocpd-module-${_MODULE}-help
        PROPERTIES TIMEOUT 120 LABELS "integration-tests;rocpd" ENVIRONMENT
                   "${rocpd-env}" FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endforeach()
