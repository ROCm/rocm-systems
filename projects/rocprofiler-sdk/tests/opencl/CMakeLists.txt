#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-opencl
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk-roctx REQUIRED)

set(OPENCL_TEST_ENVIRONMENT
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library>:$ENV{LD_LIBRARY_PATH}"
    "ROCPROFILER_REGISTER_LOG_LEVEL=info"
    "ROCPROFILER_ATTACH_LOG_LEVEL=info")

# this test verifies that OpenCL queues (which should not be intercepted) execute without
# error (i.e. does not segfault)
rocprofiler_add_integration_execute_test(
    test-opencl-basic
    COMMAND $<TARGET_FILE:opencl-vector-add>
    DEPENDS opencl-vector-add
    TIMEOUT 45
    LABELS "integration-tests"
    ENVIRONMENT "${OPENCL_TEST_ENVIRONMENT}"
    PASS_REGULAR_EXPRESSION
        "Backtrace analysis excludes queue=\\{[0-9A-Fa-f]+\\} from queue interception."
    FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}|Backtrace analysis includes queue=\\{[0-9A-Fa-f]+\\} for queue interception."
    )

# this verifies that OpenCL queues are not intercepted. The PASS_REGULAR_EXPRESSION can
# cause the return code of the application to be ignored, which is why we have test-opencl
# above
rocprofiler_add_integration_execute_test(
    test-opencl-not-intercepted
    COMMAND $<TARGET_FILE:opencl-vector-add>
    DEPENDS opencl-vector-add
    TIMEOUT 45
    LABELS "integration-tests"
    ENVIRONMENT "${OPENCL_TEST_ENVIRONMENT}"
    PASS_REGULAR_EXPRESSION
        "Backtrace analysis excludes queue=\\{[0-9A-Fa-f]+\\} from queue interception."
    FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}|Backtrace analysis includes queue=\\{[0-9A-Fa-f]+\\} for queue interception."
    )
