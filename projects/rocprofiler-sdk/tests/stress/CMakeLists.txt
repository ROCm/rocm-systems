#
# Time-boxed repetition of flaky-test suspects (see run_stress.py for the selection).
#
# Registered only when this tests project is built in-tree, which in CI is the run-ci.py step
# that submits results to CDash. The standalone trees (Test Install Build, Test Installed
# Packages, TheRock) already run close to their step timeouts and would lose all results if
# this pushed them over.
#
if(PROJECT_IS_TOP_LEVEL)
    return()
endif()

# The in-tree CI step is capped at 30 minutes and configure + build + the full suite takes
# 16-20 of them, so keep the default budget well inside the remaining headroom.
set(ROCPROFILER_STRESS_BUDGET_SECONDS
    "300"
    CACHE STRING "Wall-clock budget for the stress.* repetition test, in seconds")

add_test(
    NAME stress.repeat-suspects
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_stress.py --ctest
        ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR} --budget-seconds
        ${ROCPROFILER_STRESS_BUDGET_SECONDS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# RUN_SERIAL plus the lowest COST makes it the last test ctest starts and keeps it from
# sharing the GPU, so it neither overwrites the output of an execute test whose validation
# is still pending nor perturbs the timing of the regular run. The fault handler turns a
# SIGSEGV in any repeated test into a symbolized stack trace in the output.
set_tests_properties(
    stress.repeat-suspects
    PROPERTIES RUN_SERIAL
               TRUE
               COST
               -1000000
               TIMEOUT
               420
               LABELS
               "stress"
               ENVIRONMENT
               "ROCPROFILER_FAILURE_SIGNAL_HANDLER=1;ROCPROF_FAILURE_SIGNAL_HANDLER=1")
