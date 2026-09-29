#
# Time-boxed repetition of flaky-test suspects (see run_stress.py for the selection).
#
# Registered only when this tests project is built in-tree, which in CI is the run-ci.py step
# that submits results to CDash. The standalone trees (Test Install Build, Test Installed
# Packages, TheRock) already run close to their step timeouts and would lose all results if
# this pushed them over. Coverage builds are excluded for the same reason: their in-tree step
# already takes 22-27 of its 30 minutes, and instrumented code runs every repeat slower.
#
if(PROJECT_IS_TOP_LEVEL OR ROCPROFILER_BUILD_CODECOV)
    return()
endif()

# The in-tree CI step is capped at 30 minutes and configure + build + the full suite takes
# 14-17 of them, so the two budgets below share about six minutes of the remaining
# headroom (with six minutes in total the ubuntu step took 24 minutes).
set(ROCPROFILER_STRESS_BUDGET_SECONDS
    "240"
    CACHE STRING "Wall-clock budget for the stress.* repetition test, in seconds")
set(ROCPROFILER_STRESS_PROBE_BUDGET_SECONDS
    "120"
    CACHE STRING "Wall-clock budget for the multistream save/restore probe, in seconds")

add_test(
    NAME stress.repeat-suspects
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_stress.py --ctest
        ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR} --budget-seconds
        ${ROCPROFILER_STRESS_BUDGET_SECONDS} --report-only
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# RUN_SERIAL plus the lowest COST makes it the last test ctest starts and keeps it from
# sharing the GPU, so it neither overwrites the output of an execute test whose validation
# is still pending nor perturbs the timing of the regular run. The fault handler turns a
# SIGSEGV in any repeated test into a symbolized stack trace in the output. --report-only
# keeps it from failing: a failure fails the whole in-tree step, and GitHub then skips the
# install and installed-test steps after it, which on ubuntu are two more full-suite runs.
# Its verdict is the STRESS_RESULT_JSON line in the output recorded on CDash.
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

# Repeats multistream under rocprofv3 with SQ_WAVES_RESTORED and SQ_WAVES_SAVED next to
# SQ_WAVES, to test whether the rare SQ_WAVES_sum == 2 in
# counter-collection.test_counter_values is a preempted wave counted again on restore (see
# probe_multistream_restore.py).
if(TARGET multistream AND TARGET rocprofiler-sdk::rocprofv3)
    add_test(
        NAME stress.probe-multistream-restore
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/probe_multistream_restore.py
            --rocprofv3 $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --app
            $<TARGET_FILE:multistream> --workdir ${CMAKE_CURRENT_BINARY_DIR}/probe
            --budget-seconds ${ROCPROFILER_STRESS_PROBE_BUDGET_SECONDS} --report-only
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    set_tests_properties(
        stress.probe-multistream-restore
        PROPERTIES
            RUN_SERIAL
            TRUE
            COST
            -1000001
            TIMEOUT
            360
            LABELS
            "stress"
            ENVIRONMENT
            "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages;ROCPROFILER_FAILURE_SIGNAL_HANDLER=1;ROCPROF_FAILURE_SIGNAL_HANDLER=1"
        )
endif()
