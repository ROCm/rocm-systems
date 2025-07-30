################################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

set(ROCPROFSYS_ABORT_FAIL_REGEX
    "### ERROR ###|unknown-hash=|address of faulting memory reference|exiting with non-zero exit code|terminate called after throwing an instance|calling abort.. in |Exit code: [1-9]"
    CACHE INTERNAL
    "Regex to catch abnormal exits when a PASS_REGULAR_EXPRESSION is set"
    FORCE
)

# adds a ctest for executable
function(ROCPROFILER_SYSTEMS_ADD_BIN_TEST)
    cmake_parse_arguments(
        TEST
        "" # options
        "NAME;TARGET;TIMEOUT;WORKING_DIRECTORY" # single value args
        "ARGS;ENVIRONMENT;LABELS;PROPERTIES;PASS_REGEX;FAIL_REGEX;SKIP_REGEX;DEPENDS;COMMAND" # multiple
        # value args
        ${ARGN}
    )

    if(NOT TEST_WORKING_DIRECTORY)
        set(TEST_WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    endif()

    if(NOT TEST_ENVIRONMENT)
        set(TEST_ENVIRONMENT
            "ROCPROFSYS_TRACE=ON"
            "ROCPROFSYS_PROFILE=ON"
            "ROCPROFSYS_USE_SAMPLING=ON"
            "ROCPROFSYS_TIME_OUTPUT=OFF"
            "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}"
        )
    endif()

    # common
    list(
        APPEND
        TEST_ENVIRONMENT
        "ROCPROFSYS_CI=ON"
        "ROCPROFSYS_CI_TIMEOUT=${TEST_TIMEOUT}"
        "ROCPROFSYS_CONFIG_FILE="
        "ROCPROFSYS_OUTPUT_PATH=${PROJECT_BINARY_DIR}/rocprof-sys-tests-output"
        "TWD=${TEST_WORKING_DIRECTORY}"
    )
    # copy for inverse
    set(TEST_ENVIRONMENT_INV "${TEST_ENVIRONMENT}")

    # different for regular test and inverse test
    list(APPEND TEST_ENVIRONMENT "ROCPROFSYS_OUTPUT_PREFIX=${TEST_NAME}/")
    list(APPEND TEST_ENVIRONMENT_INV "ROCPROFSYS_OUTPUT_PREFIX=${TEST_NAME}-inverse/")

    if(
        NOT "${TEST_PASS_REGEX}" STREQUAL ""
        AND NOT "${TEST_FAIL_REGEX}" STREQUAL ""
        AND NOT "${TEST_FAIL_REGEX}" MATCHES "\\|ROCPROFSYS_ABORT_FAIL_REGEX"
    )
        rocprofiler_systems_message(
            FATAL_ERROR
            "${TEST_NAME} has set pass and fail regexes but fail regex does not include '|ROCPROFSYS_ABORT_FAIL_REGEX'"
        )
    endif()

    if("${TEST_FAIL_REGEX}" STREQUAL "")
        set(TEST_FAIL_REGEX "(${ROCPROFSYS_ABORT_FAIL_REGEX})")
    else()
        string(
            REPLACE
            "|ROCPROFSYS_ABORT_FAIL_REGEX"
            "|${ROCPROFSYS_ABORT_FAIL_REGEX}"
            TEST_FAIL_REGEX
            "${TEST_FAIL_REGEX}"
        )
    endif()

    if(TEST_COMMAND)
        add_test(
            NAME ${TEST_NAME}
            COMMAND ${TEST_COMMAND} ${TEST_ARGS}
            WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY}
        )

        set_tests_properties(
            ${TEST_NAME}
            PROPERTIES
                ENVIRONMENT "${TEST_ENVIRONMENT}"
                TIMEOUT ${TEST_TIMEOUT}
                DEPENDS "${TEST_DEPENDS}"
                LABELS "rocprofiler-systems-bin;${TEST_LABELS}"
                PASS_REGULAR_EXPRESSION "${TEST_PASS_REGEX}"
                FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGEX}"
                SKIP_REGULAR_EXPRESSION "${TEST_SKIP_REGEX}"
                ${TEST_PROPERTIES}
        )
    elseif(TARGET ${TEST_TARGET})
        add_test(
            NAME ${TEST_NAME}
            COMMAND $<TARGET_FILE:${TEST_TARGET}> ${TEST_ARGS}
            WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY}
        )

        set_tests_properties(
            ${TEST_NAME}
            PROPERTIES
                ENVIRONMENT "${TEST_ENVIRONMENT}"
                TIMEOUT ${TEST_TIMEOUT}
                DEPENDS "${TEST_DEPENDS}"
                LABELS "rocprofiler-systems-bin;${TEST_LABELS}"
                PASS_REGULAR_EXPRESSION "${TEST_PASS_REGEX}"
                FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGEX}"
                SKIP_REGULAR_EXPRESSION "${TEST_SKIP_REGEX}"
                ${TEST_PROPERTIES}
        )
    elseif(ROCPROFSYS_BUILD_TESTING)
        message(FATAL_ERROR "Error! ${TEST_TARGET} does not exist")
    endif()
endfunction()

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-instrument-help
    TARGET rocprofiler-systems-instrument
    ARGS --help
    LABELS rocprofiler-systems-instrument
    TIMEOUT 45
    PASS_REGEX
        ".*\\\[rocprof-sys-instrument\\\] Usage:.*\\\[DEBUG OPTIONS\\\].*\\\[MODE OPTIONS\\\].*\\\[LIBRARY OPTIONS\\\].*\\\[SYMBOL SELECTION OPTIONS\\\].*\\\[RUNTIME OPTIONS\\\].*\\\[GRANULARITY OPTIONS\\\].*\\\[DYNINST OPTIONS\\\].*"
)

# on RedHat, /usr/bin/ls is a script for `coreutils --coreutils-prog=ls`
if(EXISTS /usr/bin/coreutils)
    set(LS_NAME "coreutils")
    set(LS_ARGS "--coreutils-prog=ls")
else()
    set(LS_NAME ls)
    set(LS_ARGS)
endif()

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-instrument-simulate-ls
    TARGET rocprofiler-systems-instrument
    ARGS --simulate
         --print-format
         json
         txt
         xml
         -v
         2
         --all-functions
         --
         ${LS_NAME}
         ${LS_ARGS}
    LABELS "simulate"
    TIMEOUT 240
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-instrument-simulate-ls-check
    DEPENDS rocprofiler-systems-instrument-simulate-ls
    COMMAND
        ls
        rocprof-sys-tests-output/rocprofiler-systems-instrument-simulate-ls/instrumentation
    TIMEOUT 60
    PASS_REGEX
        ".*available.json.*available.txt.*available.xml.*excluded.json.*excluded.txt.*excluded.xml.*instrumented.json.*instrumented.txt.*instrumented.xml.*overlapping.json.*overlapping.txt.*overlapping.xml.*"
    FAIL_REGEX "No such file or directory|not found|ROCPROFSYS_ABORT_FAIL_REGEX"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-instrument-simulate-lib
    TARGET rocprofiler-systems-instrument
    ARGS --print-available functions -v 2 --
         $<TARGET_FILE:rocprofiler-systems-user-library>
    LABELS "simulate"
    TIMEOUT 120
    PASS_REGEX
        "\\\[rocprof-sys\\\]\\\[exe\\\] Runtime instrumentation is not possible!(.*)\n(.*)\\\[rocprof-sys\\\]\\\[exe\\\] Switching to binary rewrite mode and assuming '--simulate --all-functions'"
)

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/tmp)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-instrument-simulate-lib-basename
    TARGET rocprofiler-systems-instrument
    ARGS --print-available
         functions
         -v
         2
         -o
         ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/rocprof-sys-instrument-simulate-lib-basename/${CMAKE_SHARED_LIBRARY_PREFIX}$<TARGET_FILE_BASE_NAME:rocprofiler-systems-user-library>${CMAKE_SHARED_LIBRARY_SUFFIX}
         --
         ${CMAKE_SHARED_LIBRARY_PREFIX}$<TARGET_FILE_BASE_NAME:rocprofiler-systems-user-library>${CMAKE_SHARED_LIBRARY_SUFFIX}
    LABELS "simulate"
    TIMEOUT 120
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/tmp
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-instrument-write-log
    TARGET rocprofiler-systems-instrument
    ARGS --print-instrumented
         functions
         -v
         1
         --log-file
         user.log
         --
         ${LS_NAME}
         ${LS_ARGS}
    LABELS "log"
    TIMEOUT 120
    PASS_REGEX "Opening .*/instrumentation/user.log"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-instrument-write-log-check
    DEPENDS rocprofiler-systems-instrument-write-log
    COMMAND
        ls
        rocprof-sys-tests-output/rocprofiler-systems-instrument-write-log/instrumentation/user.log
    LABELS "log"
    TIMEOUT 60
    PASS_REGEX "user.log"
    FAIL_REGEX "No such file or directory|not found|ROCPROFSYS_ABORT_FAIL_REGEX"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-help
    TARGET rocprofiler-systems-avail
    ARGS --help
    LABELS "rocprofiler-systems-avail"
    TIMEOUT 45
    PASS_REGEX
        ".*\\\[rocprof-sys-avail\\\] Usage:.*\\\[DEBUG OPTIONS\\\].*\\\[INFO OPTIONS\\\].*\\\[FILTER OPTIONS\\\].*\\\[COLUMN OPTIONS\\\].*\\\[DISPLAY OPTIONS\\\].*\\\[OUTPUT OPTIONS\\\].*"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-all
    TARGET rocprofiler-systems-avail
    ARGS --all
    LABELS "rocprofiler-systems-avail"
    TIMEOUT 45
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-all-expand-keys
    TARGET rocprofiler-systems-avail
    ARGS --all --expand-keys
    LABELS "rocprofiler-systems-avail"
    TIMEOUT 45
    FAIL_REGEX "%[a-zA-Z_]%|ROCPROFSYS_ABORT_FAIL_REGEX"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-all-only-available-alphabetical
    TARGET rocprofiler-systems-avail
    ARGS --all --available --alphabetical --debug --output
         ${CMAKE_CURRENT_BINARY_DIR}/rocprof-sys-avail-all-only-available-alphabetical.log
    LABELS "rocprofiler-systems-avail"
    TIMEOUT 45
    PROPERTIES
        ATTACHED_FILES
        ${CMAKE_CURRENT_BINARY_DIR}/rocprof-sys-avail-all-only-available-alphabetical.log
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-all-csv
    TARGET rocprofiler-systems-avail
    ARGS --all --csv --csv-separator "#"
    LABELS "rocprofiler-systems-avail"
    TIMEOUT 45
    PASS_REGEX
        "COMPONENT#AVAILABLE#VALUE_TYPE#STRING_IDS#FILENAME#DESCRIPTION#CATEGORY#.*ENVIRONMENT VARIABLE#VALUE#DATA TYPE#DESCRIPTION#CATEGORIES#.*HARDWARE COUNTER#DEVICE#AVAILABLE#DESCRIPTION#"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-filter-wall-clock-available
    TARGET rocprofiler-systems-avail
    ARGS -r wall_clock -C --available
    LABELS "rocprofiler-systems-avail"
    TIMEOUT 45
    PASS_REGEX
        "\\\|[-]+\\\|\n\\\|[ ]+COMPONENT[ ]+\\\|\n\\\|[-]+\\\|\n\\\| (wall_clock)[ ]+\\\|\n\\\|[-]+\\\|"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-category-filter-rocprofiler-systems
    TARGET rocprofiler-systems-avail
    ARGS --categories settings::rocprofsys --brief
    LABELS "rocprofiler-systems-avail"
    TIMEOUT 45
    PASS_REGEX "ROCPROFSYS_(SETTINGS_DESC|OUTPUT_FILE|OUTPUT_PREFIX)"
    FAIL_REGEX
        "ROCPROFSYS_(ADD_SECONDARY|SCIENTIFIC|PRECISION|MEMORY_PRECISION|TIMING_PRECISION)|ROCPROFSYS_ABORT_FAIL_REGEX"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-category-filter-timemory
    TARGET rocprofiler-systems-avail
    ARGS --categories settings::timemory --brief --advanced
    LABELS "rocprofiler-systems-avail"
    TIMEOUT 45
    PASS_REGEX
        "ROCPROFSYS_(ADD_SECONDARY|SCIENTIFIC|PRECISION|MEMORY_PRECISION|TIMING_PRECISION)"
    FAIL_REGEX "ROCPROFSYS_(SETTINGS_DESC|OUTPUT_FILE)|ROCPROFSYS_ABORT_FAIL_REGEX"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-regex-negation
    TARGET rocprofiler-systems-avail
    ARGS -R
         rocprofsys
         ~timemory
         -r
         _P
         ~PERFETTO
         ~PROCESS_SAMPLING
         ~KOKKOSP
         ~PAGE
         --csv
         --brief
         --advanced
    LABELS "rocprofiler-systems-avail"
    TIMEOUT 45
    PASS_REGEX
        "ENVIRONMENT VARIABLE,[ \n]+ROCPROFSYS_THREAD_POOL_SIZE,[ \n]+ROCPROFSYS_USE_PID,[ \n]+"
    FAIL_REGEX "ROCPROFSYS_TRACE|ROCPROFSYS_ABORT_FAIL_REGEX"
)

string(
    REPLACE
    "+"
    "\\\+"
    _AVAIL_CFG_PATH
    "${PROJECT_BINARY_DIR}/rocprof-sys-tests-output/rocprof-sys-avail/rocprof-sys-"
)
# use of TWD == Test Working Directory (added by function)
rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-write-config
    TARGET rocprofiler-systems-avail
    ARGS -G
         %env{TWD}%/rocprof-sys-tests-output/rocprof-sys-avail/rocprof-sys-test.cfg
         -F
         txt
         json
         xml
         --force
         --all
         -c
         rocprofsys
    TIMEOUT 45
    LABELS "rocprofiler-systems-avail"
    PASS_REGEX
        "Outputting JSON configuration file '${_AVAIL_CFG_PATH}test\\\.json'(.*)Outputting XML configuration file '${_AVAIL_CFG_PATH}test\\\.xml'(.*)Outputting text configuration file '${_AVAIL_CFG_PATH}test\\\.cfg'(.*)"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-write-config-tweak
    TARGET rocprofiler-systems-avail
    ARGS -G %env{TWD}%/rocprof-sys-tests-output/rocprof-sys-avail/rocprof-sys-tweak.cfg -F
         txt json xml --force
    TIMEOUT 45
    LABELS "rocprofiler-systems-avail"
    ENVIRONMENT "ROCPROFSYS_TRACE=OFF;ROCPROFSYS_PROFILE=ON"
    PASS_REGEX
        "Outputting JSON configuration file '${_AVAIL_CFG_PATH}tweak\\\.json'(.*)Outputting XML configuration file '${_AVAIL_CFG_PATH}tweak\\\.xml'(.*)Outputting text configuration file '${_AVAIL_CFG_PATH}tweak\\\.cfg'(.*)"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-list-keys
    TARGET rocprofiler-systems-avail
    ARGS --list-keys --expand-keys
    TIMEOUT 45
    LABELS "rocprofiler-systems-avail"
    PASS_REGEX "Output Keys:\n(.*)%argv%(.*)%argv_hash%"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-list-keys-markdown
    TARGET rocprofiler-systems-avail
    ARGS --list-keys --expand-keys --markdown
    TIMEOUT 45
    LABELS "rocprofiler-systems-avail;markdown"
    PASS_REGEX "(.*)`%argv%`(.*)`%argv_hash%`"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-list-categories
    TARGET rocprofiler-systems-avail
    ARGS --list-categories
    TIMEOUT 45
    LABELS "rocprofiler-systems-avail"
    PASS_REGEX " component::(.*) hw_counters::(.*) settings::"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-avail-core-categories
    TARGET rocprofiler-systems-avail
    ARGS -c core
    TIMEOUT 45
    LABELS "rocprofiler-systems-avail"
    PASS_REGEX
        "ROCPROFSYS_CONFIG_FILE(.*)ROCPROFSYS_ENABLED(.*)ROCPROFSYS_SUPPRESS_CONFIG(.*)ROCPROFSYS_SUPPRESS_PARSING(.*)ROCPROFSYS_VERBOSE"
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-run-help
    TARGET rocprofiler-systems-run
    ARGS --help
    TIMEOUT 45
    LABELS "rocprofiler-systems-run"
)

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/rocprof-sys-tests-config")
file(
    WRITE
    "${PROJECT_BINARY_DIR}/rocprof-sys-tests-config/empty.cfg"
    "
#
# empty config file
#
"
)

add_executable(sleeper ${CMAKE_CURRENT_SOURCE_DIR}/sleeper.cpp)
set_target_properties(
    sleeper
    PROPERTIES
        BUILD_TYPE RelWithDebInfo
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/testing
)

rocprofiler_systems_add_bin_test(
    NAME rocprofiler-systems-run-args
    TARGET rocprofiler-systems-run
    ARGS --monochrome
         --debug=false
         -v
         1
         -c
         %env{TWD}%/rocprof-sys-tests-config/empty.cfg
         -o
         rocprof-sys-tests-output
         rocprofiler-systems-run-args-output/
         -TPHD
         -S
         cputime
         realtime
         --trace-wait=1.0e-12
         --trace-duration=5.0
         --wait=1.0
         --duration=3.0
         --trace-file=perfetto-run-args-trace.proto
         --trace-buffer-size=100
         --trace-fill-policy=ring_buffer
         --profile-format
         console
         json
         text
         --process-freq
         1000
         --process-wait
         0.0
         --process-duration
         10
         --cpus
         0-4
         --gpus
         0
         -f
         1000
         --sampling-wait
         1.0
         --sampling-duration
         10
         -t
         0-3
         --sample-cputime
         1000
         1.0
         0-3
         --sample-realtime
         10
         0.5
         0-3
         -I
         all
         -E
         mutex-locks
         rw-locks
         spin-locks
         -C
         perf::INSTRUCTIONS
         --inlines
         --hsa-interrupt
         0
         --use-causal=false
         --use-kokkosp
         --num-threads-hint=4
         --sampling-allocator-size=32
         --ci
         --dl-verbose=3
         --perfetto-annotations=off
         --kokkosp-kernel-logger
         --kokkosp-name-length-max=1024
         --kokkosp-prefix="[kokkos]"
         --tmpdir
         ${CMAKE_BINARY_DIR}/rocprof-sys-tests-config/tmpdir
         --perfetto-backend
         inprocess
         --use-pid
         false
         --time-output
         off
         --thread-pool-size
         0
         --timemory-components
         wall_clock
         cpu_clock
         peak_rss
         page_rss
         --fork
         --
         $<TARGET_FILE:sleeper>
         5
    TIMEOUT 45
    LABELS "rocprofiler-systems-run"
)
