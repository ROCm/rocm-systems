# adds a ctest for executable
function(OMNITRACE_ADD_BIN_TEST)
    cmake_parse_arguments(
        TEST
        "ADD_INVERSE" # options
        "NAME;TARGET;TIMEOUT;WORKING_DIRECTORY" # single value args
        "ARGS;ENVIRONMENT;LABELS;PROPERTIES;PASS_REGEX;FAIL_REGEX;SKIP_REGEX;DEPENDS;COMMAND" # multiple
        # value args
        ${ARGN})

    if(NOT TEST_WORKING_DIRECTORY)
        set(TEST_WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    endif()
    if(NOT OMNITRACE_DYNINST_API_RT_DIR AND OMNITRACE_DYNINST_API_RT)
        get_filename_component(OMNITRACE_DYNINST_API_RT_DIR "${OMNITRACE_DYNINST_API_RT}"
                               DIRECTORY)
    endif()

    if(OMNITRACE_BUILD_DYNINST)
        set(OMNITRACE_DYNINST_API_RT_DIR
            "${PROJECT_BINARY_DIR}/external/dyninst/dyninstAPI_RT:${PROJECT_BINARY_DIR}/external/dyninst/dyninstAPI"
            )
    endif()

    if(NOT TEST_ENVIRONMENT)
        set(TEST_ENVIRONMENT
            "OMNITRACE_USE_PERFETTO=ON"
            "OMNITRACE_USE_TIMEMORY=ON"
            "OMNITRACE_USE_SAMPLING=ON"
            "OMNITRACE_TIME_OUTPUT=OFF"
            "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
            )
    endif()

    # common
    list(APPEND TEST_ENVIRONMENT "OMNITRACE_CI=ON" "OMNITRACE_CONFIG_FILE="
         "OMNITRACE_OUTPUT_PATH=${PROJECT_BINARY_DIR}/omnitrace-tests-output"
         "TWD=${TEST_WORKING_DIRECTORY}")
    # copy for inverse
    set(TEST_ENVIRONMENT_INV "${TEST_ENVIRONMENT}")

    # different for regular test and inverse test
    list(APPEND TEST_ENVIRONMENT "OMNITRACE_OUTPUT_PREFIX=${TEST_NAME}/")
    list(APPEND TEST_ENVIRONMENT_INV "OMNITRACE_OUTPUT_PREFIX=${TEST_NAME}-inverse/")

    if(TEST_COMMAND)
        add_test(
            NAME ${TEST_NAME}
            COMMAND ${TEST_COMMAND} ${TEST_ARGS}
            WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})

        set_tests_properties(
            ${TEST_NAME}
            PROPERTIES ENVIRONMENT
                       "${TEST_ENVIRONMENT}"
                       TIMEOUT
                       ${TEST_TIMEOUT}
                       DEPENDS
                       "${TEST_DEPENDS}"
                       LABELS
                       "omnitrace-bin;${TEST_LABELS}"
                       PASS_REGULAR_EXPRESSION
                       "${TEST_PASS_REGEX}"
                       FAIL_REGULAR_EXPRESSION
                       "${TEST_FAIL_REGEX}"
                       SKIP_REGULAR_EXPRESSION
                       "${TEST_SKIP_REGEX}"
                       ${TEST_PROPERTIES})
        if(TEST_ADD_INVERSE)
            add_test(
                NAME ${TEST_NAME}-inverse
                COMMAND ${TEST_COMMAND} ${TEST_ARGS}
                WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})

            set_tests_properties(
                ${TEST_NAME}-inverse
                PROPERTIES ENVIRONMENT
                           "${TEST_ENVIRONMENT_INV}"
                           TIMEOUT
                           ${TEST_TIMEOUT}
                           DEPENDS
                           "${TEST_DEPENDS}"
                           LABELS
                           "omnitrace-bin;${TEST_LABELS}"
                           PASS_REGULAR_EXPRESSION
                           "${TEST_FAIL_REGEX}"
                           FAIL_REGULAR_EXPRESSION
                           "${TEST_PASS_REGEX}"
                           SKIP_REGULAR_EXPRESSION
                           "${TEST_SKIP_REGEX}"
                           WILL_FAIL
                           ON
                           ${TEST_PROPERTIES})
        endif()
    elseif(TARGET ${TEST_TARGET})
        add_test(
            NAME ${TEST_NAME}
            COMMAND $<TARGET_FILE:${TEST_TARGET}> ${TEST_ARGS}
            WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})

        set_tests_properties(
            ${TEST_NAME}
            PROPERTIES ENVIRONMENT
                       "${TEST_ENVIRONMENT}"
                       TIMEOUT
                       ${TEST_TIMEOUT}
                       DEPENDS
                       "${TEST_DEPENDS}"
                       LABELS
                       "omnitrace-bin;${TEST_LABELS}"
                       PASS_REGULAR_EXPRESSION
                       "${TEST_PASS_REGEX}"
                       FAIL_REGULAR_EXPRESSION
                       "${TEST_FAIL_REGEX}"
                       SKIP_REGULAR_EXPRESSION
                       "${TEST_SKIP_REGEX}"
                       ${TEST_PROPERTIES})
        if(TEST_ADD_INVERSE)
            add_test(
                NAME ${TEST_NAME}-inverse
                COMMAND $<TARGET_FILE:${TEST_TARGET}> ${TEST_ARGS}
                WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY})

            set_tests_properties(
                ${TEST_NAME}-inverse
                PROPERTIES ENVIRONMENT
                           "${TEST_ENVIRONMENT_INV}"
                           TIMEOUT
                           ${TEST_TIMEOUT}
                           DEPENDS
                           "${TEST_DEPENDS}"
                           LABELS
                           "omnitrace-bin;${TEST_LABELS}"
                           PASS_REGULAR_EXPRESSION
                           "${TEST_FAIL_REGEX}"
                           FAIL_REGULAR_EXPRESSION
                           "${TEST_PASS_REGEX}"
                           SKIP_REGULAR_EXPRESSION
                           "${TEST_SKIP_REGEX}"
                           WILL_FAIL
                           ON
                           ${TEST_PROPERTIES})
        endif()
    elseif(OMNITRACE_BUILD_TESTING)
        message(FATAL_ERROR "Error! ${TEST_TARGET} does not exist")
    endif()
endfunction()

omnitrace_add_bin_test(
    NAME omnitrace-instrument-help
    TARGET omnitrace-instrument
    ARGS --help
    LABELS omnitrace-instrument
    TIMEOUT 45
    PASS_REGEX
        ".*\\\[omnitrace-instrument\\\] Usage:.*\\\[DEBUG OPTIONS\\\].*\\\[MODE OPTIONS\\\].*\\\[LIBRARY OPTIONS\\\].*\\\[SYMBOL SELECTION OPTIONS\\\].*\\\[RUNTIME OPTIONS\\\].*\\\[GRANULARITY OPTIONS\\\].*\\\[DYNINST OPTIONS\\\].*"
    )

# on RedHat, /usr/bin/ls is a script for `coreutils --coreutils-prog=ls`
if(EXISTS /usr/bin/coreutils)
    set(LS_NAME "coreutils")
    set(LS_ARGS "--coreutils-prog=ls")
else()
    set(LS_NAME ls)
    set(LS_ARGS)
endif()

omnitrace_add_bin_test(
    NAME omnitrace-instrument-simulate-ls
    TARGET omnitrace-instrument
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
    TIMEOUT 240)

omnitrace_add_bin_test(
    NAME omnitrace-instrument-simulate-ls-check
    DEPENDS omnitrace-instrument-simulate-ls
    COMMAND ls omnitrace-tests-output/omnitrace-instrument-simulate-ls/instrumentation
    TIMEOUT 60
    PASS_REGEX
        ".*available.json.*available.txt.*available.xml.*excluded.json.*excluded.txt.*excluded.xml.*instrumented.json.*instrumented.txt.*instrumented.xml.*overlapping.json.*overlapping.txt.*overlapping.xml.*"
    FAIL_REGEX "No such file or directory|not found")

omnitrace_add_bin_test(
    ADD_INVERSE
    NAME omnitrace-instrument-simulate-lib
    TARGET omnitrace-instrument
    ARGS --print-available functions -v 2 -- $<TARGET_FILE:omnitrace-user-library>
    LABELS "simulate"
    TIMEOUT 120
    PASS_REGEX
        "\\\[omnitrace\\\]\\\[exe\\\] Runtime instrumentation is not possible!(.*)\n(.*)\\\[omnitrace\\\]\\\[exe\\\] Switching to binary rewrite mode and assuming '--simulate --all-functions'"
    )

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/omnitrace-tests-output/tmp)

omnitrace_add_bin_test(
    NAME omnitrace-instrument-simulate-lib-basename
    TARGET omnitrace-instrument
    ARGS --print-available
         functions
         -v
         2
         -o
         ${PROJECT_BINARY_DIR}/omnitrace-tests-output/omnitrace-instrument-simulate-lib-basename/${CMAKE_SHARED_LIBRARY_PREFIX}$<TARGET_FILE_BASE_NAME:omnitrace-user-library>${CMAKE_SHARED_LIBRARY_SUFFIX}
         --
         ${CMAKE_SHARED_LIBRARY_PREFIX}$<TARGET_FILE_BASE_NAME:omnitrace-user-library>${CMAKE_SHARED_LIBRARY_SUFFIX}
    LABELS "simulate"
    TIMEOUT 120
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/omnitrace-tests-output/tmp)

omnitrace_add_bin_test(
    NAME omnitrace-instrument-write-log
    TARGET omnitrace-instrument
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
    PASS_REGEX "Opening .*/instrumentation/user.log")

omnitrace_add_bin_test(
    NAME omnitrace-instrument-write-log-check
    DEPENDS omnitrace-instrument-write-log
    COMMAND ls
            omnitrace-tests-output/omnitrace-instrument-write-log/instrumentation/user.log
    LABELS "log"
    TIMEOUT 60
    PASS_REGEX "user.log"
    FAIL_REGEX "No such file or directory|not found")

omnitrace_add_bin_test(
    NAME omnitrace-avail-help
    TARGET omnitrace-avail
    ARGS --help
    LABELS "omnitrace-avail"
    TIMEOUT 45
    PASS_REGEX
        ".*\\\[omnitrace-avail\\\] Usage:.*\\\[DEBUG OPTIONS\\\].*\\\[INFO OPTIONS\\\].*\\\[FILTER OPTIONS\\\].*\\\[COLUMN OPTIONS\\\].*\\\[DISPLAY OPTIONS\\\].*\\\[OUTPUT OPTIONS\\\].*"
    )

omnitrace_add_bin_test(
    NAME omnitrace-avail-all
    TARGET omnitrace-avail
    ARGS --all
    LABELS "omnitrace-avail"
    TIMEOUT 45)

omnitrace_add_bin_test(
    NAME omnitrace-avail-all-expand-keys
    TARGET omnitrace-avail
    ARGS --all --expand-keys
    LABELS "omnitrace-avail"
    TIMEOUT 45
    FAIL_REGEX "%[a-zA-Z_]%")

omnitrace_add_bin_test(
    NAME omnitrace-avail-all-only-available-alphabetical
    TARGET omnitrace-avail
    ARGS --all --available --alphabetical --debug --output
         ${CMAKE_CURRENT_BINARY_DIR}/omnitrace-avail-all-only-available-alphabetical.log
    LABELS "omnitrace-avail"
    TIMEOUT 45
    PROPERTIES
        ATTACHED_FILES
        ${CMAKE_CURRENT_BINARY_DIR}/omnitrace-avail-all-only-available-alphabetical.log)

omnitrace_add_bin_test(
    NAME omnitrace-avail-all-csv
    TARGET omnitrace-avail
    ARGS --all --csv --csv-separator "#"
    LABELS "omnitrace-avail"
    TIMEOUT 45
    PASS_REGEX
        "COMPONENT#AVAILABLE#VALUE_TYPE#STRING_IDS#FILENAME#DESCRIPTION#CATEGORY#.*ENVIRONMENT VARIABLE#VALUE#DATA TYPE#DESCRIPTION#CATEGORIES#.*HARDWARE COUNTER#DEVICE#AVAILABLE#DESCRIPTION#"
    )

omnitrace_add_bin_test(
    NAME omnitrace-avail-filter-wall-clock-available
    TARGET omnitrace-avail
    ARGS -r wall_clock -C --available
    LABELS "omnitrace-avail"
    TIMEOUT 45
    PASS_REGEX
        "\\\|[-]+\\\|\n\\\|[ ]+COMPONENT[ ]+\\\|\n\\\|[-]+\\\|\n\\\| (wall_clock)[ ]+\\\|\n\\\| (sampling_wall_clock)[ ]+\\\|\n\\\|[-]+\\\|"
    )

omnitrace_add_bin_test(
    NAME omnitrace-avail-category-filter-omnitrace
    TARGET omnitrace-avail
    ARGS --categories settings::omnitrace --brief
    LABELS "omnitrace-avail"
    TIMEOUT 45
    PASS_REGEX "OMNITRACE_(SETTINGS_DESC|OUTPUT_FILE|OUTPUT_PREFIX)"
    FAIL_REGEX
        "OMNITRACE_(ADD_SECONDARY|SCIENTIFIC|PRECISION|MEMORY_PRECISION|TIMING_PRECISION)"
    )

omnitrace_add_bin_test(
    NAME omnitrace-avail-category-filter-timemory
    TARGET omnitrace-avail
    ARGS --categories settings::timemory --brief --advanced
    LABELS "omnitrace-avail"
    TIMEOUT 45
    PASS_REGEX
        "OMNITRACE_(ADD_SECONDARY|SCIENTIFIC|PRECISION|MEMORY_PRECISION|TIMING_PRECISION)"
    FAIL_REGEX "OMNITRACE_(SETTINGS_DESC|OUTPUT_FILE)")

omnitrace_add_bin_test(
    NAME omnitrace-avail-regex-negation
    TARGET omnitrace-avail
    ARGS -R
         omnitrace
         ~timemory
         ~critical_trace
         -r
         _P
         ~PERFETTO
         ~PROCESS_SAMPLING
         ~KOKKOSP
         --csv
         --brief
         --advanced
    LABELS "omnitrace-avail"
    TIMEOUT 45
    PASS_REGEX
        "ENVIRONMENT VARIABLE,[ \n]+OMNITRACE_THREAD_POOL_SIZE,[ \n]+OMNITRACE_USE_PID,[ \n]+"
    FAIL_REGEX "OMNITRACE_USE_PERFETTO")

string(REPLACE "+" "\\\+" _AVAIL_CFG_PATH
               "${PROJECT_BINARY_DIR}/omnitrace-tests-output/omnitrace-avail/omnitrace-")
# use of TWD == Test Working Directory (added by function)
omnitrace_add_bin_test(
    NAME omnitrace-avail-write-config
    TARGET omnitrace-avail
    ARGS -G
         %env{TWD}%/omnitrace-tests-output/omnitrace-avail/omnitrace-test.cfg
         -F
         txt
         json
         xml
         --force
         --all
         -c
         omnitrace
    TIMEOUT 45
    LABELS "omnitrace-avail"
    PASS_REGEX
        "Outputting JSON configuration file '${_AVAIL_CFG_PATH}test\\\.json'(.*)Outputting XML configuration file '${_AVAIL_CFG_PATH}test\\\.xml'(.*)Outputting text configuration file '${_AVAIL_CFG_PATH}test\\\.cfg'(.*)"
    )

omnitrace_add_bin_test(
    NAME omnitrace-avail-write-config-tweak
    TARGET omnitrace-avail
    ARGS -G %env{TWD}%/omnitrace-tests-output/omnitrace-avail/omnitrace-tweak.cfg -F txt
         json xml --force
    TIMEOUT 45
    LABELS "omnitrace-avail"
    ENVIRONMENT "OMNITRACE_USE_PERFETTO=OFF;OMNITRACE_USE_TIMEMORY=ON"
    PASS_REGEX
        "Outputting JSON configuration file '${_AVAIL_CFG_PATH}tweak\\\.json'(.*)Outputting XML configuration file '${_AVAIL_CFG_PATH}tweak\\\.xml'(.*)Outputting text configuration file '${_AVAIL_CFG_PATH}tweak\\\.cfg'(.*)"
    )

omnitrace_add_bin_test(
    NAME omnitrace-avail-list-keys
    TARGET omnitrace-avail
    ARGS --list-keys --expand-keys
    TIMEOUT 45
    LABELS "omnitrace-avail"
    PASS_REGEX "Output Keys:\n(.*)%argv%(.*)%argv_hash%")

omnitrace_add_bin_test(
    NAME omnitrace-avail-list-keys-markdown
    TARGET omnitrace-avail
    ARGS --list-keys --expand-keys --markdown
    TIMEOUT 45
    LABELS "omnitrace-avail;markdown"
    PASS_REGEX "(.*)`%argv%`(.*)`%argv_hash%`")

omnitrace_add_bin_test(
    NAME omnitrace-avail-list-categories
    TARGET omnitrace-avail
    ARGS --list-categories
    TIMEOUT 45
    LABELS "omnitrace-avail"
    PASS_REGEX " component::(.*) hw_counters::(.*) settings::")

omnitrace_add_bin_test(
    NAME omnitrace-avail-core-categories
    TARGET omnitrace-avail
    ARGS -c core
    TIMEOUT 45
    LABELS "omnitrace-avail"
    PASS_REGEX
        "OMNITRACE_CONFIG_FILE(.*)OMNITRACE_ENABLED(.*)OMNITRACE_SUPPRESS_CONFIG(.*)OMNITRACE_SUPPRESS_PARSING(.*)OMNITRACE_VERBOSE"
    )

omnitrace_add_bin_test(
    NAME omnitrace-run-help
    TARGET omnitrace-run
    ARGS --help
    TIMEOUT 45
    LABELS "omnitrace-run")

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/omnitrace-tests-config")
file(
    WRITE "${PROJECT_BINARY_DIR}/omnitrace-tests-config/empty.cfg"
    "
#
# empty config file
#
")

add_executable(sleeper ${CMAKE_CURRENT_SOURCE_DIR}/sleeper.cpp)
set_target_properties(
    sleeper PROPERTIES BUILD_TYPE RelWithDebInfo RUNTIME_OUTPUT_DIRECTORY
                                                 ${PROJECT_BINARY_DIR}/bin/testing)

omnitrace_add_bin_test(
    NAME omnitrace-run-args
    TARGET omnitrace-run
    ARGS --monochrome
         --debug=false
         -v
         1
         -c
         %env{TWD}%/omnitrace-tests-config/empty.cfg
         -o
         omnitrace-tests-output
         omnitrace-run-args-output/
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
         ${CMAKE_BINARY_DIR}/omnitrace-tests-config/tmpdir
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
         --
         $<TARGET_FILE:sleeper>
         5
    TIMEOUT 45
    LABELS "omnitrace-run")
