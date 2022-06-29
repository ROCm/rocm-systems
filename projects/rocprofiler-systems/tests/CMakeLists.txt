if(NOT OMNITRACE_DYNINST_API_RT_DIR AND OMNITRACE_DYNINST_API_RT)
    get_filename_component(OMNITRACE_DYNINST_API_RT_DIR "${OMNITRACE_DYNINST_API_RT}"
                           DIRECTORY)
endif()

include(ProcessorCount)
if(NOT DEFINED NUM_PROCS_REAL)
    processorcount(NUM_PROCS_REAL)
endif()

if(NOT DEFINED NUM_PROCS)
    set(NUM_PROCS 2)
endif()

math(EXPR NUM_THREADS "${NUM_PROCS_REAL} + (${NUM_PROCS_REAL} / 2)")
if(NUM_THREADS GREATER 12)
    set(NUM_THREADS 12)
endif()

if(OMNITRACE_BUILD_DYNINST)
    set(OMNITRACE_DYNINST_API_RT_DIR
        "${PROJECT_BINARY_DIR}/external/dyninst/dyninstAPI_RT:${PROJECT_BINARY_DIR}/external/dyninst/dyninstAPI"
        )
endif()

set(_base_environment
    "OMNITRACE_USE_PERFETTO=ON"
    "OMNITRACE_USE_TIMEMORY=ON"
    "OMNITRACE_USE_SAMPLING=ON"
    "OMNITRACE_USE_PROCESS_SAMPLING=ON"
    "OMNITRACE_TIME_OUTPUT=OFF"
    "OMP_PROC_BIND=spread"
    "OMP_PLACES=threads"
    "OMP_NUM_THREADS=2"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    )

set(_lock_environment
    "OMNITRACE_USE_SAMPLING=OFF"
    "OMNITRACE_USE_PROCESS_SAMPLING=OFF"
    "OMNITRACE_CRITICAL_TRACE=ON"
    "OMNITRACE_COLLAPSE_THREADS=ON"
    "OMNITRACE_TRACE_THREAD_LOCKS=ON"
    "OMNITRACE_COUT_OUTPUT=ON"
    "OMNITRACE_TIME_OUTPUT=OFF"
    "OMNITRACE_FLAT_PROFILE=ON"
    "OMNITRACE_TIMELINE_PROFILE=OFF"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    )

set(_ompt_environment
    "OMNITRACE_USE_PERFETTO=ON"
    "OMNITRACE_USE_TIMEMORY=ON"
    "OMNITRACE_TIME_OUTPUT=OFF"
    "OMNITRACE_USE_OMPT=ON"
    "OMNITRACE_CRITICAL_TRACE=OFF"
    "OMP_PROC_BIND=spread"
    "OMP_PLACES=threads"
    "OMP_NUM_THREADS=2"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    )

set(_perfetto_environment
    "OMNITRACE_USE_PERFETTO=ON"
    "OMNITRACE_USE_TIMEMORY=OFF"
    "OMNITRACE_USE_SAMPLING=ON"
    "OMNITRACE_USE_PROCESS_SAMPLING=ON"
    "OMNITRACE_TIME_OUTPUT=OFF"
    "OMP_PROC_BIND=spread"
    "OMP_PLACES=threads"
    "OMP_NUM_THREADS=2"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    )

set(_timemory_environment
    "OMNITRACE_USE_PERFETTO=OFF"
    "OMNITRACE_USE_TIMEMORY=ON"
    "OMNITRACE_USE_SAMPLING=ON"
    "OMNITRACE_USE_PROCESS_SAMPLING=ON"
    "OMNITRACE_TIME_OUTPUT=OFF"
    "OMP_PROC_BIND=spread"
    "OMP_PLACES=threads"
    "OMP_NUM_THREADS=2"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    )

set(_test_environment ${_base_environment} "OMNITRACE_CRITICAL_TRACE=OFF")

set(_python_environment
    "OMNITRACE_USE_PERFETTO=ON"
    "OMNITRACE_USE_TIMEMORY=ON"
    "OMNITRACE_USE_SAMPLING=OFF"
    "OMNITRACE_USE_PROCESS_SAMPLING=ON"
    "OMNITRACE_TIME_OUTPUT=OFF"
    "OMNITRACE_TREE_OUTPUT=OFF"
    "OMNITRACE_USE_PID=OFF"
    "OMNITRACE_TIMEMORY_COMPONENTS=trip_count"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    "PYTHONPATH=${PROJECT_BINARY_DIR}/lib/python/site-packages")

set(_attach_environment
    "OMNITRACE_USE_PERFETTO=ON"
    "OMNITRACE_USE_TIMEMORY=ON"
    "OMNITRACE_USE_SAMPLING=OFF"
    "OMNITRACE_USE_PROCESS_SAMPLING=ON"
    "OMNITRACE_USE_CRITICAL_TRACE=ON"
    "OMNITRACE_USE_OMPT=ON"
    "OMNITRACE_USE_KOKKOSP=ON"
    "OMNITRACE_TIME_OUTPUT=OFF"
    "OMNITRACE_USE_PID=OFF"
    "OMP_PROC_BIND=spread"
    "OMP_PLACES=threads"
    "OMP_NUM_THREADS=2"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    )

# -------------------------------------------------------------------------------------- #

set(MPIEXEC_EXECUTABLE_ARGS)
option(
    OMNITRACE_CI_MPI_RUN_AS_ROOT
    "Enabled --allow-run-as-root in MPI tests with OpenMPI. Enable with care! Should only be in docker containers"
    OFF)
mark_as_advanced(OMNITRACE_CI_MPI_RUN_AS_ROOT)
if(OMNITRACE_CI_MPI_RUN_AS_ROOT)
    execute_process(
        COMMAND ${MPIEXEC_EXECUTABLE} --allow-run-as-root --help
        RESULT_VARIABLE _mpiexec_allow_run_as_root
        OUTPUT_QUIET ERROR_QUIET)
    if(_mpiexec_allow_run_as_root EQUAL 0)
        list(APPEND MPIEXEC_EXECUTABLE_ARGS --allow-run-as-root)
    endif()
endif()

# -------------------------------------------------------------------------------------- #

function(OMNITRACE_ADD_TEST)
    foreach(_PREFIX RUNTIME REWRITE REWRITE_RUN)
        foreach(_TYPE PASS FAIL SKIP)
            list(APPEND _REGEX_OPTS "${_PREFIX}_${_TYPE}_REGEX")
        endforeach()
    endforeach()
    set(_KWARGS REWRITE_ARGS RUNTIME_ARGS RUN_ARGS ENVIRONMENT LABELS PROPERTIES
                ${_REGEX_OPTS})

    cmake_parse_arguments(
        TEST
        "SKIP_BASELINE;SKIP_REWRITE;SKIP_RUNTIME;SKIP_SAMPLING" # options
        "NAME;TARGET;MPI;NUM_PROCS;REWRITE_TIMEOUT;RUNTIME_TIMEOUT" # single value args
        "${_KWARGS}" # multiple value args
        ${ARGN})

    if("${TEST_MPI}" STREQUAL "")
        set(TEST_MPI OFF)
    endif()

    list(INSERT TEST_REWRITE_ARGS 0 --print-instrumented functions)
    list(INSERT TEST_RUNTIME_ARGS 0 --print-instrumented functions)

    if(NOT DEFINED TEST_NUM_PROCS)
        set(TEST_NUM_PROCS ${NUM_PROCS})
    endif()

    if(NUM_PROCS EQUAL 0)
        set(TEST_NUM_PROCS 0)
    endif()

    if(NOT TEST_REWRITE_TIMEOUT)
        set(TEST_REWRITE_TIMEOUT 120)
    endif()

    if(NOT TEST_RUNTIME_TIMEOUT)
        set(TEST_RUNTIME_TIMEOUT 300)
    endif()

    if(NOT DEFINED TEST_ENVIRONMENT OR "${TEST_ENVIRONMENT}" STREQUAL "")
        set(TEST_ENVIRONMENT "${_test_environment}")
    endif()

    list(APPEND TEST_ENVIRONMENT "OMNITRACE_CI=ON")

    if(TARGET ${TEST_TARGET})
        if(DEFINED TEST_MPI
           AND ${TEST_MPI}
           AND TEST_NUM_PROCS GREATER 0)
            if(NOT TEST_NUM_PROCS GREATER NUM_PROCS_REAL)
                set(COMMAND_PREFIX ${MPIEXEC_EXECUTABLE} ${MPIEXEC_EXECUTABLE_ARGS}
                                   ${MPIEXEC_NUMPROC_FLAG} ${TEST_NUM_PROCS})
                list(APPEND TEST_LABELS mpi parallel-${TEST_NUM_PROCS})
                list(APPEND TEST_PROPERTIES PARALLEL_LEVEL ${TEST_NUM_PROCS})
            else()
                set(COMMAND_PREFIX ${MPIEXEC_EXECUTABLE} ${MPIEXEC_EXECUTABLE_ARGS}
                                   ${MPIEXEC_NUMPROC_FLAG} 1)
            endif()
        else()
            list(APPEND TEST_ENVIRONMENT "OMNITRACE_USE_PID=OFF")
        endif()

        if(NOT TEST_SKIP_BASELINE)
            add_test(
                NAME ${TEST_NAME}-baseline
                COMMAND ${COMMAND_PREFIX} $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
                WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)
        endif()

        if(NOT TEST_SKIP_REWRITE)
            add_test(
                NAME ${TEST_NAME}-binary-rewrite
                COMMAND
                    $<TARGET_FILE:omnitrace-exe> -o
                    $<TARGET_FILE_DIR:${TEST_TARGET}>/${TEST_NAME}.inst
                    ${TEST_REWRITE_ARGS} -- $<TARGET_FILE:${TEST_TARGET}>
                WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)

            if(NOT TEST_SKIP_SAMPLING)
                add_test(
                    NAME ${TEST_NAME}-binary-rewrite-sampling
                    COMMAND
                        $<TARGET_FILE:omnitrace-exe> -o
                        $<TARGET_FILE_DIR:${TEST_TARGET}>/${TEST_NAME}.samp -M sampling
                        ${TEST_REWRITE_ARGS} -- $<TARGET_FILE:${TEST_TARGET}>
                    WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)
            endif()

            add_test(
                NAME ${TEST_NAME}-binary-rewrite-run
                COMMAND
                    ${COMMAND_PREFIX} $<TARGET_FILE_DIR:${TEST_TARGET}>/${TEST_NAME}.inst
                    ${TEST_RUN_ARGS}
                WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)

            if(NOT TEST_SKIP_SAMPLING)
                add_test(
                    NAME ${TEST_NAME}-binary-rewrite-run-sampling
                    COMMAND
                        ${COMMAND_PREFIX}
                        $<TARGET_FILE_DIR:${TEST_TARGET}>/${TEST_NAME}.samp
                        ${TEST_RUN_ARGS}
                    WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)
            endif()
        endif()

        if(NOT TEST_SKIP_RUNTIME)
            add_test(
                NAME ${TEST_NAME}-runtime-instrument
                COMMAND $<TARGET_FILE:omnitrace-exe> ${TEST_RUNTIME_ARGS} --
                        $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
                WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)

            if(NOT TEST_SKIP_SAMPLING)
                add_test(
                    NAME ${TEST_NAME}-runtime-instrument-sampling
                    COMMAND $<TARGET_FILE:omnitrace-exe> -M sampling ${TEST_RUNTIME_ARGS}
                            -- $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
                    WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)
            endif()
        endif()

        if(TEST ${TEST_NAME}-binary-rewrite-run)
            set_tests_properties(${TEST_NAME}-binary-rewrite-run
                                 PROPERTIES DEPENDS ${TEST_NAME}-binary-rewrite)
        endif()

        if(TEST ${TEST_NAME}-binary-rewrite-run-sampling)
            set_tests_properties(${TEST_NAME}-binary-rewrite-run-sampling
                                 PROPERTIES DEPENDS ${TEST_NAME}-binary-rewrite-sampling)
        endif()

        foreach(
            _TEST
            baseline binary-rewrite binary-rewrite-run binary-rewrite-sampling
            binary-rewrite-run-sampling runtime-instrument runtime-instrument-sampling)
            string(REGEX REPLACE "-run(-|/)" "\\1" _prefix "${TEST_NAME}-${_TEST}/")
            set(_environ "${TEST_ENVIRONMENT}")
            set(_labels "${_TEST}")
            set(_timeout ${TEST_REWRITE_TIMEOUT})
            list(APPEND _environ "OMNITRACE_OUTPUT_PATH=omnitrace-tests-output"
                 "OMNITRACE_OUTPUT_PREFIX=${_prefix}")
            string(REPLACE "-run" "" _labels "${_TEST}")
            string(REPLACE "-sampling" ";sampling" _labels "${_labels}")
            if(TEST_TARGET)
                list(APPEND _labels "${TEST_TARGET}")
            endif()
            if("${_TEST}" MATCHES "runtime-instrument")
                set(_timeout ${TEST_RUNTIME_TIMEOUT})
            endif()
            if("${_TEST}" MATCHES "binary-rewrite-run")
                list(APPEND _labels "binary-rewrite-run")
            endif()
            set(_props)
            if("${_TEST}" MATCHES "run|baseline")
                set(_props ${TEST_PROPERTIES})
                if(NOT "RUN_SERIAL" IN_LIST _props)
                    list(APPEND _props RUN_SERIAL ON)
                endif()
            endif()

            if("${_TEST}" MATCHES "binary-rewrite-run")
                set(_REGEX_VAR REWRITE_RUN)
            elseif("${_TEST}" MATCHES "runtime-instrument")
                set(_REGEX_VAR RUNTIME)
            elseif("${_TEST}" MATCHES "binary-rewrite")
                set(_REGEX_VAR REWRITE)
            else()
                set(_REGEX_VAR)
            endif()

            foreach(_TYPE PASS FAIL SKIP)
                if(_REGEX_VAR)
                    set(_${_TYPE}_REGEX TEST_${_REGEX_VAR}_${_TYPE}_REGEX)
                else()
                    set(_${_TYPE}_REGEX)
                endif()
            endforeach()

            if(TEST ${TEST_NAME}-${_TEST})
                set_tests_properties(
                    ${TEST_NAME}-${_TEST}
                    PROPERTIES ENVIRONMENT
                               "${_environ}"
                               TIMEOUT
                               ${_timeout}
                               LABELS
                               "${_labels};${TEST_LABELS}"
                               PASS_REGULAR_EXPRESSION
                               "${${_PASS_REGEX}}"
                               FAIL_REGULAR_EXPRESSION
                               "${${_FAIL_REGEX}}"
                               SKIP_REGULAR_EXPRESSION
                               "${${_SKIP_REGEX}}"
                               ${_props})
            endif()
        endforeach()
    endif()
endfunction()

# -------------------------------------------------------------------------------------- #

function(OMNITRACE_ADD_PYTHON_TEST)
    if(NOT OMNITRACE_USE_PYTHON)
        return()
    endif()

    cmake_parse_arguments(
        TEST
        "STANDALONE" # options
        "NAME;FILE;TIMEOUT;PYTHON_EXECUTABLE;PYTHON_VERSION" # single value args
        "PROFILE_ARGS;RUN_ARGS;ENVIRONMENT;LABELS;PROPERTIES;PASS_REGEX;FAIL_REGEX;SKIP_REGEX;DEPENDS;COMMAND" # multiple
        # value args
        ${ARGN})

    if(NOT TEST_TIMEOUT)
        set(TEST_TIMEOUT 120)
    endif()

    set(PYTHON_EXECUTABLE "${TEST_PYTHON_EXECUTABLE}")

    if(NOT DEFINED TEST_ENVIRONMENT OR "${TEST_ENVIRONMENT}" STREQUAL "")
        set(TEST_ENVIRONMENT "${_python_environment}")
    endif()

    list(APPEND TEST_LABELS "python" "python-${TEST_PYTHON_VERSION}")

    if(NOT TEST_COMMAND)
        list(APPEND TEST_ENVIRONMENT "OMNITRACE_CI=ON"
             "OMNITRACE_OUTPUT_PATH=omnitrace-tests-output"
             "OMNITRACE_OUTPUT_PREFIX=${TEST_NAME}/${TEST_PYTHON_VERSION}/")
        get_filename_component(_TEST_FILE "${TEST_FILE}" NAME)
        set(_TEST_FILE
            ${CMAKE_BINARY_DIR}/python/tests/${TEST_PYTHON_VERSION}/${_TEST_FILE})
        configure_file(${TEST_FILE} ${_TEST_FILE} @ONLY)
        if(TEST_STANDALONE)
            add_test(
                NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}
                COMMAND ${TEST_PYTHON_EXECUTABLE} ${_TEST_FILE} ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
        else()
            add_test(
                NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}
                COMMAND ${TEST_PYTHON_EXECUTABLE} -m omnitrace ${TEST_PROFILE_ARGS} --
                        ${_TEST_FILE} ${TEST_RUN_ARGS}
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
        endif()
    else()
        list(APPEND TEST_LABELS "python-check" "python-${TEST_PYTHON_VERSION}-check")
        add_test(
            NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}
            COMMAND ${TEST_COMMAND} ${TEST_FILE}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
        add_test(
            NAME ${TEST_NAME}-${TEST_PYTHON_VERSION}-inverse
            COMMAND ${TEST_COMMAND} ${TEST_FILE}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()

    foreach(_TEST ${TEST_NAME}-${TEST_PYTHON_VERSION}
                  ${TEST_NAME}-${TEST_PYTHON_VERSION}-inverse)

        if(NOT TEST "${_TEST}")
            continue()
        endif()

        set(_TEST_PROPERTIES "${TEST_PROPERTIES}")
        if(NOT "${_TEST}" MATCHES "inverse")
            # assign pass variable to pass regex
            set(_PASS_REGEX TEST_PASS_REGEX)
            # assign fail variable to fail regex
            set(_FAIL_REGEX TEST_FAIL_REGEX)
        else()
            # assign pass variable to fail regex
            set(_PASS_REGEX TEST_FAIL_REGEX)
            # assign fail variable to pass regex
            set(_FAIL_REGEX TEST_PASS_REGEX)
            # set to will fail
            list(APPEND _TEST_PROPERTIES WILL_FAIL ON)
        endif()

        set_tests_properties(
            ${_TEST}
            PROPERTIES ENVIRONMENT
                       "${TEST_ENVIRONMENT}"
                       TIMEOUT
                       ${TEST_TIMEOUT}
                       LABELS
                       "${TEST_LABELS}"
                       DEPENDS
                       "${TEST_DEPENDS}"
                       PASS_REGULAR_EXPRESSION
                       "${${_PASS_REGEX}}"
                       FAIL_REGULAR_EXPRESSION
                       "${${_FAIL_REGEX}}"
                       SKIP_REGULAR_EXPRESSION
                       "${TEST_SKIP_REGEX}"
                       REQUIRED_FILES
                       "${TEST_FILE}"
                       ${_TEST_PROPERTIES})
    endforeach()
endfunction()

# -------------------------------------------------------------------------------------- #
#
# general config file tests
#
# -------------------------------------------------------------------------------------- #
file(
    WRITE ${CMAKE_CURRENT_BINARY_DIR}/invalid.cfg
    "
FOOBAR = ON
")

if(TARGET parallel-overhead)
    set(_CONFIG_TEST_EXE $<TARGET_FILE:parallel-overhead>)
else()
    set(_CONFIG_TEST_EXE ls)
endif()

add_test(
    NAME omnitrace-invalid-config
    COMMAND $<TARGET_FILE:omnitrace-exe> -- ${_CONFIG_TEST_EXE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(
    omnitrace-invalid-config
    PROPERTIES ENVIRONMENT
               "OMNITRACE_CONFIG_FILE=${CMAKE_CURRENT_BINARY_DIR}/invalid.cfg" TIMEOUT
               120 LABELS "config" WILL_FAIL ON)

add_test(
    NAME omnitrace-missing-config
    COMMAND $<TARGET_FILE:omnitrace-exe> -- ${_CONFIG_TEST_EXE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(
    omnitrace-missing-config
    PROPERTIES ENVIRONMENT
               "OMNITRACE_CONFIG_FILE=${CMAKE_CURRENT_BINARY_DIR}/missing.cfg" TIMEOUT
               120 LABELS "config" WILL_FAIL ON)

# -------------------------------------------------------------------------------------- #
#
# binary-rewrite and runtime-instrumentation tests
#
# -------------------------------------------------------------------------------------- #

omnitrace_add_test(
    NAME transpose
    TARGET transpose
    MPI ${TRANSPOSE_USE_MPI}
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS -e -v 2 --print-instructions -E uniform_int_distribution
    RUNTIME_ARGS
        -e
        -v
        1
        --label
        file
        line
        return
        args
        -E
        uniform_int_distribution
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=ON")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING SKIP_RUNTIME
    NAME transpose-loops
    TARGET transpose
    LABELS "loops"
    MPI ${TRANSPOSE_USE_MPI}
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS
        -e
        -v
        2
        --label
        return
        args
        -l
        -i
        8
        -E
        uniform_int_distribution
    RUN_ARGS 2 100 50
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure transpose")

omnitrace_add_test(
    NAME parallel-overhead
    TARGET parallel-overhead
    REWRITE_ARGS -e -v 2 --min-instructions=8
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=8
        --label
        file
        line
        return
        args
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF")

omnitrace_add_test(
    SKIP_RUNTIME SKIP_SAMPLING
    NAME parallel-overhead-locks-timemory
    TARGET parallel-overhead-locks
    LABELS "locks"
    REWRITE_ARGS -e -v 2 --min-instructions=4
    RUN_ARGS 10 4 1000
    ENVIRONMENT
        "${_lock_environment};OMNITRACE_USE_TIMEMORY=ON;OMNITRACE_USE_PERFETTO=OFF"
    REWRITE_RUN_PASS_REGEX
        "start_thread (.*) 4 (.*) pthread_mutex_lock (.*) 4000 (.*) pthread_mutex_unlock (.*) 4000"
    )

omnitrace_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_SAMPLING
    NAME parallel-overhead-locks-perfetto
    TARGET parallel-overhead-locks
    LABELS "locks"
    REWRITE_ARGS -e -v 2 --min-instructions=8
    RUN_ARGS 10 4 1000
    ENVIRONMENT
        "${_lock_environment};OMNITRACE_USE_TIMEMORY=OFF;OMNITRACE_USE_PERFETTO=ON"
    PROPERTIES WILL_FAIL ON)

omnitrace_add_test(
    NAME user-api
    TARGET user-api
    LABELS "loops"
    REWRITE_ARGS -e -v 2 -l --min-instructions=8 -E custom_push_region
    RUNTIME_ARGS
        -e
        -v
        1
        -l
        --min-instructions=8
        -E
        custom_push_region
        --label
        file
        line
        return
        args
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure")

omnitrace_add_test(
    SKIP_RUNTIME
    NAME "mpi"
    TARGET mpi-example
    MPI ON
    NUM_PROCS 4
    REWRITE_ARGS
        -e
        -v
        2
        --label
        file
        line
        return
        args
        --min-instructions
        0
    RUNTIME_ARGS
        -e
        -v
        1
        --label
        file
        line
        return
        args
        --min-instructions
        0
    ENVIRONMENT "${_base_environment}")

omnitrace_add_test(
    NAME lulesh
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    NUM_PROCS 8
    LABELS "kokkos"
    REWRITE_ARGS -e -v 2 --label file line return args
    RUNTIME_ARGS
        -e
        -v
        1
        --label
        file
        line
        return
        args
        -ME
        [==[lib(gomp|m-)]==]
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT
        "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=ON;KOKKOS_PROFILE_LIBRARY=libomnitrace.so"
    )

omnitrace_add_test(
    SKIP_BASELINE
    NAME lulesh-kokkosp
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    NUM_PROCS 8
    LABELS "kokkos"
    REWRITE_ARGS -e -v 2
    RUNTIME_ARGS
        -e
        -v
        1
        --label
        file
        line
        return
        args
        -ME
        [==[lib(gomp|m-)]==]
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT
        "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=ON")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME lulesh-perfetto
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    NUM_PROCS 8
    LABELS "kokkos;loops"
    REWRITE_ARGS -e -v 2
    RUNTIME_ARGS
        -e
        -v
        1
        -l
        --dynamic-callsites
        --traps
        --allow-overlapping
        -ME
        [==[libgomp]==]
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT
        "${_perfetto_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=OFF")

omnitrace_add_test(
    SKIP_SAMPLING
    NAME lulesh-timemory
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    NUM_PROCS 8
    LABELS "kokkos;loops"
    REWRITE_ARGS -e -v 2 -l --dynamic-callsites --traps --allow-overlapping
    RUNTIME_ARGS
        -e
        -v
        1
        -l
        --dynamic-callsites
        -ME
        [==[libgomp]==]
        -d
        wall_clock
        peak_rss
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT
        "${_timemory_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=OFF"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure")

omnitrace_add_test(
    SKIP_SAMPLING
    NAME openmp-cg
    TARGET openmp-cg
    LABELS "openmp"
    REWRITE_ARGS -e -v 2 --instrument-loops
    RUNTIME_ARGS -e -v 1 --label return args
    REWRITE_TIMEOUT 180
    RUNTIME_TIMEOUT 360
    ENVIRONMENT "${_ompt_environment};OMNITRACE_USE_SAMPLING=OFF"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure")

omnitrace_add_test(
    SKIP_RUNTIME
    NAME openmp-lu
    TARGET openmp-lu
    LABELS "openmp"
    REWRITE_ARGS -e -v 2 --instrument-loops
    RUNTIME_ARGS -e -v 1 --label return args -E ^GOMP
    REWRITE_TIMEOUT 180
    RUNTIME_TIMEOUT 360
    ENVIRONMENT
        "${_ompt_environment};OMNITRACE_USE_SAMPLING=ON;OMNITRACE_SAMPLING_FREQ=100"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage
    TARGET code-coverage
    REWRITE_ARGS
        -e
        -v
        2
        --min-instructions=4
        -E
        ^std::
        -M
        coverage
        --coverage
        function
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        -M
        coverage
        --coverage
        function
        --module-restrict
        code.coverage
    LABELS "coverage;function-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage-hybrid
    TARGET code-coverage
    REWRITE_ARGS -e -v 2 --min-instructions=4 -E ^std:: --coverage function
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        --coverage
        function
        --module-restrict
        code.coverage
    LABELS "coverage;function-coverage;hybrid-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage-basic-blocks
    TARGET code-coverage
    REWRITE_ARGS
        -e
        -v
        2
        --min-instructions=4
        -E
        ^std::
        -M
        coverage
        --coverage
        basic_block
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        -M
        coverage
        --coverage
        basic_block
        --module-restrict
        code.coverage
    LABELS "coverage;bb-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage-basic-blocks-hybrid
    TARGET code-coverage
    REWRITE_ARGS -e -v 2 --min-instructions=4 -E ^std:: --coverage basic_block
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        --coverage
        basic_block
        --module-restrict
        code.coverage
    LABELS "coverage;bb-coverage;hybrid-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%")

# -------------------------------------------------------------------------------------- #
#
# attach tests
#
# -------------------------------------------------------------------------------------- #

set(_VALID_PTRACE_SCOPE OFF)
if(EXISTS "/proc/sys/kernel/yama/ptrace_scope")
    file(READ "/proc/sys/kernel/yama/ptrace_scope" _PTRACE_SCOPE LIMIT 1)
    if("${_PTRACE_SCOPE}" EQUAL 0)
        set(_VALID_PTRACE_SCOPE ON)
    endif()
else()
    omnitrace_message(
        AUTHOR_WARNING
        "Disabling attach tests. Run 'echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope' to enable attaching to process"
        )
endif()

if(TARGET parallel-overhead AND _VALID_PTRACE_SCOPE)
    add_test(
        NAME parallel-overhead-attach
        COMMAND
            ${CMAKE_CURRENT_LIST_DIR}/run-omnitrace-pid.sh $<TARGET_FILE:omnitrace-exe>
            -ME "\.c$" -e -v 1 --label return args file -l --
            $<TARGET_FILE:parallel-overhead> 30 8 1000
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    set(_parallel_overhead_attach_environ
        "${_attach_environment}" "OMNITRACE_OUTPUT_PATH=omnitrace-tests-output"
        "OMNITRACE_OUTPUT_PREFIX=parallel-overhead-attach/")

    set_tests_properties(
        parallel-overhead-attach
        PROPERTIES ENVIRONMENT
                   "${_parallel_overhead_attach_environ}"
                   TIMEOUT
                   300
                   LABELS
                   "parallel-overhead;attach"
                   PASS_REGULAR_EXPRESSION
                   "Outputting.*(perfetto-trace.proto).*Outputting.*(wall_clock.txt)"
                   FAIL_REGULAR_EXPRESSION
                   "Dyninst was unable to attach to the specified process")
endif()

# -------------------------------------------------------------------------------------- #
#
# python tests
#
# -------------------------------------------------------------------------------------- #

set(_INDEX 0)
foreach(_VERSION ${OMNITRACE_PYTHON_VERSIONS})
    list(GET OMNITRACE_PYTHON_ROOT_DIRS ${_INDEX} _PYTHON_ROOT_DIR)

    omnitrace_find_python(
        _PYTHON
        ROOT_DIR "${_PYTHON_ROOT_DIR}"
        COMPONENTS Interpreter)

    # ---------------------------------------------------------------------------------- #
    # python tests
    # ---------------------------------------------------------------------------------- #
    omnitrace_add_python_test(
        NAME python-external
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/external.py
        PROFILE_ARGS "--label" "file"
        RUN_ARGS -v 10 -n 5
        ENVIRONMENT "${_python_environment}")

    omnitrace_add_python_test(
        NAME python-external-exclude-inefficient
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/external.py
        PROFILE_ARGS -E "^inefficient$"
        RUN_ARGS -v 10 -n 5
        ENVIRONMENT "${_python_environment}")

    omnitrace_add_python_test(
        NAME python-builtin
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/builtin.py
        PROFILE_ARGS "-b" "--label" "file" "line"
        RUN_ARGS -v 10 -n 5
        ENVIRONMENT "${_python_environment}")

    omnitrace_add_python_test(
        STANDALONE
        NAME python-source
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/source.py
        RUN_ARGS -v 5 -n 5
        ENVIRONMENT "${_python_environment}")

    # ---------------------------------------------------------------------------------- #
    # python output tests
    # ---------------------------------------------------------------------------------- #

    if(CMAKE_VERSION VERSION_LESS "3.18.0")
        find_program(
            OMNITRACE_CAT_EXE
            NAMES cat
            PATH_SUFFIXES bin)
        if(OMNITRACE_CAT_EXE)
            set(OMNITRACE_CAT_COMMAND ${OMNITRACE_CAT_EXE})
        endif()
    else()
        set(OMNITRACE_CAT_COMMAND ${CMAKE_COMMAND} -E cat)
    endif()

    if(OMNITRACE_CAT_COMMAND)
        omnitrace_add_python_test(
            NAME python-external-check
            COMMAND ${OMNITRACE_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/python-external/${_VERSION}/trip_count.txt
            PASS_REGEX
                "(\\\[compile\\\]).*(\\\| \\\|0>>> \\\[run\\\]\\\[external.py\\\]).*(\\\| \\\|0>>> \\\|_\\\[fib\\\]\\\[external.py\\\]).*(\\\| \\\|0>>> \\\|_\\\[inefficient\\\]\\\[external.py\\\])"
            DEPENDS python-external-${_VERSION}
            ENVIRONMENT "${_python_environment}")

        omnitrace_add_python_test(
            NAME python-external-exclude-inefficient-check
            COMMAND ${OMNITRACE_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/python-external-exclude-inefficient/${_VERSION}/trip_count.txt
            FAIL_REGEX "(\\\|_inefficient).*(\\\|_sum)"
            DEPENDS python-external-exclude-inefficient-${_VERSION}
            ENVIRONMENT "${_python_environment}")

        omnitrace_add_python_test(
            NAME python-builtin-check
            COMMAND ${OMNITRACE_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/python-builtin/${_VERSION}/trip_count.txt
            PASS_REGEX "\\\[inefficient\\\]\\\[builtin.py:14\\\]"
            DEPENDS python-builtin-${_VERSION}
            ENVIRONMENT "${_python_environment}")
    else()
        omnitrace_message(
            WARNING
            "Neither 'cat' nor 'cmake -E cat' are available. Python source checks are disabled"
            )
    endif()

    function(OMNITRACE_ADD_PYTHON_VALIDATION_TEST)
        cmake_parse_arguments(TEST "" "NAME;TIMEMORY_METRIC;TIMEMORY_FILE;PERFETTO_FILE"
                              "ARGS" ${ARGN})

        omnitrace_add_python_test(
            NAME ${TEST_NAME}-validate-timemory
            COMMAND
                ${_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/validate-timemory-json.py
                -m ${TEST_TIMEMORY_METRIC} ${TEST_ARGS} -i
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/${TEST_NAME}/${_VERSION}/${TEST_TIMEMORY_FILE}
            DEPENDS ${TEST_NAME}-${_VERSION}
            PASS_REGEX
                "omnitrace-tests-output/${TEST_NAME}/${_VERSION}/${TEST_TIMEMORY_FILE} validated"
            ENVIRONMENT "${_python_environment}")

        omnitrace_add_python_test(
            NAME ${TEST_NAME}-validate-perfetto
            COMMAND
                ${_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py
                ${TEST_ARGS} -p -i
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/${TEST_NAME}/${_VERSION}/${TEST_PERFETTO_FILE}
            DEPENDS ${TEST_NAME}-${_VERSION}
            PASS_REGEX
                "omnitrace-tests-output/${TEST_NAME}/${_VERSION}/${TEST_PERFETTO_FILE} validated"
            ENVIRONMENT "${_python_environment}")
    endfunction()

    omnitrace_add_python_validation_test(
        NAME python-source
        TIMEMORY_METRIC "trip_count"
        TIMEMORY_FILE "trip_count.json"
        PERFETTO_FILE "perfetto-trace.proto"
        ARGS -l
             run
             fib
             fib
             fib
             fib
             fib
             inefficient
             _sum
             -c
             5
             5
             10
             20
             30
             10
             5
             5
             -d
             0
             1
             2
             3
             4
             5
             1
             2)

    math(EXPR _INDEX "${_INDEX} + 1")
endforeach()
