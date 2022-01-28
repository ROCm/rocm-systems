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
    "OMNITRACE_USE_SAMPLING=OFF"
    "OMNITRACE_TIME_OUTPUT=OFF"
    "OMP_PROC_BIND=spread"
    "OMP_PLACES=threads"
    "OMP_NUM_THREADS=2"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    )

set(_test_environment ${_base_environment} "OMNITRACE_CRITICAL_TRACE=OFF")

set(_fast_environment
    "OMNITRACE_USE_PERFETTO=OFF"
    "OMNITRACE_USE_TIMEMORY=OFF"
    "OMNITRACE_USE_SAMPLING=OFF"
    "OMNITRACE_CRITICAL_TRACE=OFF"
    "OMNITRACE_TIME_OUTPUT=OFF"
    "OMP_PROC_BIND=spread"
    "OMP_PLACES=threads"
    "OMP_NUM_THREADS=2"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    )

function(OMNITRACE_ADD_TEST)
    cmake_parse_arguments(
        TEST
        "" # options
        "NAME;TARGET;MPI;NUM_PROCS" # single value args
        "REWRITE_ARGS;RUNTIME_ARGS;RUN_ARGS;ENVIRONMENT;LABELS" # multiple value args
        ${ARGN})

    if("${TEST_MPI}" STREQUAL "")
        set(TEST_MPI OFF)
    endif()

    if(NOT DEFINED TEST_NUM_PROCS)
        set(TEST_NUM_PROCS ${NUM_PROCS})
    endif()

    if(NUM_PROCS EQUAL 0)
        set(TEST_NUM_PROCS 0)
    endif()

    if(NOT DEFINED TEST_ENVIRONMENT OR "${TEST_ENVIRONMENT}" STREQUAL "")
        set(TEST_ENVIRONMENT "${_test_environment}")
    endif()

    if(TARGET ${TEST_TARGET})
        if(DEFINED TEST_MPI
           AND ${TEST_MPI}
           AND TEST_NUM_PROCS GREATER 0)
            if(NOT TEST_NUM_PROCS GREATER NUM_PROCS_REAL)
                set(COMMAND_PREFIX ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
                                   ${TEST_NUM_PROCS})
            else()
                set(COMMAND_PREFIX ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1)
            endif()
        else()
            list(APPEND TEST_ENVIRONMENT "OMNITRACE_USE_PID=OFF")
        endif()

        add_test(
            NAME ${TEST_NAME}-baseline
            COMMAND $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)

        add_test(
            NAME ${TEST_NAME}-binary-rewrite
            COMMAND
                $<TARGET_FILE:omnitrace-exe> -o
                $<TARGET_FILE_DIR:${TEST_TARGET}>/${TEST_TARGET}.inst ${TEST_REWRITE_ARGS}
                -- $<TARGET_FILE:${TEST_TARGET}>
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)

        add_test(
            NAME ${TEST_NAME}-binary-rewrite-sampling
            COMMAND
                $<TARGET_FILE:omnitrace-exe> -o
                $<TARGET_FILE_DIR:${TEST_TARGET}>/${TEST_TARGET}.samp -M sampling
                ${TEST_REWRITE_ARGS} -- $<TARGET_FILE:${TEST_TARGET}>
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)

        add_test(
            NAME ${TEST_NAME}-binary-rewrite-run
            COMMAND ${COMMAND_PREFIX}
                    $<TARGET_FILE_DIR:${TEST_TARGET}>/${TEST_TARGET}.inst ${TEST_RUN_ARGS}
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)

        add_test(
            NAME ${TEST_NAME}-binary-rewrite-run-sampling
            COMMAND ${COMMAND_PREFIX}
                    $<TARGET_FILE_DIR:${TEST_TARGET}>/${TEST_TARGET}.samp ${TEST_RUN_ARGS}
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)

        add_test(
            NAME ${TEST_NAME}-runtime-instrument
            COMMAND $<TARGET_FILE:omnitrace-exe> ${TEST_RUNTIME_ARGS} --
                    $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)

        add_test(
            NAME ${TEST_NAME}-runtime-instrument-sampling
            COMMAND
                $<TARGET_FILE:omnitrace-exe> -M sampling --env
                OMNITRACE_OUTPUT_PREFIX=sampling- ${TEST_RUNTIME_ARGS} --
                $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)

        set_tests_properties(${TEST_NAME}-binary-rewrite-run
                             PROPERTIES DEPENDS ${TEST_NAME}-binary-rewrite)

        set_tests_properties(${TEST_NAME}-binary-rewrite-run-sampling
                             PROPERTIES DEPENDS ${TEST_NAME}-binary-rewrite-sampling)

        foreach(
            _TEST
            baseline binary-rewrite binary-rewrite-run binary-rewrite-sampling
            binary-rewrite-run-sampling runtime-instrument runtime-instrument-sampling)
            string(REPLACE "-run-" "-" _PREFIX "${TEST_NAME}-${_TEST}-")
            set(_environ "${TEST_ENVIRONMENT}")
            list(APPEND _environ "OMNITRACE_OUTPUT_PATH=omnitrace-tests-output"
                 "OMNITRACE_OUTPUT_PREFIX=${_PREFIX}")
            set(_LABELS "${_TEST}")
            string(REPLACE "-run" "" _LABELS "${_TEST}")
            string(REPLACE "-sampling" ";sampling" _LABELS "${_LABELS}")
            set_tests_properties(
                ${TEST_NAME}-${_TEST} PROPERTIES ENVIRONMENT "${_environ}" TIMEOUT 600
                                                 LABELS "${_LABELS};${TEST_LABELS}")
        endforeach()
    endif()
endfunction()

omnitrace_add_test(
    NAME transpose
    TARGET transpose
    MPI ${TRANSPOSE_USE_MPI}
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS -e -v 1
    RUNTIME_ARGS -e -v 1 --label file line return args
    RUN_ARGS ""
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=ON")

omnitrace_add_test(
    NAME transpose-no-save-fpr
    TARGET transpose
    MPI ${TRANSPOSE_USE_MPI}
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS -e -v 1 --dyninst-options DelayedParsing TypeChecking
    RUNTIME_ARGS
        -e
        -v
        1
        --label
        file
        line
        return
        args
        --dyninst-options
        DelayedParsing
        TypeChecking
    RUN_ARGS ""
    ENVIRONMENT "${_fast_environment}")

omnitrace_add_test(
    NAME parallel-overhead
    TARGET parallel-overhead
    REWRITE_ARGS -e -v 1 --min-address-range-loop=64
    RUNTIME_ARGS
        -e
        -v
        1
        --min-address-range-loop=64
        --label
        file
        line
        return
        args
    RUN_ARGS 10 ${NUM_THREADS}
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF")

omnitrace_add_test(
    NAME parallel-overhead-no-save-fpr
    TARGET parallel-overhead
    REWRITE_ARGS -e -v 1 --min-address-range-loop=32 --dyninst-options DelayedParsing
                 TypeChecking
    RUNTIME_ARGS
        -e
        -v
        1
        --min-address-range-loop=32
        --label
        file
        line
        return
        args
        --dyninst-options
        DelayedParsing
        TypeChecking
    RUN_ARGS 10 ${NUM_THREADS}
    ENVIRONMENT "${_fast_environment}")

omnitrace_add_test(
    NAME lulesh
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    NUM_PROCS 8
    REWRITE_ARGS -e -v 1
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
        [==['lib(gomp|m-)']==]
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF")
