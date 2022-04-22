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
    "OMNITRACE_TIME_OUTPUT=OFF"
    "OMP_PROC_BIND=spread"
    "OMP_PLACES=threads"
    "OMP_NUM_THREADS=2"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    )

set(_ompt_environment
    "OMNITRACE_USE_PERFETTO=ON"
    "OMNITRACE_USE_TIMEMORY=ON"
    "OMNITRACE_USE_SAMPLING=ON"
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
    "OMNITRACE_TIME_OUTPUT=OFF"
    "OMNITRACE_TREE_OUTPUT=OFF"
    "OMNITRACE_USE_PID=OFF"
    "OMNITRACE_TIMEMORY_COMPONENTS=trip_count"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${OMNITRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    "PYTHONPATH=${CMAKE_BINARY_DIR}/python")

# -------------------------------------------------------------------------------------- #

function(OMNITRACE_ADD_TEST)
    cmake_parse_arguments(
        TEST
        "SKIP_REWRITE;SKIP_RUNTIME;SKIP_SAMPLING" # options
        "NAME;TARGET;MPI;NUM_PROCS;REWRITE_TIMEOUT;RUNTIME_TIMEOUT" # single value args
        "REWRITE_ARGS;RUNTIME_ARGS;RUN_ARGS;ENVIRONMENT;LABELS;PROPERTIES" # multiple
                                                                           # value args
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
            COMMAND ${COMMAND_PREFIX} $<TARGET_FILE:${TEST_TARGET}> ${TEST_RUN_ARGS}
            WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_TARGET}>)

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
            if("${_TEST}" MATCHES "runtime-instrument")
                set(_timeout ${TEST_RUNTIME_TIMEOUT})
            endif()
            set(_props)
            if("${_TEST}" MATCHES "run|baseline")
                set(_props ${TEST_PROPERTIES})
                if(NOT "RUN_SERIAL" IN_LIST _props)
                    list(APPEND _props RUN_SERIAL ON)
                endif()
            endif()
            if(TEST ${TEST_NAME}-${_TEST})
                set_tests_properties(
                    ${TEST_NAME}-${_TEST}
                    PROPERTIES ENVIRONMENT "${_environ}" TIMEOUT ${_timeout} LABELS
                               "${_labels};${TEST_LABELS}" ${_props})
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
        "PROFILE_ARGS;RUN_ARGS;ENVIRONMENT;LABELS;PROPERTIES;PASS_REGULAR_EXPRESSION;FAIL_REGULAR_EXPRESSION;SKIP_REGULAR_EXPRESSION;DEPENDS;COMMAND" # multiple
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
            set(_PASS_REGEX TEST_PASS_REGULAR_EXPRESSION)
            # assign fail variable to fail regex
            set(_FAIL_REGEX TEST_FAIL_REGULAR_EXPRESSION)
        else()
            # assign pass variable to fail regex
            set(_PASS_REGEX TEST_FAIL_REGULAR_EXPRESSION)
            # assign fail variable to pass regex
            set(_FAIL_REGEX TEST_PASS_REGULAR_EXPRESSION)
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
                       "${TEST_SKIP_REGULAR_EXPRESSION}"
                       REQUIRED_FILES
                       "${TEST_FILE}"
                       ${_TEST_PROPERTIES})
    endforeach()
endfunction()

# -------------------------------------------------------------------------------------- #

omnitrace_add_test(
    NAME transpose
    TARGET transpose
    MPI ${TRANSPOSE_USE_MPI}
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS -e -v 2
    RUNTIME_ARGS -e -v 1 --label file line return args
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=ON")

omnitrace_add_test(
    NAME parallel-overhead
    TARGET parallel-overhead
    REWRITE_ARGS -e -v 2 --min-address-range-loop=64
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
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF")

omnitrace_add_test(
    NAME user-api
    TARGET user-api
    REWRITE_ARGS
        -e
        -v
        2
        -l
        --min-address-range=0
        --min-address-range-loop=0
        --min-instructions=8
        -E
        custom_push_region
    RUNTIME_ARGS
        -e
        -v
        1
        -l
        --min-address-range=0
        --min-address-range-loop=0
        --min-instructions=8
        -E
        custom_push_region
        --label
        file
        line
        return
        args
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF")

omnitrace_add_test(
    NAME lulesh
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    NUM_PROCS 8
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
        "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=OFF")

omnitrace_add_test(
    NAME lulesh-kokkosp
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    NUM_PROCS 8
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
    NAME lulesh-perfetto
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    NUM_PROCS 8
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
    NAME lulesh-timemory
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    NUM_PROCS 8
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
        "${_timemory_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=OFF")

omnitrace_add_test(
    NAME openmp-cg
    TARGET openmp-cg
    REWRITE_ARGS -e -v 2 --instrument-loops
    RUNTIME_ARGS -e -v 1 --label return args
    REWRITE_TIMEOUT 180
    RUNTIME_TIMEOUT 360
    ENVIRONMENT "${_ompt_environment}")

omnitrace_add_test(
    NAME openmp-lu
    TARGET openmp-lu
    REWRITE_ARGS -e -v 2 --instrument-loops
    RUNTIME_ARGS -e -v 1 --label return args
    REWRITE_TIMEOUT 180
    RUNTIME_TIMEOUT 360
    ENVIRONMENT "${_ompt_environment}")

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
        PROFILE_ARGS -f
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
        PROFILE_ARGS -b -l -f
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
            PASS_REGULAR_EXPRESSION
                "(\\\[compile\\\]).*(\\\| \\\|0>>> \\\[run\\\]\\\[external.py\\\]).*(\\\| \\\|0>>> \\\|_\\\[fib\\\]\\\[external.py\\\]).*(\\\| \\\|0>>> \\\|_\\\[inefficient\\\]\\\[external.py\\\])"
            DEPENDS python-external-${_VERSION}
            ENVIRONMENT "${_python_environment}")

        omnitrace_add_python_test(
            NAME python-external-exclude-inefficient-check
            COMMAND ${OMNITRACE_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/python-external-exclude-inefficient/${_VERSION}/trip_count.txt
            FAIL_REGULAR_EXPRESSION "(\\\|_inefficient).*(\\\|_sum)"
            DEPENDS python-external-exclude-inefficient-${_VERSION}
            ENVIRONMENT "${_python_environment}")

        omnitrace_add_python_test(
            NAME python-builtin-check
            COMMAND ${OMNITRACE_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/python-builtin/${_VERSION}/trip_count.txt
            PASS_REGULAR_EXPRESSION "\\\[inefficient\\\]\\\[builtin.py:14\\\]"
            DEPENDS python-builtin-${_VERSION}
            ENVIRONMENT "${_python_environment}")

        omnitrace_add_python_test(
            NAME python-source-check
            COMMAND ${OMNITRACE_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/python-source/${_VERSION}/trip_count.txt
            PASS_REGULAR_EXPRESSION
                "(\\\| \\\|0>>> run           \\\|      5).*(\\\| \\\|0>>> \\\|_fib         \\\|     40).*(\\\| \\\|0>>>         \\\|_fib \\\|      5).*(\\\| \\\|0>>> \\\|_inefficient \\\|      5).*(\\\| \\\|0>>>   \\\|__sum      \\\|      5)"
            DEPENDS python-source-${_VERSION}
            ENVIRONMENT "${_python_environment}")
    else()
        omnitrace_message(
            WARNING
            "Neither 'cat' nor 'cmake -E cat' are available. Python source checks are disabled"
            )
    endif()
    math(EXPR _INDEX "${_INDEX} + 1")
endforeach()
