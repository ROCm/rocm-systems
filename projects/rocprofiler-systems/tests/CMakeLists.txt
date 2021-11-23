if(NOT HOSTTRACE_DYNINST_API_RT_DIR AND HOSTTRACE_DYNINST_API_RT)
    get_filename_component(HOSTTRACE_DYNINST_API_RT_DIR "${HOSTTRACE_DYNINST_API_RT}"
                           DIRECTORY)
endif()

if(NOT DEFINED NUM_PROCS)
    set(NUM_PROCS 2)
endif()

if(HOSTTRACE_BUILD_DYNINST)
    set(HOSTTRACE_DYNINST_API_RT_DIR
        "${PROJECT_BINARY_DIR}/external/dyninst/dyninstAPI_RT:${PROJECT_BINARY_DIR}/external/dyninst/dyninstAPI"
        )
endif()

set(_test_environment
    "HOSTTRACE_USE_PERFETTO=ON"
    "HOSTTRACE_USE_TIMEMORY=ON"
    "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${HOSTTRACE_DYNINST_API_RT_DIR}:$ENV{LD_LIBRARY_PATH}"
    )

if(TARGET transpose)
    if(TRANSPOSE_USE_MPI AND NUM_PROCS GREATER 0)
        set(COMMAND_PREFIX ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NUM_PROCS})
    endif()

    add_test(
        NAME transpose-binary-rewrite
        COMMAND
            $<TARGET_FILE:hosttrace-exe> -o $<TARGET_FILE_DIR:transpose>/transpose.inst -v
            1 -- $<TARGET_FILE:transpose>
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    add_test(
        NAME transpose-binary-rewrite-run
        COMMAND ${COMMAND_PREFIX} $<TARGET_FILE_DIR:transpose>/transpose.inst
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    add_test(
        NAME transpose-runtime-instrument
        COMMAND $<TARGET_FILE:hosttrace-exe> -v 1 --label file line return args --
                $<TARGET_FILE:transpose>
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    set_tests_properties(transpose-binary-rewrite-run PROPERTIES DEPENDS
                                                                 transpose-binary-rewrite)

    set_tests_properties(
        transpose-binary-rewrite transpose-binary-rewrite-run transpose-runtime-instrument
        PROPERTIES ENVIRONMENT "${_test_environment}" TIMEOUT 600)
endif()

if(TARGET parallel-overhead)
    add_test(
        NAME parallel-overhead-binary-rewrite
        COMMAND
            $<TARGET_FILE:hosttrace-exe> -o
            $<TARGET_FILE_DIR:parallel-overhead>/parallel-overhead.inst -v 1
            --min-address-range-loop=72 --label file line return args --
            $<TARGET_FILE:parallel-overhead>
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    add_test(
        NAME parallel-overhead-binary-rewrite-run
        COMMAND $<TARGET_FILE_DIR:parallel-overhead>/parallel-overhead.inst
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    add_test(
        NAME parallel-overhead-runtime-instrument
        COMMAND $<TARGET_FILE:hosttrace-exe> -v 1 -- $<TARGET_FILE:parallel-overhead>
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    set_tests_properties(parallel-overhead-binary-rewrite-run
                         PROPERTIES DEPENDS parallel-overhead-binary-rewrite)

    set_tests_properties(
        parallel-overhead-binary-rewrite parallel-overhead-binary-rewrite-run
        parallel-overhead-runtime-instrument
        PROPERTIES ENVIRONMENT "${_test_environment}" TIMEOUT 600)
endif()
