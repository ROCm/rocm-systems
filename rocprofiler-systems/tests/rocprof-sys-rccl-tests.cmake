# -------------------------------------------------------------------------------------- #
#
# rccl tests
#
# -------------------------------------------------------------------------------------- #

foreach(_TARGET ${RCCL_TEST_TARGETS})
    string(REPLACE "rccl-tests::" "" _NAME "${_TARGET}")
    string(REPLACE "_" "-" _NAME "${_NAME}")
    rocprofiler_systems_add_test(
        NAME rccl-test-${_NAME}
        TARGET ${_TARGET}
        LABELS "rccl-tests;rcclp"
        MPI ON
        GPU ON
        NUM_PROCS 1
        REWRITE_ARGS
            -e
            -v
            2
            -i
            8
            --label
            file
            line
            return
            args
        RUNTIME_ARGS
            -e
            -v
            1
            -i
            8
            --label
            file
            line
            return
            args
            -ME
            sysdeps
            --log-file
            rccl-test-${_NAME}.log
        RUN_ARGS -t
                 1
                 -g
                 1
                 -i
                 10
                 -w
                 2
                 -m
                 2
                 -p
                 -c
                 1
                 -z
                 -s
                 1
        ENVIRONMENT "${_rccl_environment}")
endforeach()
