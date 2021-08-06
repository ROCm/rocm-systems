
if(TARGET transpose)
    add_test(
        NAME transpose-binary-rewrite
        COMMAND $<TARGET_FILE:hosttrace-exe> -o transpose.inst -- ${CMAKE_BINARY_DIR}/transpose/transpose
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )

    add_test(
        NAME transpose-binary-rewrite-run
        COMMAND ./transpose.inst
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )

    add_test(
        NAME transpose-runtime-instrumentation
        COMMAND $<TARGET_FILE:hosttrace-exe> -- ${CMAKE_BINARY_DIR}/transpose/transpose
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )

    set_tests_properties(
        transpose-binary-rewrite-run
        PROPERTIES
            DEPENDS transpose-binary-rewrite
    )

    get_filename_component(DYNINST_LIB_DIR "${DYNINST_API_RT}" DIRECTORY CACHE)
    set_tests_properties(
        transpose-binary-rewrite
        transpose-binary-rewrite-run
        transpose-runtime-instrumentation
        PROPERTIES
            ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:${DYNINST_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
            TIMEOUT 600
    )
endif()
