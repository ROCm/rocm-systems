# -------------------------------------------------------------------------------------- #
#
# general config file tests
#
# -------------------------------------------------------------------------------------- #

file(
    WRITE ${CMAKE_CURRENT_BINARY_DIR}/invalid.cfg
    "
OMNITRACE_CONFIG_FILE =
FOOBAR = ON
")

if(TARGET parallel-overhead)
    set(_CONFIG_TEST_EXE $<TARGET_FILE:parallel-overhead>)
else()
    set(_CONFIG_TEST_EXE ls)
endif()

add_test(
    NAME omnitrace-invalid-config
    COMMAND $<TARGET_FILE:omnitrace-instrument> -- ${_CONFIG_TEST_EXE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(
    omnitrace-invalid-config
    PROPERTIES ENVIRONMENT
               "OMNITRACE_CONFIG_FILE=${CMAKE_CURRENT_BINARY_DIR}/invalid.cfg" TIMEOUT
               120 LABELS "config" WILL_FAIL ON)

add_test(
    NAME omnitrace-missing-config
    COMMAND $<TARGET_FILE:omnitrace-instrument> -- ${_CONFIG_TEST_EXE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(
    omnitrace-missing-config
    PROPERTIES ENVIRONMENT
               "OMNITRACE_CONFIG_FILE=${CMAKE_CURRENT_BINARY_DIR}/missing.cfg" TIMEOUT
               120 LABELS "config" WILL_FAIL ON)
