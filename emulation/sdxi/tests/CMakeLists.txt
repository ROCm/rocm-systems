find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(SDXI_CLI_TEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cli/run_tests_via_ssh.sh")

add_test(
    NAME sdxi-cli-vm
    COMMAND bash ${SDXI_CLI_TEST_SCRIPT}
)
set_tests_properties(sdxi-cli-vm PROPERTIES
    TIMEOUT 120
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE
)

set(SDXI_PY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/py")
set(SDXI_SERVER_BIN "$<TARGET_FILE:rocm-sdxi>")

add_test(
    NAME sdxi-protocol
    COMMAND ${Python3_EXECUTABLE} ${SDXI_PY_DIR}/test_protocol.py
)
set_tests_properties(sdxi-protocol PROPERTIES
    TIMEOUT 120
    RUN_SERIAL TRUE
    ENVIRONMENT "PYTHONPATH=${SDXI_PY_DIR};SDXI_SERVER_BIN=${SDXI_SERVER_BIN}"
)

add_test(
    NAME sdxi-dma
    COMMAND ${Python3_EXECUTABLE} ${SDXI_PY_DIR}/test_dma.py
)
set_tests_properties(sdxi-dma PROPERTIES
    TIMEOUT 120
    RUN_SERIAL TRUE
    ENVIRONMENT "PYTHONPATH=${SDXI_PY_DIR};SDXI_SERVER_BIN=${SDXI_SERVER_BIN}"
)

add_test(
    NAME sdxi-cxt
    COMMAND ${Python3_EXECUTABLE} ${SDXI_PY_DIR}/test_cxt.py
)
set_tests_properties(sdxi-cxt PROPERTIES
    TIMEOUT 120
    RUN_SERIAL TRUE
    ENVIRONMENT "PYTHONPATH=${SDXI_PY_DIR};SDXI_SERVER_BIN=${SDXI_SERVER_BIN}"
)
