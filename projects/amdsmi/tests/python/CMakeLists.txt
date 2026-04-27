message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("            CMake AMD SMI Python Tests (restructured)              ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Share Install Prefix: " ${SHARE_INSTALL_PREFIX})

# ---------------------------------------------------------------------------
# Install the entire tests/python/ tree into the shared install prefix so that
# the tests can be run from the installed location without the source tree.
# ---------------------------------------------------------------------------
install(
    DIRECTORY ./
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests/python/
    COMPONENT ${TESTS_COMPONENT}
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
    PATTERN "*.py"
)

# Tests are run directly via the runner scripts installed alongside the test tree:
#   unit_tests.py -v
#   integration_test.py -v
#   cli_unit_test.py -v
# See tests/python/unit_tests.py (and integration_test.py, cli_unit_test.py) for usage.
