message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("            CMake AMD SMI Python Tests (restructured)              ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Share Install Prefix: " ${SHARE_INSTALL_PREFIX})

# ---------------------------------------------------------------------------
# Note: Tests are installed by tests/python_unittest/CMakeLists.txt to
# ${SHARE_INSTALL_PREFIX}/tests/python_unittest/ for backward compatibility.
# No installation from this CMakeLists.txt to avoid duplication.
# ---------------------------------------------------------------------------

# Tests are run directly via the runner scripts installed at python_unittest/:
#   .../python_unittest/unit_tests.py -v
#   .../python_unittest/integration_test.py -v
#   .../python_unittest/cli_unit_test.py -v
# See tests/python/unit_tests.py (and integration_test.py, cli_unit_test.py) for usage.
