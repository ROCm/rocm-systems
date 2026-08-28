message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  CMake AMD SMI Pytest                             ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Share Install Prefix: " ${SHARE_INSTALL_PREFIX})

# Install the tests/python/ tree into the shared install prefix under the legacy
# python_unittest/ path so existing invocations
# (e.g. .../tests/python_unittest/unit_tests.py -v) keep working unchanged.
install(
    DIRECTORY ./
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests/python_unittest/
    COMPONENT ${TESTS_COMPONENT}
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
    PATTERN "*.py"
)
