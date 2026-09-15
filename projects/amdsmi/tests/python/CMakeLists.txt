# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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

# test_abi_compat.py exercises the wheel loader's system-fallback guard by
# invoking tools/disable_system_fallback.py. In the installed layout the test
# resolves that tool relative to SHARE_INSTALL_PREFIX (its REPO_ROOT), so it
# must be shipped alongside the tests -- otherwise the fallback-disabled tests
# silently skip and the wheel's core safety property goes unverified in CI.
install(
    PROGRAMS ${PROJECT_SOURCE_DIR}/tools/disable_system_fallback.py
    DESTINATION ${SHARE_INSTALL_PREFIX}/tools/
    COMPONENT ${TESTS_COMPONENT}
)

# test_dual_copy_guard.py loads run_amdsmi_dual_copy_test.py relative to its
# REPO_ROOT (SHARE_INSTALL_PREFIX in the installed layout), so the runner must
# ship under tests/ -- otherwise the guard's four unit tests raise SkipTest and
# the drift-guard logic is never exercised in the installed test run.
install(
    PROGRAMS ${PROJECT_SOURCE_DIR}/tests/run_amdsmi_dual_copy_test.py
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests/
    COMPONENT ${TESTS_COMPONENT}
)
