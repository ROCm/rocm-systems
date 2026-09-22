# CTestCustom.cmake — project-level CTest customization limits.
# CMake copies this to the binary directory during configure so ctest picks it
# up automatically before running any tests.  These values match what
# run-ci.py previously injected at runtime.

set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS "100")
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS "100")
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "51200")
set(CTEST_CUSTOM_COVERAGE_EXCLUDE "/usr/.*;.*external/.*;.*examples/.*")
