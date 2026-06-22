# CTestConfig.cmake — CDash connection settings.
# CTest automatically loads this file from the source directory when running
# dashboard operations (-T Submit, ctest_submit(), etc.).

set(CTEST_PROJECT_NAME "rocprofiler-systems")
set(CTEST_NIGHTLY_START_TIME "05:00:00 UTC")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE_CDASH TRUE)
set(CTEST_SUBMIT_URL "https://my.cdash.org/submit.php?project=rocprofiler-systems")
