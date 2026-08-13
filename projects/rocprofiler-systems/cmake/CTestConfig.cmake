# CTestConfig.cmake — CDash connection settings committed to source tree.
# CTest picks this up automatically when running from the build directory.
# run-ci.py generates compatible settings at runtime in its CTestCustom.cmake;
# this file is the source-of-truth reference and enables direct ctest -D
# invocations without run-ci.py.

set(CTEST_PROJECT_NAME "rocprofiler-systems")
set(CTEST_NIGHTLY_START_TIME "05:00:00 UTC")

set(CTEST_DROP_METHOD "http")
set(CTEST_DROP_SITE_CDASH TRUE)
set(CTEST_SUBMIT_URL "https://my.cdash.org/submit.php?project=rocprofiler-systems")

set(CTEST_UPDATE_TYPE git)
set(CTEST_UPDATE_VERSION_ONLY TRUE)
set(CTEST_GIT_INIT_SUBMODULES TRUE)
