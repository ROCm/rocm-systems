#
# PC sampling tests
#

set(ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX
    "PC sampling unavailable|PC sampling configuration is not supported"
    CACHE STRING "regex for skipping PC sampling test when not supported by given agents")

# rocprofv3 refuses to configure PC sampling unless the feature is opted into, and the
# resulting error matches the skip regex above, so without this the tests silently skip.
set(ROCPROFV3_TESTS_PC_SAMPLING_ENVIRONMENT
    "ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON"
    CACHE STRING "environment opting PC sampling tests into the beta feature")

add_subdirectory(host-trap)
add_subdirectory(stochastic)
