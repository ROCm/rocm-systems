#
# PC sampling tests
#

set(ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX
    "PC sampling unavailable|PC sampling configuration is not supported"
    CACHE STRING "regex for skipping PC sampling test when not supported by given agents")

add_subdirectory(host-trap)
add_subdirectory(stochastic)
