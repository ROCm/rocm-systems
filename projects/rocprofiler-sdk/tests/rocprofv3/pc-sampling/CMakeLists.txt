#
# PC sampling tests
#

set(ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX
    "PC sampling unavailable|PC sampling configuration is not supported"
    CACHE STRING "regex for skipping PC sampling test when not supported by given agents")

# rocprofv3 refuses to configure PC sampling unless the beta feature is opted into, and
# the resulting error matches the skip regex above, so without the opt-in the tests
# silently skip. The opt-in grants the whole process access to the feature rather than the
# one test that asked for it, so it is only made on architectures that support PC
# sampling; elsewhere the tests keep skipping as they did before.
rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED ECHO)

set(PC_SAMPLING_BETA_ENV "ROCPROFILER_PC_SAMPLING_BETA_ENABLED=ON")
if(IS_PC_SAMPLING_DISABLED)
    set(PC_SAMPLING_BETA_ENV "")
endif()

set(ROCPROFV3_TESTS_PC_SAMPLING_ENVIRONMENT
    "${PC_SAMPLING_BETA_ENV}"
    CACHE STRING "environment opting PC sampling tests into the beta feature")

add_subdirectory(host-trap)
add_subdirectory(stochastic)
