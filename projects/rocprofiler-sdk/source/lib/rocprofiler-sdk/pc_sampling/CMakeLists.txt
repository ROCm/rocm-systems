#
# PC Sampling
# - HSA support officially added in HSA-Runtime v1.14.0
#
add_subdirectory(parser)

# queue_hooks provides always-linkable shims (is_configured_on_agent /
# signal_completion_hook) that the HSA queue interceptor calls unconditionally, so it must
# compile even when PC sampling HSA support is unavailable (its body is guarded by
# ROCPROFILER_SDK_HSA_PC_SAMPLING). Add it before the early return.
target_sources(rocprofiler-sdk-object-library PRIVATE queue_hooks.cpp queue_hooks.hpp)

if(hsa-runtime64_VERSION AND hsa-runtime64_VERSION VERSION_LESS 1.14.0)
    return()
endif()

set(ROCPROFILER_PC_SAMPLING_SOURCES hsa_adapter.cpp utils.cpp service.cpp cid_manager.cpp
                                    code_object.cpp)
set(ROCPROFILER_PC_SAMPLING_HEADERS hsa_adapter.hpp utils.hpp service.hpp types.hpp
                                    cid_manager.hpp code_object.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_PC_SAMPLING_SOURCES}
                                                      ${ROCPROFILER_PC_SAMPLING_HEADERS})

add_subdirectory(ioctl)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
