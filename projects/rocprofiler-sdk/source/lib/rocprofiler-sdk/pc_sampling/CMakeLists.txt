#
# PC Sampling
# - HSA support officially added in HSA-Runtime v1.14.0
#
add_subdirectory(parser)

# queue_hooks.{cpp,hpp} is always-available: it provides the is_configured_on_agent
# wrapper and maybe_marker_packet/signal_completion_hook shims that the HSA queue
# write/completion paths call unconditionally. The translation unit guards its body
# with ROCPROFILER_SDK_HSA_PC_SAMPLING; in disabled builds the symbols return
# nullopt / false / no-op. It MUST be added before the HSA version early-return so
# consumers can link to it on any HSA runtime.
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
