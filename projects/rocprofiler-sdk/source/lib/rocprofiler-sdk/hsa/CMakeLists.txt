#
set(ROCPROFILER_LIB_HSA_SOURCES
    abi.cpp
    agent_cache.cpp
    aql_packet.cpp
    async_copy.cpp
    hsa_barrier.cpp
    hsa.cpp
    memory_allocation.cpp
    pc_sampling.hpp
    profile_serializer.cpp
    queue_controller.cpp
    queue.cpp
    scratch_memory.cpp)

set(ROCPROFILER_LIB_HSA_HEADERS
    agent_cache.hpp
    aql_packet.hpp
    async_copy.hpp
    memory_allocation.hpp
    defines.hpp
    hsa_barrier.hpp
    hsa.hpp
    pc_sampling.cpp
    profile_serializer.hpp
    queue_controller.hpp
    queue.hpp
    queue_info_session.hpp
    rocprofiler_packet.hpp
    scratch_memory.hpp
    signal.hpp
    utils.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_HSA_SOURCES}
                                                      ${ROCPROFILER_LIB_HSA_HEADERS})

# Windows-only: HSA symbol resolver against amdhip64_X.dll. Per user-locked
# decision R7, the SDK does NOT link against any standalone hsa-runtime on
# Windows -- it dlsyms the handful of HSA bootstrap functions it needs from
# amdhip64 at start-up.
if(WIN32)
    target_sources(rocprofiler-sdk-object-library PRIVATE win_hsa_loader.cpp)
endif()

add_subdirectory(details)
