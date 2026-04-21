#
set(ROCPROFILER_LIB_HSA_SOURCES
    abi.cpp
    agent_cache.cpp
    aql_packet.cpp
    async_copy.cpp
    hsa_api_intercept.cpp
    hsa_barrier.cpp
    hsa.cpp
    memory_allocation.cpp
    packet_transformer.cpp
    pc_sampling.hpp
    profile_serializer.cpp
    queue_controller.cpp
    queue_intercept.cpp
    queue_state_registry.cpp
    queue.cpp
    ring_buffer.cpp
    scratch_memory.cpp)

set(ROCPROFILER_LIB_HSA_HEADERS
    agent_cache.hpp
    aql_packet.hpp
    async_copy.hpp
    memory_allocation.hpp
    defines.hpp
    hsa_api_intercept.hpp
    hsa_barrier.hpp
    hsa.hpp
    packet_transformer.hpp
    pc_sampling.cpp
    profile_serializer.hpp
    queue_controller.hpp
    queue_intercept.hpp
    queue_state_registry.hpp
    queue.hpp
    queue_info_session.hpp
    ring_buffer.hpp
    rocprofiler_packet.hpp
    scratch_memory.hpp
    signal.hpp
    utils.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_HSA_SOURCES}
                                                      ${ROCPROFILER_LIB_HSA_HEADERS})

add_subdirectory(details)
