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
    queue_signal_subscription.cpp
    scratch_memory.cpp
    signal_monitor.cpp
    signal_monitor_ioctl.cpp
    signal_monitor_polling.cpp)

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
    queue_signal_subscription.hpp
    rocprofiler_packet.hpp
    scratch_memory.hpp
    signal.hpp
    signal_monitor.hpp
    utils.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_HSA_SOURCES}
                                                      ${ROCPROFILER_LIB_HSA_HEADERS})

add_subdirectory(details)
