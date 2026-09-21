#
if(WIN32)
    # On Windows, only compile the HSA files that don't require Linux-specific headers
    # (dlfcn.h, unistd.h, GCC __atomic builtins, kfd ioctl, etc.). queue.cpp,
    # queue_interposition.cpp, queue_controller.cpp, hsa.cpp, abi.cpp, async_copy.cpp,
    # memory_allocation.cpp, scratch_memory.cpp all require Linux.
    set(ROCPROFILER_LIB_HSA_SOURCES
        agent_cache.cpp aql_packet.cpp hsa_barrier.cpp profile_serializer.cpp
        signal_pool.cpp windows_stubs.cpp)
    set(ROCPROFILER_LIB_HSA_HEADERS
        agent_cache.hpp
        aql_packet.hpp
        defines.hpp
        hsa_barrier.hpp
        hsa.hpp
        profile_serializer.hpp
        queue_controller.hpp
        queue_interposition.hpp
        queue.hpp
        queue_info_session.hpp
        rocprofiler_packet.hpp
        signal.hpp
        signal_pool.hpp
        utils.hpp)
else()
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
        queue_interposition.cpp
        queue.cpp
        scratch_memory.cpp
        signal_pool.cpp)
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
        queue_interposition.hpp
        queue.hpp
        queue_info_session.hpp
        rocprofiler_packet.hpp
        scratch_memory.hpp
        signal.hpp
        signal_pool.hpp
        utils.hpp)
endif()

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_HSA_SOURCES}
                                                      ${ROCPROFILER_LIB_HSA_HEADERS})

add_subdirectory(details)
