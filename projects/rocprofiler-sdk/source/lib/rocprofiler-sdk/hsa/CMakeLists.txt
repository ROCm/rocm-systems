#
set(ROCPROFILER_LIB_HSA_SOURCES
    agent_cache.cpp
    aql_packet.cpp
    async_copy.cpp
    code_object.cpp
    hsa.cpp
    queue_controller.cpp
    queue.cpp
    hsa_barrier.cpp
    profile_serializer.cpp
    scratch_memory.cpp)

set(ROCPROFILER_LIB_HSA_HEADERS
    agent_cache.hpp
    aql_packet.hpp
    async_copy.hpp
    code_object.hpp
    defines.hpp
    hsa_barrier.hpp
    hsa.hpp
    profile_serializer.hpp
    queue_controller.hpp
    queue.hpp
    scratch_memory.hpp
    types.hpp
    utils.hpp)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_HSA_SOURCES}
                                                  ${ROCPROFILER_LIB_HSA_HEADERS})

add_subdirectory(details)
