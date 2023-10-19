set(ROCPROFILER_LIB_HSA_SOURCES hsa.cpp queue.cpp queue_controller.cpp agent_cache.cpp)
set(ROCPROFILER_LIB_HSA_HEADERS hsa.hpp defines.hpp types.hpp utils.hpp queue.hpp
                                queue_controller.hpp agent_cache.hpp aql_packet.hpp)
target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_HSA_SOURCES}
                                                  ${ROCPROFILER_LIB_HSA_HEADERS})

add_subdirectory(details)
