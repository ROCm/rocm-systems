set(ROCPROFILER_LIB_HSA_SOURCES hsa.cpp agent.cpp)
set(ROCPROFILER_LIB_HSA_HEADERS hsa.hpp defines.hpp types.hpp utils.hpp agent.hpp)
target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_HSA_SOURCES}
                                                  ${ROCPROFILER_LIB_HSA_HEADERS})

add_subdirectory(details)
