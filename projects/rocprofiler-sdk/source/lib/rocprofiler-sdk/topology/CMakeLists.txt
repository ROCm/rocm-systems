#
#
#
set(ROCPROFILER_TOPOLOGY_HEADERS topology_provider.hpp linux_kfd_provider.hpp
                                 dxg_provider.hpp d3dkmt_provider.hpp)

set(ROCPROFILER_TOPOLOGY_SOURCES topology_provider.cpp linux_kfd_provider.cpp
                                 dxg_provider.cpp d3dkmt_provider.cpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_TOPOLOGY_SOURCES}
                                                      ${ROCPROFILER_TOPOLOGY_HEADERS})
