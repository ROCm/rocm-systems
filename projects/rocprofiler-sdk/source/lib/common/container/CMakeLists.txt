#
set(containers_sources)

set(containers_headers atomic_ring_buffer.hpp c_array.hpp operators.hpp ring_buffer.hpp
                       stable_vector.hpp static_vector.hpp)

set(containers_sources atomic_ring_buffer.cpp ring_buffer.cpp)

target_sources(rocprofiler-common-library PRIVATE ${containers_sources}
                                                  ${containers_headers})
