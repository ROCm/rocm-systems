#
#   add container sources and headers to common library target
#
set(containers_headers ring_buffer.hpp c_array.hpp operators.hpp record_header_buffer.hpp
                       ring_buffer.hpp stable_vector.hpp static_vector.hpp)
set(containers_sources ring_buffer.cpp record_header_buffer.cpp ring_buffer.cpp)

target_sources(rocprofiler-common-library PRIVATE ${containers_sources}
                                                  ${containers_headers})
