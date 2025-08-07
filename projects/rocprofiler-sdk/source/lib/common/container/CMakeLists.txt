#
#   add container sources and headers to common library target
#
set(containers_headers
    ring_buffer.hpp c_array.hpp operators.hpp record_header_buffer.hpp ring_buffer.hpp
    small_vector.hpp stable_vector.hpp static_vector.hpp)
set(containers_sources ring_buffer.cpp record_header_buffer.cpp ring_buffer.cpp
                       small_vector.cpp)

target_sources(rocprofiler-sdk-common-library PRIVATE ${containers_sources}
                                                      ${containers_headers})
target_link_libraries(rocprofiler-sdk-common-library PRIVATE rocprofiler-sdk-hip-nolink)
