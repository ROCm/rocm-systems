#
set(rocprofiler_sdk_sources ${CMAKE_CURRENT_LIST_DIR}/counters.cpp
                            ${CMAKE_CURRENT_LIST_DIR}/fwd.cpp)

set(rocprofiler_sdk_headers ${CMAKE_CURRENT_LIST_DIR}/counters.hpp
                            ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp)

target_sources(rocprofiler-systems-object-library PRIVATE ${rocprofiler_sdk_sources}
                                                          ${rocprofiler_sdk_headers})
