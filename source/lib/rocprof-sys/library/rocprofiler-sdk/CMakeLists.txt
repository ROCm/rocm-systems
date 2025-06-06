#
set(rocprofiler_sdk_sources
    ${CMAKE_CURRENT_LIST_DIR}/counters.cpp ${CMAKE_CURRENT_LIST_DIR}/fwd.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rccl.cpp)

set(rocprofiler_sdk_headers
    ${CMAKE_CURRENT_LIST_DIR}/counters.hpp ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rccl.hpp)

target_sources(rocprofiler-systems-object-library PRIVATE ${rocprofiler_sdk_sources}
                                                          ${rocprofiler_sdk_headers})
