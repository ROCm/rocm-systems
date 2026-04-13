#
set(rocprofiler_sdk_sources
    ${CMAKE_CURRENT_LIST_DIR}/counters.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fwd.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kfd_events.cpp
    ${CMAKE_CURRENT_LIST_DIR}/roctx_client.cpp
    ${CMAKE_CURRENT_LIST_DIR}/marker_writer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/trace_control.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rccl.cpp
)

set(rocprofiler_sdk_headers
    ${CMAKE_CURRENT_LIST_DIR}/counters.hpp
    ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp
    ${CMAKE_CURRENT_LIST_DIR}/kfd_events.hpp
    ${CMAKE_CURRENT_LIST_DIR}/roctx_client.hpp
    ${CMAKE_CURRENT_LIST_DIR}/marker_writer.hpp
    ${CMAKE_CURRENT_LIST_DIR}/trace_control.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rccl.hpp
    ${CMAKE_CURRENT_LIST_DIR}/rccl_internal.hpp
)

target_sources(
    rocprofiler-systems-object-library
    PRIVATE ${rocprofiler_sdk_sources} ${rocprofiler_sdk_headers}
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
