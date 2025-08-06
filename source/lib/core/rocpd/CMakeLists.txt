set(rocpd_sources
    ${CMAKE_CURRENT_LIST_DIR}/data_processor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/json.cpp
)

set(rocpd_headers
    ${CMAKE_CURRENT_LIST_DIR}/data_processor.hpp
    ${CMAKE_CURRENT_LIST_DIR}/json.hpp
)

target_sources(rocprofiler-systems-core-library PRIVATE ${rocpd_sources} ${rocpd_headers})

add_subdirectory(data_storage)
