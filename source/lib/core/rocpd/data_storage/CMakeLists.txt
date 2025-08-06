set(data_storage_sources ${CMAKE_CURRENT_LIST_DIR}/database.cpp)

set(data_storage_headers
    ${CMAKE_CURRENT_LIST_DIR}/database.hpp
    ${CMAKE_CURRENT_LIST_DIR}/insert_query_builders.hpp
    ${CMAKE_CURRENT_LIST_DIR}/table_insert_query.hpp
)

target_sources(rocprofiler-systems-core-library PRIVATE ${data_storage_sources})

target_link_libraries(
    rocprofiler-systems-core-library
    PRIVATE $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-sqlite3>
)
