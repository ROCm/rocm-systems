#
set(causal_sources
    ${CMAKE_CURRENT_LIST_DIR}/data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/delay.cpp
    ${CMAKE_CURRENT_LIST_DIR}/experiment.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sample_data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sampling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/selected_entry.cpp
)

set(causal_headers
    ${CMAKE_CURRENT_LIST_DIR}/data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/delay.hpp
    ${CMAKE_CURRENT_LIST_DIR}/experiment.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sample_data.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sampling.hpp
    ${CMAKE_CURRENT_LIST_DIR}/selected_entry.hpp
)

target_sources(
    rocprofiler-systems-object-library
    PRIVATE ${causal_sources} ${causal_headers}
)

add_subdirectory(components)
