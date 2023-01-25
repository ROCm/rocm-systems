#
set(containers_sources)

set(containers_headers ${CMAKE_CURRENT_LIST_DIR}/stable_vector.hpp
                       ${CMAKE_CURRENT_LIST_DIR}/static_vector.hpp)

target_sources(omnitrace-object-library PRIVATE ${containers_sources}
                                                ${containers_headers})
