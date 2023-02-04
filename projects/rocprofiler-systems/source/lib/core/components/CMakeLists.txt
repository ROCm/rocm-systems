#
set(component_sources)

set(component_headers ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp)

target_sources(omnitrace-core-library PRIVATE ${component_sources} ${component_headers})
