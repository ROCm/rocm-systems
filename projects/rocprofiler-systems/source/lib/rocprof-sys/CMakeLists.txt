# ------------------------------------------------------------------------------#
#
# rocprofiler-systems object library
#
# ------------------------------------------------------------------------------#

add_library(rocprofiler-systems-object-library OBJECT)
add_library(
    rocprofiler-systems::rocprofiler-systems-object-library
    ALIAS rocprofiler-systems-object-library
)

target_sources(
    rocprofiler-systems-object-library
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/library.cpp
        ${CMAKE_CURRENT_LIST_DIR}/regions.cpp
        ${CMAKE_CURRENT_LIST_DIR}/progress.cpp
        ${CMAKE_CURRENT_LIST_DIR}/api.cpp
        ${CMAKE_CURRENT_LIST_DIR}/timeout.cpp
        ${CMAKE_CURRENT_LIST_DIR}/api.hpp
)

add_subdirectory(library)

target_link_libraries(
    rocprofiler-systems-object-library
    PRIVATE rocprofiler-systems::rocprofiler-systems-interface-library
)

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems static library
#
# ------------------------------------------------------------------------------#

add_library(
    rocprofiler-systems-static-library
    STATIC
    $<TARGET_OBJECTS:rocprofiler-systems-object-library>
)
add_library(
    rocprofiler-systems::librocprofiler-systems-static
    ALIAS rocprofiler-systems-static-library
)

target_link_libraries(
    rocprofiler-systems-static-library
    PRIVATE
        rocprofiler-systems::rocprofiler-systems-interface-library
        rocprofiler-systems::rocprofiler-systems-core
        rocprofiler-systems::rocprofiler-systems-binary
)

set_target_properties(
    rocprofiler-systems-static-library
    PROPERTIES OUTPUT_NAME ${BINARY_NAME_PREFIX}
)

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems shared library
#
# ------------------------------------------------------------------------------#

add_library(
    rocprofiler-systems-shared-library
    SHARED
    $<TARGET_OBJECTS:rocprofiler-systems-object-library>
)
add_library(
    rocprofiler-systems::librocprofiler-systems-shared
    ALIAS rocprofiler-systems-shared-library
)
add_library(
    rocprofiler-systems::rocprofiler-systems-library
    ALIAS rocprofiler-systems-shared-library
)

target_link_libraries(
    rocprofiler-systems-shared-library
    PRIVATE
        rocprofiler-systems::rocprofiler-systems-interface-library
        rocprofiler-systems::rocprofiler-systems-core
        rocprofiler-systems::rocprofiler-systems-binary
)

set_target_properties(
    rocprofiler-systems-shared-library
    PROPERTIES
        OUTPUT_NAME ${BINARY_NAME_PREFIX}
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        INSTALL_RPATH "${ROCPROFSYS_LIB_INSTALL_RPATH}"
)

rocprofiler_systems_strip_target(rocprofiler-systems-shared-library)

install(
    TARGETS rocprofiler-systems-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL
)
