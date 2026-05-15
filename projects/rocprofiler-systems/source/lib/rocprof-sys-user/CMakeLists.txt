# ------------------------------------------------------------------------------#
#
# rocprofiler-systems user library
#
# ------------------------------------------------------------------------------#

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_SKIP_RPATH OFF)
set(BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(rocprofiler-systems-user-library SHARED)
add_library(
    rocprofiler-systems::rocprofiler-systems-user-library
    ALIAS rocprofiler-systems-user-library
)

set(_user_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/user.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/causal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/categories.h
)

set(_user_sources ${CMAKE_CURRENT_SOURCE_DIR}/user.cpp)

target_sources(rocprofiler-systems-user-library PRIVATE ${_user_sources} ${_user_headers})
target_include_directories(
    rocprofiler-systems-user-library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
    rocprofiler-systems-user-library
    PRIVATE
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-compile-definitions>
)

set_target_properties(
    rocprofiler-systems-user-library
    PROPERTIES
        OUTPUT_NAME ${BINARY_NAME_PREFIX}-user
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        POSITION_INDEPENDENT_CODE ON
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
)

rocprofiler_systems_strip_target(rocprofiler-systems-user-library)

foreach(_HEADER ${_user_headers})
    string(
        REPLACE
        "${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/"
        "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/"
        _DEST
        ${_HEADER}
    )
    string(
        REPLACE
        "${CMAKE_CURRENT_SOURCE_DIR}/"
        "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/"
        _DEST
        "${_DEST}"
    )

    configure_file(${_HEADER} ${_DEST} COPYONLY)
endforeach()

install(FILES ${_user_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(
    TARGETS rocprofiler-systems-user-library
    EXPORT rocprofiler-systems-library-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
