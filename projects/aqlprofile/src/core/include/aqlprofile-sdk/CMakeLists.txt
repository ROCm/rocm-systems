set(AQLPROFILE_HEADER_FILES
    aql_profile_v2.h
    ${CMAKE_CURRENT_BINARY_DIR}/version.h)

add_library(aqlprofile-headers INTERFACE)
add_library(aqlprofile::headers ALIAS aqlprofile-headers)
target_include_directories(aqlprofile-headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(aqlprofile-headers
    INTERFACE ${AQLPROFILE_HEADER_FILES})

configure_file(${CMAKE_CURRENT_LIST_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

install(FILES ${AQLPROFILE_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aqlprofile-sdk
    COMPONENT development
)
