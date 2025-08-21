set(AQLPROFILE_HEADER_FILES
    aql_profile_v2.h
    ${CMAKE_CURRENT_BINARY_DIR}/version.h)

configure_file(${CMAKE_CURRENT_LIST_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

install(
    FILES ${AQLPROFILE_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aqlprofile-sdk)
