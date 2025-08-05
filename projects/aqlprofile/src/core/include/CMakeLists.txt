set(AQLPROFILE_HEADER_FILES
  aqlprofile-sdk/aql_profile_v2.h
)

install(
    FILES ${AQLPROFILE_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aqlprofile-sdk)