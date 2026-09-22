#
#
#

# install the downloaded timem libraries
install(
    DIRECTORY ${PROJECT_BINARY_DIR}/lib64/
    DESTINATION ${CMAKE_INSTALL_LIB64DIR}
    FILES_MATCHING
    PATTERN "*libtimem.*"
    PATTERN "*/timemory/*")
