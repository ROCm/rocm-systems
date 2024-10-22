#
#
#

rocprofiler_activate_clang_tidy()

# Adding main rocprofv3
configure_file(rocprofv3.py ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3
               COPYONLY)

install(
    FILES ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ
                WORLD_EXECUTE
    COMPONENT tools)
