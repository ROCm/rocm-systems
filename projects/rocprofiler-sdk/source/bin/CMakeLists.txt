#
#
#

rocprofiler_activate_clang_tidy()

add_subdirectory(rocprofv3_avail_module)

# Adding main rocprofv3
configure_file(rocprofv3.py ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3 @ONLY)

install(
    FILES ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ
                WORLD_EXECUTE
    COMPONENT tools)

configure_file(rocprofv3-avail.py
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3-avail COPYONLY)

install(
    FILES ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3-avail
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ
                WORLD_EXECUTE
    COMPONENT tools)
