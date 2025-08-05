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

# for each entry here there must be a ROCPD_BIN_<entry> list
set(ROCPD_EXECUTABLES "all" "csv" "otf2" "pftrace" "summary")

# format is list: [<exe-name>, <python-module>, <extra-args-to-python-module>]
set(ROCPD_BIN_all "rocpd" "rocpd" "")
set(ROCPD_BIN_csv "rocpd2csv" "rocpd.csv" "")
set(ROCPD_BIN_otf2 "rocpd2otf2" "rocpd.otf2" "")
set(ROCPD_BIN_pftrace "rocpd2pftrace" "rocpd.pftrace" "")
set(ROCPD_BIN_summary "rocpd2summary" "rocpd.summary" "")

foreach(_EXE IN LISTS ROCPD_EXECUTABLES)
    list(GET ROCPD_BIN_${_EXE} 0 ROCPD_EXE_NAME)
    list(GET ROCPD_BIN_${_EXE} 1 ROCPD_EXE_MODULE)
    list(GET ROCPD_BIN_${_EXE} 2 ROCPD_EXE_MODULE_ARGS)

    if(NOT ROCPD_EXE_NAME
       OR NOT ROCPD_EXE_MODULE
       OR (NOT ROCPD_EXE_MODULE_ARGS AND NOT ROCPD_EXE_MODULE_ARGS STREQUAL ""))
        message(
            FATAL_ERROR "ROCPD_BIN_${_EXE} not properly defined: ${ROCPD_BIN_${_EXE}}")
    endif()

    set(ROCPD_EXE_OUTPUT_FILE
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/${ROCPD_EXE_NAME})

    # Adding main rocpd
    configure_file(rocpd.py ${ROCPD_EXE_OUTPUT_FILE} @ONLY)

    install(
        FILES ${ROCPD_EXE_OUTPUT_FILE}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE
        COMPONENT rocpd)
endforeach()
