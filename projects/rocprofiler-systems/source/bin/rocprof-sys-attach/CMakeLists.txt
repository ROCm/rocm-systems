
set(TARGET rocprofiler-systems-attach)

add_executable(
    ${TARGET}
    ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-attach.cpp
)

target_link_libraries(${TARGET} rocprof-sys-attach-tool)

set_target_properties(
    ${TARGET}
    PROPERTIES
        BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
        INSTALL_RPATH "${ROCPROFSYS_EXE_INSTALL_RPATH}"
        OUTPUT_NAME ${BINARY_NAME_PREFIX}-attach
)

rocprofiler_systems_strip_target(${TARGET})

install(TARGETS ${TARGET} DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
