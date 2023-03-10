# ------------------------------------------------------------------------------#
#
# omnitrace-exe target (deprecated 'omnitrace' executable, now 'omnitrace-instrument')
#
# ------------------------------------------------------------------------------#

add_executable(omnitrace-exe ${CMAKE_CURRENT_LIST_DIR}/omnitrace.cpp)

target_link_libraries(omnitrace-exe PRIVATE omnitrace::omnitrace-threading)

set_target_properties(
    omnitrace-exe
    PROPERTIES OUTPUT_NAME omnitrace
               BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
               INSTALL_RPATH "${OMNITRACE_EXE_INSTALL_RPATH}"
               INSTALL_RPATH_USE_LINK_PATH ON)

omnitrace_strip_target(omnitrace-exe)

install(
    TARGETS omnitrace-exe
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL)
