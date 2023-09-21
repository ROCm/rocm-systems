#
#
#   Installation of public headers
#
#
configure_file(${CMAKE_CURRENT_LIST_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

set(ROCPROFILER_HEADER_FILES
    # core headers
    rocprofiler.h
    rocprofiler_plugin.h
    # secondary headers
    agent.h
    agent_profile.h
    buffer.h
    buffer_tracing.h
    callback_tracing.h
    context.h
    counters.h
    defines.h
    dispatch_profile.h
    external_correlation.h
    fwd.h
    hip.h
    hsa.h
    internal_threading.h
    marker.h
    pc_sampling.h
    profile_config.h
    spm.h
    ${CMAKE_CURRENT_BINARY_DIR}/version.h)

install(FILES ${ROCPROFILER_HEADER_FILES}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler)

add_subdirectory(hip)
add_subdirectory(hsa)
add_subdirectory(marker)
