#
#
#   Installation of public headers
#
#
configure_file(${CMAKE_CURRENT_LIST_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

configure_file(${CMAKE_CURRENT_LIST_DIR}/ext_version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/ext_version.h @ONLY)

set(ROCPROFILER_HEADER_FILES
    # core headers
    rocprofiler.h
    # secondary headers
    agent.h
    buffer.h
    buffer_tracing.h
    callback_tracing.h
    context.h
    counter_config.h
    counters.h
    defines.h
    device_counting_service.h
    dispatch_counting_service.h
    external_correlation.h
    fwd.h
    hip.h
    hsa.h
    intercept_table.h
    internal_threading.h
    marker.h
    ompt.h
    pc_sampling.h
    rccl.h
    registration.h
    rocdecode.h
    rocjpeg.h
    spm.h
    ${CMAKE_CURRENT_BINARY_DIR}/version.h
    ${CMAKE_CURRENT_BINARY_DIR}/ext_version.h)

install(
    FILES ${ROCPROFILER_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk
    COMPONENT development)

add_subdirectory(hip)
add_subdirectory(hsa)
add_subdirectory(marker)
add_subdirectory(ompt)
add_subdirectory(rccl)
add_subdirectory(rocdecode)
add_subdirectory(rocjpeg)
add_subdirectory(cxx)
add_subdirectory(kfd)
add_subdirectory(experimental)
add_subdirectory(deprecated)
