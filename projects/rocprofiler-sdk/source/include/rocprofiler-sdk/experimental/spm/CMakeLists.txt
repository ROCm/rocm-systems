set(ROCPROFILER_EXPERIMENTAL_SPM_HEADER_FILES core.h capture.h decode.h)

install(
    FILES ${ROCPROFILER_EXPERIMENTAL_SPM_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/experimental/spm
    COMPONENT development)
