#
#
#
set(ROCPROFILER_DETAILS_SOURCES)
set(ROCPROFILER_DETAILS_HEADERS kfd_ioctl.h)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_DETAILS_SOURCES}
                                                  ${ROCPROFILER_DETAILS_HEADERS})
