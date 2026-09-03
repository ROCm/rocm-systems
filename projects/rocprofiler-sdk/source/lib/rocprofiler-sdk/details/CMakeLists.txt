#
#
#
set(ROCPROFILER_DETAILS_SOURCES)
set(ROCPROFILER_DETAILS_HEADERS kfd_ioctl.h amdgpu_drm_pcs.h)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_DETAILS_SOURCES}
                                                      ${ROCPROFILER_DETAILS_HEADERS})
