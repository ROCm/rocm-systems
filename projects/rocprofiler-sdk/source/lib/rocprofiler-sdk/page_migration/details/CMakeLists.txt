#
#
set(ROCPROFILER_LIB_UVM_DETAILS_SOURCES)
set(ROCPROFILER_LIB_UVM_DETAILS_HEADERS kfd_ioctl.h)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_UVM_DETAILS_SOURCES}
                                                  ${ROCPROFILER_LIB_UVM_DETAILS_HEADERS})
