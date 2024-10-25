#
# code object data structures from HIP
#
set(ROCPROFILER_LIB_CODE_OBJECT_HIP_SOURCES)
set(ROCPROFILER_LIB_CODE_OBJECT_HIP_HEADERS)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_CODE_OBJECT_HIP_SOURCES}
                                           ${ROCPROFILER_LIB_CODE_OBJECT_HIP_HEADERS})
