#
# code object data structures from HIP
#
set(ROCPROFILER_LIB_CODE_OBJECT_HIP_SOURCES code_object.cpp)
set(ROCPROFILER_LIB_CODE_OBJECT_HIP_HEADERS code_object.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_CODE_OBJECT_HIP_SOURCES}
                                           ${ROCPROFILER_LIB_CODE_OBJECT_HIP_HEADERS})
