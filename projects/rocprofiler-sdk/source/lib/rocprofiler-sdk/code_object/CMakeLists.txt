#
# code object
#
set(ROCPROFILER_LIB_CODE_OBJECT_SOURCES code_object.cpp)
set(ROCPROFILER_LIB_CODE_OBJECT_HEADERS code_object.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_CODE_OBJECT_SOURCES}
                                           ${ROCPROFILER_LIB_CODE_OBJECT_HEADERS})

add_subdirectory(hsa)
add_subdirectory(hip)
