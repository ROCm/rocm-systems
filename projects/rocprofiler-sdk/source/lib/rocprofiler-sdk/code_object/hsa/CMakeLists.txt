#
# code object data structures from HSA
#
set(ROCPROFILER_LIB_CODE_OBJECT_HSA_SOURCES code_object.cpp kernel_symbol.cpp)
set(ROCPROFILER_LIB_CODE_OBJECT_HSA_HEADERS code_object.hpp kernel_symbol.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_CODE_OBJECT_HSA_SOURCES}
                                           ${ROCPROFILER_LIB_CODE_OBJECT_HSA_HEADERS})
