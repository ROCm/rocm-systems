set(ROCPROFILER_LIB_ROCDECODE_SOURCES abi.cpp rocdecode.cpp)
set(ROCPROFILER_LIB_ROCDECODE_HEADERS defines.hpp rocdecode.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_ROCDECODE_SOURCES}
                                           ${ROCPROFILER_LIB_ROCDECODE_HEADERS})
