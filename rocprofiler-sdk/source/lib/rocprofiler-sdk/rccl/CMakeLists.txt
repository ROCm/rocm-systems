set(ROCPROFILER_LIB_RCCL_SOURCES abi.cpp rccl.cpp)
set(ROCPROFILER_LIB_RCCL_HEADERS defines.hpp rccl.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_RCCL_SOURCES}
                                                      ${ROCPROFILER_LIB_RCCL_HEADERS})
