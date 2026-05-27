set(ROCPROFILER_LIB_ROCSHMEM_SOURCES abi.cpp rocshmem.cpp)
set(ROCPROFILER_LIB_ROCSHMEM_HEADERS defines.hpp rocshmem.hpp utils.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_ROCSHMEM_SOURCES}
                                                      ${ROCPROFILER_LIB_ROCSHMEM_HEADERS})
