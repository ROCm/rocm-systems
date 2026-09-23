if(WIN32)
    # ROCm SHMEM has no Windows build, so its API table is never captured.
    set(ROCPROFILER_LIB_ROCSHMEM_SOURCES windows_stubs.cpp)
else()
    set(ROCPROFILER_LIB_ROCSHMEM_SOURCES abi.cpp rocshmem.cpp)
endif()
set(ROCPROFILER_LIB_ROCSHMEM_HEADERS defines.hpp rocshmem.hpp utils.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_ROCSHMEM_SOURCES}
                                                      ${ROCPROFILER_LIB_ROCSHMEM_HEADERS})
