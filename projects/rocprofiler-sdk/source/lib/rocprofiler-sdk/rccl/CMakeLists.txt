if(WIN32)
    # RCCL has no Windows build, so its API table is never captured.
    set(ROCPROFILER_LIB_RCCL_SOURCES windows_stubs.cpp)
else()
    set(ROCPROFILER_LIB_RCCL_SOURCES abi.cpp rccl.cpp)
endif()
set(ROCPROFILER_LIB_RCCL_HEADERS defines.hpp rccl.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_RCCL_SOURCES}
                                                      ${ROCPROFILER_LIB_RCCL_HEADERS})
