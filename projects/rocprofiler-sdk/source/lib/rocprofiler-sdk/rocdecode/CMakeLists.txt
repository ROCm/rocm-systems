if(WIN32)
    # rocDecode has no Windows build, so its API table is never captured.
    set(ROCPROFILER_LIB_ROCDECODE_SOURCES windows_stubs.cpp)
else()
    set(ROCPROFILER_LIB_ROCDECODE_SOURCES abi.cpp rocdecode.cpp)
endif()
set(ROCPROFILER_LIB_ROCDECODE_HEADERS defines.hpp rocdecode.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_ROCDECODE_SOURCES}
                                           ${ROCPROFILER_LIB_ROCDECODE_HEADERS})
