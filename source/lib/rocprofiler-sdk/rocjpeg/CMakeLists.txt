set(ROCPROFILER_LIB_ROCJPEG_SOURCES abi.cpp rocjpeg.cpp)
set(ROCPROFILER_LIB_ROCJPEG_HEADERS defines.hpp rocjpeg.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_ROCJPEG_SOURCES}
                                                      ${ROCPROFILER_LIB_ROCJPEG_HEADERS})
