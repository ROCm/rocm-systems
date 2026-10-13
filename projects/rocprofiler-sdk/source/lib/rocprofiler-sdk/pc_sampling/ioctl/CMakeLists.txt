set(ROCPROFILER_PC_SAMPLING_IOCTL_SOURCES ioctl_adapter.cpp)
set(ROCPROFILER_PC_SAMPLING_IOCTL_HEADERS ioctl_adapter.hpp ioctl_adapter_types.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_PC_SAMPLING_IOCTL_SOURCES}
                                           ${ROCPROFILER_PC_SAMPLING_IOCTL_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
