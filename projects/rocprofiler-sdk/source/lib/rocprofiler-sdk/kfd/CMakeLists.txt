#
#
set(ROCPROFILER_LIB_KFD_EVENT_SOURCES abi.cpp kfd.cpp)
set(ROCPROFILER_LIB_KFD_EVENT_HEADERS defines.hpp kfd.hpp utils.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_KFD_EVENT_SOURCES}
                                           ${ROCPROFILER_LIB_KFD_EVENT_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
