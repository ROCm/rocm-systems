#
#
if(WIN32)
    # The OpenMP tool interface reaches rocprofiler through the device runtime's callback
    # registration, which has no Windows implementation.
    set(ROCPROFILER_LIB_OMPT_SOURCES windows_stubs.cpp)
else()
    set(ROCPROFILER_LIB_OMPT_SOURCES ompt.cpp)
endif()
set(ROCPROFILER_LIB_OMPT_HEADERS defines.hpp ompt.hpp utils.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_OMPT_SOURCES})

add_subdirectory(details)
