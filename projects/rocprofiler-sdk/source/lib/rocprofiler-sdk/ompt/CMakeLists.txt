#
#
set(ROCPROFILER_LIB_OMPT_SOURCES ompt.cpp)
set(ROCPROFILER_LIB_OMPT_HEADERS defines.hpp ompt.hpp utils.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_OMPT_SOURCES})

add_subdirectory(details)
