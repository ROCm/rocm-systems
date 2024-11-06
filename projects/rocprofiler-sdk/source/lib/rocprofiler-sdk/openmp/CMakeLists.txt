#
#
set(ROCPROFILER_LIB_OPENMP_SOURCES openmp.cpp)
set(ROCPROFILER_LIB_OPENMP_HEADERS defines.hpp openmp.hpp utils.hpp)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_OPENMP_SOURCES})

add_subdirectory(details)
