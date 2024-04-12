#
# context
#
set(ROCPROFILER_LIB_CONFIG_SOURCES context.cpp correlation_id.cpp domain.cpp)
set(ROCPROFILER_LIB_CONFIG_HEADERS context.hpp correlation_id.hpp domain.hpp
                                   allocator.hpp)

target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_CONFIG_SOURCES}
                                                  ${ROCPROFILER_LIB_CONFIG_HEADERS})
