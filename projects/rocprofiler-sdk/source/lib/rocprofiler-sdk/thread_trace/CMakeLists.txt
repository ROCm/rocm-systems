set(ROCPROFILER_LIB_THREAD_TRACE_SOURCES att_core.cpp att_service.cpp)
set(ROCPROFILER_LIB_THREAD_TRACE_HEADERS att_core.hpp)
target_sources(rocprofiler-object-library PRIVATE ${ROCPROFILER_LIB_THREAD_TRACE_SOURCES}
                                                  ${ROCPROFILER_LIB_THREAD_TRACE_HEADERS})

# if(ROCPROFILER_BUILD_TESTS) add_subdirectory(tests) endif()
