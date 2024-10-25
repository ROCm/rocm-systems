set(ROCPROFILER_LIB_THREAD_TRACE_SOURCES att_core.cpp att_service.cpp att_parser.cpp
                                         code_object.cpp)
set(ROCPROFILER_LIB_THREAD_TRACE_HEADERS att_core.hpp code_object.hpp)
target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_THREAD_TRACE_SOURCES}
                                           ${ROCPROFILER_LIB_THREAD_TRACE_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
