set(ROCPROFILER_LIB_THREAD_TRACE_SOURCES core.cpp service.cpp code_object.cpp decode.cpp
                                         dl.cpp)
set(ROCPROFILER_LIB_THREAD_TRACE_HEADERS core.hpp code_object.hpp dl.hpp
                                         trace_decoder_api.h)
target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_THREAD_TRACE_SOURCES}
                                           ${ROCPROFILER_LIB_THREAD_TRACE_HEADERS})

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
