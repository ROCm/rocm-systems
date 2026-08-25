#
#
#
set(ROCPROFILER_LIB_HSA_QUEUE_HOOKS_SOURCES activation.cpp)
set(ROCPROFILER_LIB_HSA_QUEUE_HOOKS_HEADERS activation.hpp client_ids.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_HSA_QUEUE_HOOKS_SOURCES}
                                           ${ROCPROFILER_LIB_HSA_QUEUE_HOOKS_HEADERS})
