#
# Shared, header-only client-id tags used by the per-service queue hooks.
#
set(ROCPROFILER_LIB_HSA_QUEUE_HOOKS_SOURCES)
set(ROCPROFILER_LIB_HSA_QUEUE_HOOKS_HEADERS client_ids.hpp)

target_sources(
    rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_HSA_QUEUE_HOOKS_SOURCES}
                                           ${ROCPROFILER_LIB_HSA_QUEUE_HOOKS_HEADERS})
