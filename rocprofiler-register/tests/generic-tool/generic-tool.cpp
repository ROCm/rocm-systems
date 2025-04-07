
#include <pthread.h>
#include <cstdint>
#include <stdexcept>
#include <string_view>

extern "C" {
typedef struct rocprofiler_client_id_t
{
    const char*    name;    ///< clients should set this value for debugging
    const uint32_t handle;  ///< internal handle
} rocprofiler_client_id_t;

typedef void (*rocprofiler_client_finalize_t)(rocprofiler_client_id_t);

typedef int (*rocprofiler_tool_initialize_t)(rocprofiler_client_finalize_t finalize_func,
                                             void*                         tool_data);

typedef void (*rocprofiler_tool_finalize_t)(void* tool_data);

typedef struct rocprofiler_tool_configure_result_t
{
    size_t                        size;        ///< in case of future extensions
    rocprofiler_tool_initialize_t initialize;  ///< context creation
    rocprofiler_tool_finalize_t   finalize;    ///< cleanup
    void* tool_data;  ///< data to provide to init and fini callbacks
} rocprofiler_tool_configure_result_t;

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
    __attribute__((visibility("default")));

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* tool_id)
{
    (void) version;
    (void) runtime_version;
    (void) priority;
    (void) tool_id;

    return nullptr;
}
}
