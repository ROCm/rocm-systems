/**
 * @file profiler_hub_types.h
 * @brief Types shared by the profiler-hub C ABI (see profiler_hub.h).
 */

#ifndef PROFILER_HUB_TYPES_INC
#define PROFILER_HUB_TYPES_INC

// NOLINTBEGIN
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /** @brief Result code returned by every profiler-hub C API function. */
    typedef enum
    {
        PH_RESULT_SUCCESS,                   /**< Call completed successfully. */
        PH_RESULT_CONTEXT_ALLOCATION_FAILED, /**< Context could not be created (e.g. trace
                                                 could not be opened/parsed). */
        PH_RESULT_INVALID_CONTEXT,           /**< Context argument was null or invalid. */
        PH_RESULT_INVALID_ARGUMENT,          /**< A non-context argument was null or
                                                 invalid. */
    } ph_result_t;

    struct ph_ctx;
    struct ph_future;

    /** @brief Opaque handle to a trace context. Created by ph_ctx_create(),
     *         released by ph_ctx_free(). */
    typedef struct ph_ctx*   ph_ctx_t;
    typedef struct ph_trace* ph_trace_t;
    /** @brief Opaque handle to an in-flight asynchronous task. */
    typedef struct ph_future* ph_future_t;

    /** @brief Semantic version triple (major.minor.patch). */
    typedef struct
    {
        uint32_t major;
        uint32_t minor;
        uint32_t patch;
    } ph_version_t;

    /**
     * @brief A single track in a trace.
     * @note track_name points into memory owned by the ph_ctx_t that
     *       produced it; do not free it and do not use it after the context
     *       is freed.
     */
    typedef struct
    {
        uint32_t    id;
        const char* track_name;
        uint32_t    nid; /**< Node id this track belongs to. */
        uint32_t    pid; /**< Process id this track belongs to. */
        uint32_t    tid; /**< Thread id this track belongs to, or 0 if unknown. */
    } ph_track_t;

    /**
     * @brief A list of tracks.
     * @note tracks points into memory owned by the ph_ctx_t that produced
     *       this list; valid only until that context is freed or the list is
     *       re-queried.
     */
    typedef struct
    {
        uint32_t    list_size;
        ph_track_t* tracks;
    } ph_track_list_t;

    /**
     * @brief Node identity information.
     * @note name points into memory owned by the producing ph_ctx_t.
     */
    typedef struct
    {
        uint32_t    id;
        const char* machine_id;
        const char* system_name;
        const char* hostname;
        const char* release;
        const char* version;
        const char* hardware_name;
        const char* domain_name;
    } ph_node_info_t;

    /**
     * @brief A compute agent (e.g. CPU/GPU) belonging to a node.
     * @note All const char* fields point into memory owned by the producing
     *       ph_ctx_t.
     */
    typedef struct
    {
        uint32_t    id;
        const char* agent_type;
        uint32_t    absolute_index;
        uint32_t    logical_index;
        uint32_t    uuid;
        const char* name;
        const char* model_name;
        const char* vendor_name;
        const char* product_name;
        const char* user_name;
    } ph_agent_t;

    /** @brief A list of agents; same lifetime rule as ph_track_list_t. */
    typedef struct
    {
        uint32_t    list_size;
        ph_agent_t* agents;
    } ph_agent_list_t;

    /** @brief Node info bundled with its agents and tracks. */
    typedef struct
    {
        ph_node_info_t  info;
        ph_agent_list_t agents;
        ph_track_list_t track_list;
    } ph_node_t;

    typedef ph_version_t ph_library_version_t;
    typedef ph_version_t ph_schema_version_t;

    /** @brief Callback signature for ph_future_get() (not yet implemented). */
    typedef void (*ph_task_fn)(void* user_data);

    // NOLINTEND

#ifdef __cplusplus
}
#endif

#endif  // #ifndef PROFILER_HUB_TYPES_INC
