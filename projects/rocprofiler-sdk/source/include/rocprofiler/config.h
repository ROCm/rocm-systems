// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler/rocprofiler.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROCPROFILER_API_VERSION_ID 1
#define ROCPROFILER_DOMAIN_OPS_MAX 512
#define ROCPROFILER_DOMAIN_OPS_RESERVED                                                            \
    ((ROCPROFILER_DOMAIN_OPS_MAX * ROCPROFILER_TRACER_ACTIVITY_DOMAIN_LAST / 8))

typedef uint64_t (*rocprofiler_external_cid_cb_t)(rocprofiler_tracer_activity_domain_t,
                                                  uint32_t,
                                                  uint64_t);
typedef int (*rocprofiler_filter_name_t)(const char*);
typedef int (*rocprofiler_filter_op_id_t)(uint32_t);
typedef int (*rocprofiler_filter_range_t)(uint32_t, uint32_t);
typedef int (*rocprofiler_filter_dispatch_id_t)(uint64_t);

/// permits tools opportunity to modify the correlation id based on the domain, op, and
/// the rocprofiler generated correlation id
struct rocprofiler_correlation_config
{
    rocprofiler_external_cid_cb_t external_id_callback;
};

/// how the tools specify the tracing domain and (optionally) which operations in the
/// domain they want to trace
struct rocprofiler_domain_config
{
    rocprofiler_tracer_callback_t callback;
    char                          reserved0[sizeof(uint64_t)];
    char                          reserved1[ROCPROFILER_DOMAIN_OPS_RESERVED];
};

/// for buffered callbacks, the tool provides a callback to create a buffer and the size
struct rocprofiler_buffer_config
{
    rocprofiler_buffer_callback_t callback;
    uint64_t                      buffer_size;
    // void*                         reserved0;
    char reserved1[sizeof(uint64_t)];
};

/// filters are available to make quick decisions about whether rocprofiler should
/// assemble the data necessary for a callback. This is more for convenience and
/// performance -- anything decisions here could be made in the callback but rocprofiler
/// has to first assemble all the infomation on the callback before it (eventually) gets
/// discarded because the tool has decided it (after configuration), that it no longer
/// wants info meeting certain requirements
struct rocprofiler_filter_config
{
    // filter callbacks
    rocprofiler_filter_name_t        name;
    rocprofiler_filter_op_id_t       hip_function_id;
    rocprofiler_filter_op_id_t       hsa_function_id;
    rocprofiler_filter_range_t       range;
    rocprofiler_filter_dispatch_id_t dispatch_id;

    // reserved padding
    char padding[24 * sizeof(void*)];
};

/// this is the "single source of truth" for the capabilities of rocprofiler.
/// you can one configuration that activates all the capabilities you want
/// and holistically start/stop the sum of those features. Alternatively,
/// you can have multiple configurations in order to activate certain features
/// modularly.
///
/// The general workflow is:
///
/// 1. invoke rocprofiler_allocate_config(...)
///     - rocprofiler allocates any space internally needed for the config
///     - rocprofiler sets a few initial values:
///         - "size" to the size of the config structure used internally
///         - "api_version" to the version id of the API in the rocprofiler library that
///           is being used.
///         - these two values can be used by the tool to identify any potential
///           incompatibilities that the tool might want to know about
///     - rocprofiler checks whether it is too late to configure the tool, e.g.
///       something went wrong and rocprofiler was not able to set itself up as
///       the intercepter
/// 2. tool sets up the configuration struct and sets the "size" variable to the size of
///   their configuration struct and sets the "compat_version" field to the
///   ROCPROFILER_API_VERSION_ID defined by the rocprofiler headers when the tool was
///   built
///     - in other words, the user can communicate to rocprofiler, don't read
///       past this distance in my configuration struct and I built against X version
///       so assume the default behavior and capabilties of version X.
/// 3. tool passes this struct to rocprofiler_validate_config(...)
///     - this step checks the config in isolation and will communicate any potential
///       warnings/issues with that configuration, e.g. rocprofiler_X_config is needed,
///       to HW counters XYZ are not available, etc. The tool then has an opportunity
///       to address these issues however they see fit.
/// 4. tool passes this struct to rocprofiler_start_config(...)
///     - internally, we make a call to rocprofiler_validate_config(...) and if any
///       issues still exist with the config in isolation, rocprofiler tells the app
///       to abort -- mechanisms were provided to prevent aborting prior to this call,
///       aborting the app at this point is to guard against rocprofiler "silently"
///       not working because error codes were ignored
///     - rocprofiler then checks whether this config can actually be activated
///       alongside any other active configuration, e.g. this config wants 4 HW counters
///       and another wants 4 HW counters but we can only activate 6 out of 8 of
///       them in this run. Any issues here will not abort execution but, instead,
///       the features of this configuration will not happen (i.e. config won't be
///       activated) and the issues will be communicated with error codes -- giving
///       the tool the opportunity to address the conflicts (i.e. only request tracing
///       and no HW counters) before attempting to activate the modified config.
///     - once rocprofiler determines all features of a config can be activated, it
///       makes an internal copy of the config and returns an identifier for that
///       configuration. The tool is then free to delete the config and any modification
///       to the config will NOT be reflected in the behavior of rocprofiler.
///
///
struct rocprofiler_config
{
    // size is used to ensure that we never read past the end of the version
    size_t                                 size;            // = sizeof(rocprofiler_config)
    uint32_t                               compat_version;  // set by user
    uint32_t                               api_version;     // set by rocprofiler
    uint64_t                               reserved0;       // internal field
    void*                                  user_data;       // data passed to callbacks
    struct rocprofiler_correlation_config* correlation_id;  // = &my_cid_config (optional)
    struct rocprofiler_buffer_config*      buffer;          // = &my_buffer_config (required)
    struct rocprofiler_domain_config*      domain;          // = &my_domain_config (required)
    struct rocprofiler_filter_config*      filter;          // = &my_filter_config (optional)
};

/// \brief returns a properly initialized config struct and allocates any data structures
/// necessary for the config to be used
///
/// \param [out] cfg may adjust config or assign values within structs.
rocprofiler_status_t
rocprofiler_allocate_config(struct rocprofiler_config* cfg);

/// \brief rocprofiler validates config, checks for conflicts, etc. Ensures that
///  the configuration is valid *in isolation*, e.g. it may check that the user
///  set the compat_version field and that required config fields, such as buffer
///  are set. This function will be called before \ref rocprofiler_start_config
///  but is provided to help the user validate one or more configs without starting
///  them
///
/// \param [in] cfg configuration to validate
rocprofiler_status_t
rocprofiler_validate_config(const struct rocprofiler_config* cfg);

/// \brief rocprofiler activates configuration and provides a context identifier
/// \param [in] cfg may adjust config or assign values within structs. If error
///                 occurs, could nullptr valid sub-configs and leave the pointers to
///                 invalid configs
/// \param [out] id the context identifier for this config.
rocprofiler_status_t
rocprofiler_start_config(struct rocprofiler_config*, rocprofiler_context_id_t* id);

/// \brief disable the configuration.
rocprofiler_status_t rocprofiler_stop_config(rocprofiler_context_id_t);

///
///
/// the following 4 functions may be changed to permit removing domain/ops and/or
/// identifying domains and operations via strings
///
///
rocprofiler_status_t
rocprofiler_domain_set_domain(struct rocprofiler_domain_config*,
                              rocprofiler_tracer_activity_domain_t);

rocprofiler_status_t
rocprofiler_domain_add_domains(struct rocprofiler_domain_config*,
                               rocprofiler_tracer_activity_domain_t*,
                               size_t);

rocprofiler_status_t
rocprofiler_domain_add_op(struct rocprofiler_domain_config*,
                          rocprofiler_tracer_activity_domain_t,
                          uint32_t);

rocprofiler_status_t
rocprofiler_domain_add_ops(struct rocprofiler_domain_config*,
                           rocprofiler_tracer_activity_domain_t,
                           uint32_t*,
                           size_t);

#ifdef __cplusplus
}
#endif
