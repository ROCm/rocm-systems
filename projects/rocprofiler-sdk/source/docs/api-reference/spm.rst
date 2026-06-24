.. meta::
    :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
    :keywords: ROCprofiler-SDK API reference, Streaming Performance Monitor, SPM 

.. _SPM:

ROCprofiler-SDK Streaming Performance Monitor method
===================================

Streaming Performance Monitor (SPM) sampling is a profiling method that samples hardware performance counters at regular intervals. This provides a granual insight into behavior of kernel during its execution.
.. warning::

    Risk acknowledgment: The SPM feature is under development and might not be completely stable. Use this beta feature cautiously. It may affect your system's stability and performance. Proceed at your own risk.

    By activating this feature through ``ROCPROFILER_SPM_BETA_ENABLED`` environment variable, you acknowledge and accept the following potential risks:

    - System reboot: This beta feature could cause your hardware to restart unexpectedly.

ROCprofiler-SDK SPM service
------------------------------------

This section describes how to use ROCProfiler-SDK SPM API to configure and use SPM service. For fully functional examples, see `Samples <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk/samples>`_.

Currently,  **Dispatch counting** is supported for SPM. Please refer to :ref:`counter collection services` for information on dispatch counting, counters and profile configuration. 

The set up for SPM service is similar to counter collection services. 

SPM counter service cannot be enabled together with PMC or PC sampling service.

kernels are serialized in dispatch counting SPM service.

Currently, collection of basic counters are supported with SPM.

tool_init() setup
++++++++++++++++++

Here are the steps to set up ``tool_init()``:

.. code-block:: cpp

    auto ctx = rocprofiler_context_id_t{0};
    auto buff = rocprofiler_buffer_id_t{};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");
    ROCPROFILER_CALL(rocprofiler_create_buffer(ctx,
                                                8192,
                                                8192,
                                                ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                spm_sampling_callback, // Callback to process PC samples
                                                user_data,
                                                &buff),
                        "buffer creation failed");

For more details on buffer creation, see :ref:`buffered-services`.
    
.. code-block:: cpp

    /* For Dispatch Counting */
    // Setup the dispatch profile counting service. This service will trigger the dispatch_callback
    // when a kernel dispatch is enqueued into the HSA queue. The callback will specify what
    // counters to collect by returning a profile config id.
    ROCPROFILER_CALL(rocprofiler_spm_configure_buffer_dispatch_service(
                         ctx, buff, spm_dispatch_callback, nullptr),
                     "Could not setup buffered service");




.. code-block:: cpp

    std::vector<rocprofiler_agent_v0_t> agents;

    // Callback used by rocprofiler_query_available_agents to return
    // agents on the device. This can include CPU agents as well.
    // Select GPU agents only (type == ROCPROFILER_AGENT_TYPE_GPU)
    rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                            const void**                agents_arr,
                                                            size_t                      num_agents,
                                                            void*                       udata) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) agents_v->emplace_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    // Query the agents. Only a single callback is made that contains a vector
    // of all agents.
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");

Profile Setup
-------------

1. The first step in constructing a SPM counter collection profile is to find the GPU agents on the machine.

.. code-block:: cpp

    std::vector<rocprofiler_agent_v0_t> agents;

    // Callback used by rocprofiler_query_available_agents to return
    // agents on the device. This can include CPU agents as well. We
    // select GPU agents only (i.e. type == ROCPROFILER_AGENT_TYPE_GPU)
    rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                            const void**                agents_arr,
                                                            size_t                      num_agents,
                                                            void*                       udata) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
            throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) agents_v->emplace_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    // Query the agents, only a single callback is made that contains a vector
    // of all agents.
    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           iterate_cb,
                                           sizeof(rocprofiler_agent_t),
                                           const_cast<void*>(static_cast<const void*>(&agents))),
        "query available agents");

2. To identify the counters supported by an agent, query the available counters with ``rocprofiler_spm_iterate_agent_supported_counters``. Here is an example of a single agent returning the available counters in ``gpu_counters``:

.. code-block:: cpp

    std::vector<rocprofiler_counter_id_t> gpu_counters;

    // Iterate all the counters on the agent and store them in gpu_counters.
    ROCPROFILER_CALL(rocprofiler_spm_iterate_agent_supported_counters(
                         agent,
                         [](rocprofiler_agent_id_t,
                            rocprofiler_counter_id_t* counters,
                            size_t                    num_counters,
                            void*                     user_data) {
                             std::vector<rocprofiler_counter_id_t>* vec =
                                 static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                             for(size_t i = 0; i < num_counters; i++)
                             {
                                 vec->push_back(counters[i]);
                             }
                             return ROCPROFILER_STATUS_SUCCESS;
                         },
                         static_cast<void*>(&gpu_counters)),
                     "Could not fetch supported counters");

3. After identifying the counters to be sampled, construct a profile by passing a list of these counters and input parameters to ``rocprofiler_spm_create_counter_config``.

.. code-block:: cpp

    std::vector<rocprofiler_spm_parameters_t*> input_params{};
    auto                                       param = rocprofiler_spm_parameters_t{
        .size = sizeof(rocprofiler_spm_parameters_t),
        .type = ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES,
        .value = 4200};
    input_params.push_back(&param);

    // Create and return the profile
    rocprofiler_counter_config_id_t profile = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_spm_create_counter_config(
                         agent, counters_array, counters_array_count, input_params.data(), input_params.size(), &profile),
                     "Could not construct profile cfg");

Dispatch Counting Callback
--------------------------

When a kernel is dispatched, a dispatch callback is issued to the tool for supplying a profile.

.. code-block:: cpp

    void
    spm_dispatch_callback(const rocprofiler_spm_dispatch_counting_service_data_t* dispatch_data,
                          rocprofiler_counter_config_id_t*                        config,
                          rocprofiler_user_data_t* user_data,
                          void* /*callback_data_args*/)

``dispatch_data`` contains information about the dispatch being launched such as its name. ``config`` is used by the tool to specify the profile, which allows counter sampling for the dispatch. If no profile is supplied, no counters are collected for this dispatch. ``user_data`` contains user data supplied to ``rocprofiler_spm_configure_buffer_dispatch_service``.


Processing SPM samples
----------------------

SPM buffered dispatch service asynchronously delivers samples via a dedicated callback ``(spm_sampling_callback)``. The following code snippet outlines the process of iterating over samples.

.. code-block:: cpp

    void
    spm_sampling_callback(rocprofiler_context_id_t,
                          rocprofiler_buffer_id_t,
                          rocprofiler_record_header_t** headers,
                          size_t                        num_headers,
                          void*                         user_data,
                          uint64_t                      drop_count)
    {
        for(size_t i = 0; i < num_headers; ++i)
        {
            auto* header = headers[i];

            if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
               header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
            {
                // Process the dispatch header
                auto* record =
                    static_cast<rocprofiler_spm_dispatch_counting_service_data_t*>(header->payload);
            }
            else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
                    header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
            {
                // Process a counter sample
                auto* record = static_cast<rocprofiler_spm_counter_record_t*>(header->payload);
            }
        }
    }

For more information on the data comprising a single sample, see `spm.h <https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-sdk/source/include/rocprofiler-sdk/experimental/spm.h>`_.

.. note::
    A user can synchronously flush buffers via ``rocprofiler_buffer_flush`` that triggers ``spm_sampling_callback``. SPM is currently supported on AMD Instinct MI300, MI325, MI350, and MI355.
