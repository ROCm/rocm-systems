---
myst:
    html_meta:
        "description": "ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software."
        "keywords": "ROCprofiler-SDK API reference, ROCprofiler-SDK PC sampling, Program counter sampling, PC sampling"
---

# ROCprofiler-SDK PC sampling method

Program Counter (PC) sampling is a profiling method that uses statistical approximation of the kernel execution by sampling GPU program counters. Furthermore, this method periodically chooses an active wave in a round robin manner and snapshots its PC. This process takes place on every compute unit simultaneously, making it device-wide PC sampling. The outcome is the histogram of samples, explaining how many times each kernel instruction was sampled.

> **Warning:**
> Risk acknowledgment: The PC sampling feature is under development and might not be completely stable. Use this beta feature cautiously. It may affect your system's stability and performance. Proceed at your own risk.
>
> By activating this feature through `ROCPROFILER_PC_SAMPLING_BETA_ENABLED` environment variable, you acknowledge and accept the following potential risks:
>
> - Hardware freeze: This beta feature could cause your hardware to freeze unexpectedly.
> - Need for cold restart: In the event of a hardware freeze, you might need to perform a cold restart (turning the hardware off and on) to restore normal operations.

## ROCprofiler-SDK PC Sampling Service

This section describes usage of ROCProfiler-SDK PC Sampling API to configure and use PC sampling service. For a fully functional example, see [Samples](https://github.com/ROCm/rocprofiler-sdk/tree/amd-mainline/samples).

### tool_init() Setup

As the PC sampling service belongs to the group of [buffered services](buffered_services.md), it requires a buffer and a context to be set up in this phase.

```cpp
rocprofiler_context_id_t ctx{0};
rocprofiler_buffer_id_t buff;
ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");
ROCPROFILER_CALL(rocprofiler_create_buffer(ctx,
                                            8192,
                                            2048,
                                            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                            pc_sampling_callback, // Callback to process PC samples
                                            user_data,
                                            &buff),
                    "buffer creation failed");
```

For more details about the buffer creation, please refer to the [buffered services section](buffered_services.md).

The PC sampling service is tied to a GPU agent. To extract the list of available agents, one could use the `rocprofiler_query_available_agents` as the following snippet outlines.

```cpp
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
```

Only recent GPU architectures support the feature. To determine whether an agent with `agent_it` supports the PC sampling and what configurations (`rocprofiler_pc_sampling_configuration_t`) are available, one should use the `rocprofiler_query_pc_sampling_agent_configurations`.

```cpp
std::vector<rocprofiler_pc_sampling_configuration_t> available_configurations;

auto cb = [](const rocprofiler_pc_sampling_configuration_t* configs,
             size_t                                         num_config,
             void*                                          user_data) {
    auto* avail_configs = static_cast<avail_configs_vec_t*>(user_data);
    for(size_t i = 0; i < num_config; i++)
    {
        avail_configs->emplace_back(configs[i]);
    }
    return ROCPROFILER_STATUS_SUCCESS;
};

auto status = rocprofiler_query_pc_sampling_agent_configurations(
    agent_id, cb, &available_configurations);
```

Assuming the `available_configurations` contains a single element:

```cpp
rocprofiler_pc_sampling_configuration_t {
    .method = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
    .unit = ROCPROFILER_PC_SAMPLING_UNIT_TIME,
    .min_interval = 1,
    .max_interval = 10000
};
```

one proceeds configuring the PC sampling service on an agent with `agent_id` to generate samples every 1000 micro-seconds in the following way:

```cpp
auto status = rocprofiler_configure_pc_sampling_service(ctx,
                                                        agent_id,
                                                        picked_cfg->method,
                                                        picked_cfg->unit,
                                                        1000,  // 1000 us
                                                        buffer_id,
                                                        0);
if (status == ROCPROFILER_STATUS_SUCCESS)
{
    // PC Sampling service has been configured successfully.
}
else
{
    // code for error handling
}
```

> **Note**
>
> Multiple processes can share the same GPU agent simultaneously, so the following ABA problem is possible on shared systems. Namely, process A can query available configurations and decide to configure the service with configuration CA. However, process B manages to finish configuring the service with configuration CB, meaning process A will fail. Thus, we advise that process A repeat the querying process to observe configuration CB and reuse it for configuring the PC sampling service. Please refer to the [Samples](https://github.com/ROCm/rocprofiler-sdk/tree/amd-mainline/samples) section for more technical details.

### Processing PC Samples (`pc_sampling_callback`)

PC sampling service asynchronously delivers samples via a dedicated callback. The following code outlines the process of iterating over samples.

```cpp
void
pc_sampling_callback(rocprofiler_context_id_t ctx,
                     rocprofiler_buffer_id_t buff,
                     rocprofiler_record_header_t** headers,
                     size_t                        num_headers,
                     void* data,
                     uint64_t drop_count)
{
    for(size_t i = 0; i < num_headers; i++)
    {
        auto* cur_header = headers[i];

        if(cur_header->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
        {
            if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_HOST_TRAP_V0_SAMPLE)
            {
                auto* pc_sample = static_cast<rocprofiler_pc_sampling_record_host_trap_v0_t*>(
                    cur_header->payload);

                // Processing a single sample...
            }
            else
            {
                // ...
            }
        }
    }
}
```

For more information about what data comprises a single sample, please refer to the [pc_sampling.h](https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/include/rocprofiler-sdk/pc_sampling.h).

Note, a user can synchronously flush buffers via `rocprofiler_buffer_flush` that triggers `pc_sampling_callback`.
