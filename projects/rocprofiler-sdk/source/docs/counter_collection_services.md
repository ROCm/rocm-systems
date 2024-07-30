# Counter Collection Services

## Definitions

*Profile Config*: A configuration to specify what counters should be collected on an agent. This needs to be supplied to various counter collection APIs to initiate collection of counter data. Profiles are agent specific and cannot be used on different agents.

*Counter ID*: Unique ID (per-architecture) that specifies the counter. The counter interface can be used to fetch information about the counter (such as its name or expression).

*Instance ID*: Unique record id encoding both the counter id and dimension for a specific collected value.

*Dimension*: Dimensions provide context to the raw counter values to specify the specific hardware register (such as shader engine) that the value was collected from. All counter values have dimension data encoded in its instance id and functions in the counter interface can be used to extract the values for individual dimensions. There following dimensions are currently supported by rocprofiler-sdk:

```c
    ROCPROFILER_DIMENSION_XCC,            ///< XCC dimension of result
    ROCPROFILER_DIMENSION_AID,            ///< AID dimension of result
    ROCPROFILER_DIMENSION_SHADER_ENGINE,  ///< SE dimension of result
    ROCPROFILER_DIMENSION_AGENT,          ///< Agent dimension
    ROCPROFILER_DIMENSION_SHADER_ARRAY,   ///< Number of shader arrays
    ROCPROFILER_DIMENSION_WGP,            ///< Number of workgroup processors
    ROCPROFILER_DIMENSION_INSTANCE,       ///< From unspecified hardware register
```

## Using The Counter Collection Service

There are two modes for the counter collection service: *dispatch profiling* where counters are collected on a per kernel launch basis and *agent profiling* where counters are collected on a device level. Dispatch profiling is useful for collecting highly detailed counters for a specific kernel execution in isolation (Note: dispatch profiling allows only a single kernel to execute in hardware at a time). Agent profiling is useful for collecting device level counters not tied to a specific kernel execution (i.e. collecting counter values for a specific time range). 

This guide explains how to setup dispatch and agent profiling along will describing the usage of the common counter collection APIs. More detail on the APIs themselves (as well as non-common options) is available in the API documentation. Fully functional examples of both dispatch and agent profiling can be found on the sample directory of rocprofiler-sdk.

### tool_init() setup

The setup for dispatch and agent profiling is similar (with only minor changes needed to adapt code from one to another). In tool_init, similar to tracing services, you need to create a context and a buffer to collect the output. Important Note: buffered_callback in rocprofiler_create_buffer is called when the buffer is full with a vector of collected counter samples, see the buffered callback section below for processing.  

```CPP
rocprofiler_context_id_t ctx;
rocprofiler_buffer_id_t buff;
ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");
ROCPROFILER_CALL(rocprofiler_create_buffer(ctx,
                                            4096,
                                            2048,
                                            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                            buffered_callback, // Callback to process data
                                            user_data,
                                            &buff),
                    "buffer creation failed");
```

After creating a context and buffer to store results, it is highly recommended (but not required) that you construct the profiles for each agent containing the counters you wish to collect in tool_init. Profile creation has a high time cost associated with it due to validating that the counters can be collected on the agent and thus should be avoided in the time critical dispatch profiling callback. After profile setup, the collection service for dispatch or agent profiling can be setup. The following two calls can be used to setup either dispatch or agent profiling (only one can be in use at a time).

```CPP
    /* For Dispatch Profiling */
    // Setup the dispatch profile counting service. This service will trigger the dispatch_callback
    // when a kernel dispatch is enqueued into the HSA queue. The callback will specify what
    // counters to collect by returning a profile config id. 
    ROCPROFILER_CALL(rocprofiler_configure_buffered_dispatch_profile_counting_service(
                         ctx, buff, dispatch_callback, nullptr),
                     "Could not setup buffered service");

    /* For Agent Profiling */
    // set_profile is a callback that is use to select the profile to use when
    // the context is started. It is called at every rocprofiler_ctx_start() call.
    ROCPROFILER_CALL(rocprofiler_configure_agent_profile_counting_service(
                         ctx, buff, agent_id, set_profile, nullptr),
                     "Could not setup buffered service");
```

#### Profile Setup

The first step in constructing a counter collection profile is to find the GPU agents on the machine. A profile will need to be created for each set of counters you want to collect on every agent on the machine. You can use rocprofiler_query_available_agents to find agents on the system. The below example will collect all GPU agents on the device and store them in the vector agents.

```CPP
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

To identify the counters that an agent supports, you can query the available counters with rocprofiler_iterate_agent_supported_counters. An example with a single agent (returning the available counters in gpu_counters) would be the following:

```CPP
    std::vector<rocprofiler_counter_id_t> gpu_counters;

    // Iterate all the counters on the agent and store them in gpu_counters.
    ROCPROFILER_CALL(rocprofiler_iterate_agent_supported_counters(
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
```

rocprofiler_counter_id_t is a handle to a counter. The information about the counter (such as its name) can be fetched using rocprofiler_query_counter_info.

```CPP
    for(auto& counter : gpu_counters)
    {
        // Contains name and other attributes about the counter.
        // See API documenation for more info on the contents of this struct.
        rocprofiler_counter_info_v0_t version;
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&version)),
            "Could not query info for counter");
    }
```

After you have identified a set of counters you wish to collect, a profile can be constructed by passing a list of these counters to rocprofiler_create_profile_config.

```C++
    // Create and return the profile
    rocprofiler_profile_config_id_t profile;
    ROCPROFILER_CALL(rocprofiler_create_profile_config(
                         agent, counters_array, counters_array_count, &profile),
                     "Could not construct profile cfg");
```

The created profile can in turn be used for both dispatch and agent counter collection services. 

##### Special Notes On Profile Behavior
- Profile created is *only valid* for the agent it was created for.
- Profiles are immutable. If a new counter set is desired to be collected, construct a new profile. 
- A single profile can be used multiple times on the same agent. 
- Counter IDs that are supplied to rocprofiler_create_profile_config are *agent specific* and cannot be used to construct profiles for other agents.

### Dispatch Profiling Callback

When a kernel is dispatched, a dispatch callback is issued to the tool to allow for the selection of counters to collect for the dispatch (via supplying a profile). 

```CPP
void
dispatch_callback(rocprofiler_profile_counting_dispatch_data_t dispatch_data,
                  rocprofiler_profile_config_id_t*             config,
                  rocprofiler_user_data_t* user_data,
                  void* /*callback_data_args*/)
```

Dispatch data contains information about the dispatch that is being launched (such as its name) and config is where the tool can specify the profile (and in turn counters) to collect for the dispatch. If no profile is supplied, no counters are collected for this dispatch. User data contains user data supplied to rocprofiler_configure_buffered_dispatch_profile_counting_service. 

### Agent Set Profile Callback

This callback is called when the context is started and allows for the tool to specify the profile to be used. 

```CPP
void
set_profile(rocprofiler_context_id_t                 context_id,
            rocprofiler_agent_id_t                   agent,
            rocprofiler_agent_set_profile_callback_t set_config,
            void*)
```

The profile to be used for this agent is specified by calling set_config(agent, profile). 

### Buffered Callback

Data from collected counter values is returned via a buffered callback. The buffered callback routines are similar between dispatch and agent profiling with the exception that some data (such as kernel launch ids) are not available in agent profiling mode. A sample iteration to print out counter collection data is the following:

```CPP
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
           header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
        {
            // Print the returned counter data.
            auto* record =
                static_cast<rocprofiler_profile_counting_dispatch_record_t*>(header->payload);
            ss << "[Dispatch_Id: " << record->dispatch_info.dispatch_id
               << " Kernel_ID: " << record->dispatch_info.kernel_id
               << " Corr_Id: " << record->correlation_id.internal << ")]\n";
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
                header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            // Print the returned counter data.
            auto* record = static_cast<rocprofiler_record_counter_t*>(header->payload);
            rocprofiler_counter_id_t counter_id = {.handle = 0};

            rocprofiler_query_record_counter_id(record->id, &counter_id);

            ss << "  (Dispatch_Id: " << record->dispatch_id << " Counter_Id: " << counter_id.handle
               << " Record_Id: " << record->id << " Dimensions: [";

            for(auto& dim : counter_dimensions(counter_id))
            {
                size_t pos = 0;
                rocprofiler_query_record_dimension_position(record->id, dim.id, &pos);
                ss << "{" << dim.name << ": " << pos << "},";
            }
            ss << "] Value [D]: " << record->counter_value << "),";
        }
    }
```

## Counter Definitions

Counters are defined in yaml format in the file counter_defs.yaml. The counter definition has the following format

```yaml
counter_name:       # Counter name
  architectures:
    gfx90a:         # Architecture name 
      block:        # Block information (SQ/etc)
      event:        # Event ID (used by AQLProfile to identify counter register)
      expression:   # Formula for the counter (if derrived counter)
      description:  # Per-arch description (optional)
    gfx1010:
       ...
  description:      # Description of the counter
```

Architectures can be separately defined with their own definitions (i.e. gfx90a and gfx1010 in the above example). If two or more architectures share the same block/event/expression definition, they can be "/" delimited on a single line (i.e. "gfx90a/gfx1010:"). Hardware metrics have the elements block, event, and description defined. Derrived metrics have the element expression defined (and cannot have block or event defined).

## Derived Metrics

Derrived metrics allow for computations (via expressions) to be performed on collected hardware metrics with the result returned as it it were a real hardware counter.

```yaml
GPU_UTIL:
  architectures:
    gfx942/gfx941/gfx10/gfx1010/gfx1030/gfx1031/gfx11/gfx1032/gfx1102/gfx906/gfx1100/gfx1101/gfx940/gfx908/gfx90a/gfx9:
      expression: 100*GRBM_GUI_ACTIVE/GRBM_COUNT
  description: Percentage of the time that GUI is active
```

GPU_UTIL is an example of a derrived metric which takes the values of two GRBM hardware counters (GRBM_GUI_ACTIVE and GRBM_COUNT) and uses a mathematic expression to calculate the utilization rate of the GPU. Expressions support the standard set of math operators (/,*,-,+) along with a set of special functions (reduce and accumulate).

### Reduce Function

```yaml
expression: 100*reduce(GL2C_HIT,sum)/(reduce(GL2C_HIT,sum)+reduce(GL2C_MISS,sum))
```

Reduce() reduces counter values across all dimensions (shader engine, SIMD, etc) to produce a single output value. This is useful when you want to collect and compare values across the entire device. There are a number of reduction operations that can be perfomed: sum, average (avr), minimum value (selects minimum value across all dimensions, min), and max (selects the maximum value across all dimensions). For example reduce(GL2C_HIT,sum) sums all GL2C_HIT hardware register values together to return a single output value.

### Accumulate Function
```yaml
expression: accumulate(<basic_level_counter>, <resolution>)
```
#### Description
- The accumulate metric is used to sum the values of a basic level counter over a specified number of cycles. By setting the resolution parameter, you can control the frequency of the summing operation:
    - HIGH_RES: Sums up the basic counter every clock cycle. Captures the value every single cycle for higher accuracy, suitable for fine-grained analysis.
    - LOW_RES: Sums up the basic counter every four clock cycles. Reduces the data points and provides less detailed summing, useful for reducing data volume.
    - NONE: Does nothing and is equivalent to collecting basic_level_counter. Outputs the value of the basic counter without any summing operation.

#### Usage
```yaml
MeanOccupancyPerCU:
  architectures:
    gfx942/gfx941/gfx940:
      expression: accumulate(SQ_LEVEL_WAVES,HIGH_RES)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM
  description: Mean occupancy per compute unit.
```
    <metric name="MeanOccupancyPerCU" expr=accumulate(SQ_LEVEL_WAVES,HIGH_RES)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM descr="Mean occupancy per compute unit."></metric>
- MeanOccupancyPerCU: This metric calculates the mean occupancy per compute unit. It uses the accumulate function with HIGH_RES to sum the SQ_LEVEL_WAVES counter at every clock cycle. This sum is then divided by GRBM_GUI_ACTIVE and the number of compute units (CU_NUM) to derive the mean occupancy.
