.. meta::
    :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
    :keywords: ROCprofiler-SDK API reference, ROCprofiler-SDK counter collection

.. _rocprofiler_sdk_counter_collection_services:

ROCprofiler-SDK counter collection services
===========================================

There are two modes of counter collection service:

- **Dispatch counting**: In this mode, counters are collected on a per-kernel launch basis. This mode is useful for collecting highly detailed counters for a specific kernel execution in isolation. Note that dispatch counting allows only a single kernel to execute in hardware at a time.

- **Device counting**: In this mode, counters are collected on a device level. This mode is useful for collecting device level counters not tied to a specific kernel execution, which encompasses collecting counter values for a specific time range.

This topic explains how to setup dispatch and device counting and use common counter collection APIs. For details on the APIs including the less commonly used counter collection APIs, see the API library. For fully functional examples of both dispatch and device counting, see `Samples <https://github.com/ROCm/rocprofiler-sdk/tree/amd-mainline/samples>`_.

Definitions
-----------

**Profile Config**: A configuration to specify the counters to be collected on an agent. This must be supplied to various counter collection APIs to initiate collection of counter data. Profiles are agent-specific and can't be used on different agents.

**Counter ID**: Unique Id (per-architecture) that specifies the counter. The counter Id can be used to fetch counter information such as its name or expression.

**Instance ID**: Unique record Id that encodes the counter Id and dimension for a collected value.

**Dimension**: Dimensions help to provide context to the raw counter values by specifying the hardware register that is the source of counter collection such as a shader engine. All counter values have dimension data encoded in their instance Id, which allows you to extract the values for individual dimensions using functions in the counter interface. The following dimensions are supported:

.. code-block:: c

    ROCPROFILER_DIMENSION_XCC,            ///< XCC dimension of result
    ROCPROFILER_DIMENSION_AID,            ///< AID dimension of result
    ROCPROFILER_DIMENSION_SHADER_ENGINE,  ///< SE dimension of result
    ROCPROFILER_DIMENSION_AGENT,          ///< Agent dimension
    ROCPROFILER_DIMENSION_SHADER_ARRAY,   ///< Number of shader arrays
    ROCPROFILER_DIMENSION_WGP,            ///< Number of workgroup processors
    ROCPROFILER_DIMENSION_INSTANCE,       ///< From unspecified hardware register

Using the Counter Collection Service
------------------------------------

The setup for dispatch and device counting is similar with only minor changes needed to adapt code from one to another. Here are the steps required to configure the counter collection services:

tool_init() setup
+++++++++++++++++++

Similar to tracing services, you must create a context and a buffer to collect the output when initializing the tool.

.. code-block:: cpp

    rocprofiler_context_id_t ctx{0};
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


After creating a context and buffer to store results in ``tool_init``, it is highly recommended but not mandatory for you to construct the profiles for each agent, containing the counters for collection. Profile creation should be avoided in the time critical dispatch counting callback as it involves validating if the counters can be collected on the agent. After profile setup, you can set up the collection service for dispatch or device counting. To set up either dispatch or device counting (only one can be used at a time), use:

.. code-block:: cpp

    /* For Dispatch Counting */
    // Setup the dispatch profile counting service. This service will trigger the dispatch_callback
    // when a kernel dispatch is enqueued into the HSA queue. The callback will specify what
    // counters to collect by returning a profile config id.
    ROCPROFILER_CALL(rocprofiler_configure_buffer_dispatch_counting_service(
                         ctx, buff, dispatch_callback, nullptr),
                     "Could not setup buffered service");

    /* For Agent Counting */
    // set_profile is a callback that is use to select the profile to use when
    // the context is started. It is called at every rocprofiler_ctx_start() call.
    ROCPROFILER_CALL(rocprofiler_configure_device_counting_service(
                         ctx, buff, agent_id, set_profile, nullptr),
                     "Could not setup buffered service");


Profile Setup
-------------

1. The first step in constructing a counter collection profile is to find the GPU agents on the machine. You must create a profile for each set of counters to be collected on every agent on the machine. You can use ``rocprofiler_query_available_agents`` to find agents on the system. The following example collects all GPU agents on the device and stores them in the vector agents:

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

2. To identify the counters supported by an agent, query the available counters with ``rocprofiler_iterate_agent_supported_counters``. Here is an example of a single agent returning the available counters in ``gpu_counters``:

.. code-block:: cpp

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

3. ``rocprofiler_counter_id_t`` is a handle to a counter. To fetch information about the counter such as its name, use ``rocprofiler_query_counter_info``:

.. code-block:: cpp

    for(auto& counter : gpu_counters)
    {
        // Contains name and other attributes about the counter.
        // See API documentation for more info on the contents of this struct.
        rocprofiler_counter_info_v0_t info;
        ROCPROFILER_CALL(
            rocprofiler_query_counter_info(
                counter, ROCPROFILER_COUNTER_INFO_VERSION_0, static_cast<void*>(&info)),
            "Could not query info for counter");
    }


4. After identifying the counters to be collected, construct a profile by passing a list of these counters to ``rocprofiler_create_counter_config``.

.. code-block:: cpp

    // Create and return the profile
    rocprofiler_counter_config_id_t profile;
    ROCPROFILER_CALL(rocprofiler_create_counter_config(
                         agent, counters_array, counters_array_count, &profile),
                     "Could not construct profile cfg");


5. You can use the created profile for both dispatch and agent counter collection services.

.. note::
    Points to note on profile behavior:

    - Profile created is *only valid* for the agent it was created for.
    - Profiles are immutable. To collect a new counter set, construct a new profile.
    - A single profile can be used multiple times on the same agent.
    - Counter Ids supplied to ``rocprofiler_create_counter_config`` are *agent-specific* and can't be used to construct profiles for other agents.

Dispatch Counting Callback
--------------------------

When a kernel is dispatched, a dispatch callback is issued to the tool to allow selection of counters to be collected for the dispatch by supplying a profile.

.. code-block:: cpp

    void
    dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                      rocprofiler_counter_config_id_t*             config,
                      rocprofiler_user_data_t* user_data,
                      void* /*callback_data_args*/)

``dispatch_data`` contains information about the dispatch being launched such as its name. ``config`` is used by the tool to specify the profile, which allows counter collection for the dispatch. If no profile is supplied, no counters are collected for this dispatch. ``user_data`` contains user data supplied to ``rocprofiler_configure_buffered_dispatch_profile_counting_service``.

Agent Set Profile Callback
--------------------------

This callback is invoked after the context starts and allows the tool to specify the profile to be used.

.. code-block:: cpp

    void
    set_profile(rocprofiler_context_id_t               context_id,
                rocprofiler_agent_id_t                 agent,
                rocprofiler_device_counting_agent_cb_t set_config,
                void*)

The profile to be used for this agent is specified by calling ``set_config(agent, profile)``.

Buffered callback
++++++++++++++++++

Data from collected counter values is returned through a buffered callback. The buffered callback routines are similar for dispatch and device counting except that some data such as kernel launch Ids is not available in device counting mode. Here is a sample iteration to print out counter collection data:

.. code-block:: cpp

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
           header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
        {
            // Print the returned counter data.
            auto* record =
                static_cast<rocprofiler_dispatch_counting_service_record_t*>(header->payload);
            ss << "[Dispatch_Id: " << record->dispatch_info.dispatch_id
               << " Kernel_ID: " << record->dispatch_info.kernel_id
               << " Corr_Id: " << record->correlation_id.internal << ")]\n";
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS &&
                header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
        {
            // Print the returned counter data.
            auto* record = static_cast<rocprofiler_counter_record_t*>(header->payload);
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

Counter Definitions
-------------------

Counters are defined in yaml format in the ``counter_defs.yaml`` file. The counter definition has the following format:

.. code-block:: yaml

    counter_name:       # Counter name
      architectures:
        gfx90a:         # Architecture name
          block:        # Block information (SQ/etc)
          event:        # Event ID (used by AQLProfile to identify counter register)
          expression:   # Formula for the counter (if derived counter)
          description:  # Per-arch description (optional)
        gfx1010:
           ...
      description:      # Description of the counter

You can separately define the counters for different architectures as shown in the preceding example for gfx90a and gfx1010. If two or more architectures share the same block, event, or expression definition, they can be specified together using "/" delimiter ("gfx90a/gfx1010:"). Hardware metrics have the elements block, event, and description defined. Derived metrics have the element expression defined and can't have block or event defined.

Derived Metrics
---------------

Derived metrics are expressions performing computation on collected hardware metrics. These expressions produce result similar to a real hardware counter.

.. code-block:: yaml

    GPU_UTIL:
      architectures:
        gfx942/gfx941/gfx10/gfx1010/gfx1030/gfx1031/gfx11/gfx1032/gfx1102/gfx906/gfx1100/gfx1101/gfx940/gfx908/gfx90a/gfx9:
          expression: 100*GRBM_GUI_ACTIVE/GRBM_COUNT
      description: Percentage of the time that GUI is active

In the preceding example, ``GPU_UTIL`` is a derived metric that uses a mathematic expression to calculate the utilization rate of the GPU using values of two GRBM hardware counters ``GRBM_GUI_ACTIVE`` and ``GRBM_COUNT``. Expressions support the standard set of math operators (/,*,-,+) along with a set of special functions such as reduce and accumulate.

Reduce Function
++++++++++++++++

.. code-block:: yaml

    Expression: 100*reduce(GL2C_HIT,sum)/(reduce(GL2C_HIT,sum)+reduce(GL2C_MISS,sum))

The reduce function reduces counter values across all dimensions such as shader engine, SIMD, and so on, to produce a single output value. This helps to collect and compare values across the entire device. Here are the common reduction operations:

- ``sum``: Sums to create a single output. For example, ``reduce(GL2C_HIT,sum)`` sums all ``GL2C_HIT`` hardware register values.
- ``avr``: Calculates the average across all dimensions.
- ``min``: Selects minimum value across all dimensions.
- ``max``: Selects the maximum value across all dimensions.

.. code-block:: yaml

    expression: reduce(X,sum,[DIMENSION_XCC])

Reduce() also supports dimension wise reduction, when provided dimensions in 3rd parameter. In the expression above, if ``X`` has two dimensions ``DIMENSION_XCC``, ``DIMENSION_SHADER_ARRAY``, and ``DIMENSION_WGP``, the reduce happens across counter values where ``DIMENSION_SHADER_ARRAY`` and ``DIMENSION_WGP`` dimensions are same as shown below.

Let's say DIM sizes of XCC, SHADER_ARRAY(SH), WGP be 2, 4, 4 respectively.

Raw Counter Data in 3D space:

#### XCC[0]:

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    | SH[0] |   1  |   2  |   3  |   4  |
    | SH[1] |   5  |   6  |   7  |   8  |
    | SH[2] |   9  |   10 |   11 |   12 |
    | SH[3] |   13 |   14 |   15 |   16 |

#### XCC[1]:

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    | SH[0] |   1  |   2  |   3  |   4  |
    | SH[1] |   5  |   6  |   7  |   8  |
    | SH[2] |   9  |   10 |   11 |   12 |
    | SH[3] |   13 |   14 |   15 |   16 |

Reducing XCC dim with sum, results to 2D space with only WGP and SH.

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    | SH[0] |  2   |   4  |   6  |   8  |
    | SH[1] |  10  |   12 |   14 |   16 |
    | SH[2] |  18  |   20 |   22 |   24 |
    | SH[3] |  26  |   28 |   30 |   32 |

similarly, for ``reduce(X,sum,[DIMENSION_XCC,DIMENSION_SHADER_ARRAY])`` results in only WGP dimension.

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    |       |  56  |  64  |  72  |  80  |

Select Function
++++++++++++++++

.. code-block:: yaml

    expression: select(Y, [DIMENSION_XCC=[0],DIMENSION_SHADER_ENGINE=[2]])

select() only returns counter values which match the dimension indexes provided by the user in expression. This operation is to allow a user to state they only want to select specific dimensions index. Supported dimensions include ``DIMENSION_XCC, DIMENSION_AID, DIMENSION_SHADER_ENGINE, DIMENSION_AGENT, DIMENSION_SHADER_ARRAY, DIMENSION_WGP, DIMENSION_INSTANCE``. For example ``select(Y, [DIMENSION_XCC=[0],DIMENSION_SHADER_ENGINE=[2]])`` gives counter values which are from DIMENSION_XCC= 0 and DIMENSION_SHADER_ENGINE= 2 for Y Metric.

Let's say Y has XCC, SHADER_ENGINE (SE), WGP dimensions with sizes 2, 4, 4 respectively.

Raw Counter Data in 3D space:

#### XCC[0]:

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    | SE[0] |   1  |   2  |   3  |   4  |
    | SE[1] |   5  |   6  |   7  |   8  |
    | SE[2] |   9  |   10 |   11 |   12 |
    | SE[3] |   13 |   14 |   15 |   16 |

#### XCC[1]:

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    | SE[0] |   17 |   18 |   19 |   20 |
    | SE[1] |   21 |   22 |   23 |   24 |
    | SE[2] |   25 |   26 |   27 |   28 |
    | SE[3] |   29 |   30 |   31 |   32 |

Selecting at XCC=0 results to 2D space with WGP and SH dimensions, as shown below.

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    | SE[0] |   1  |   2  |   3  |   4  |
    | SE[1] |   5  |   6  |   7  |   8  |
    | SE[2] |   9  |   10 |   11 |   12 |
    | SE[3] |   13 |   14 |   15 |   16 |

similarly, for ``select(Y, [DIMENSION_XCC=[0],DIMENSION_SHADER_ENGINE=[2]])`` results in only WGP dimension with XCC=0 and SE=2.

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    |       |  9   |  10  |  11  |  12  |

Accumulate Function
-------------------

.. code-block:: yaml

    Expression: accumulate(<basic_level_counter>, <resolution>)

- The accumulate function sums the values of a basic level counter over the specified number of cycles. The ``resolution`` parameter allows you to control the frequency of the following summing operation:

  - ``HIGH_RES``: Sums up the basic level counter every clock cycle. Captures the value every cycle for higher accuracy, which helps in fine-grained analysis.
  - ``LOW_RES``: Sums up the basic level counter every four clock cycles. Reduces the data points and provides less detailed summing, which helps in reducing data volume.
  - ``NONE``: Does nothing and is equivalent to collecting basic level counter. Outputs the value of the basic level counter without performing any summing operation.

**Example:**

.. code-block:: yaml

    MeanOccupancyPerCU:
      architectures:
        gfx942/gfx941/gfx940:
          expression: accumulate(SQ_LEVEL_WAVES,HIGH_RES)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM
      description: Mean occupancy per compute unit.

<metric name="MeanOccupancyPerCU" expr=accumulate(SQ_LEVEL_WAVES,HIGH_RES)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM descr="Mean occupancy per compute unit."></metric>

- ``MeanOccupancyPerCU``: In the preceding example, the ``MeanOccupancyPerCU`` metric calculates the mean occupancy per compute unit. It uses the accumulate function with ``HIGH_RES`` to sum the ``SQ_LEVEL_WAVES`` counter every clock cycle. This sum is then divided by the maximum value of GRBM_GUI_ACTIVE and the number of compute units ``CU_NUM`` to derive the mean occupancy.

Kernel Serialization
--------------------

Counter collection in *dispatch counting* mode requires serialized execution of kernels on a target device. Kernel serialization isolates kernel executions, which helps to collect performance counter data. However, for applications requiring two kernels to execute on the same device simultaneously (co-dependent kernels), kernel serialization leads to deadlock in dispatch counter collection mode. To avoid deadlock in such applications, opt for any of the following options:

- Avoid co-dependent kernels in application.

- Don't collect performance data for co-dependent kernels by using kernel filtration methods in the rocprofv3’s input configuration PMC file.

- Use ROCprofiler-SDK's device-wide counter collection mode to collect performance data. You can use tools such as RDC and PAPI to collect information. Note that the device-wide counter collection captures data for all executions on the device and not specific to the kernels.
