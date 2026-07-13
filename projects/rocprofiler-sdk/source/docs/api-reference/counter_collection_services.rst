.. meta::
    :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
    :keywords: ROCprofiler-SDK API reference, ROCprofiler-SDK counter collection

.. _rocprofiler_sdk_counter_collection_services:

ROCprofiler-SDK counter collection services
===========================================

There are two modes of counter collection service:

- **Dispatch counting**: In this mode, counters are collected on a per-kernel launch basis. This mode is useful for collecting highly detailed counters for a specific kernel execution in isolation. Note that dispatch counting allows only a single kernel to execute in hardware at a time.

- **Device counting**: In this mode, counters are collected on a device level. This mode is useful for collecting device level counters not tied to a specific kernel execution, which encompasses collecting counter values for a specific time range.

This topic explains how to setup dispatch and device counting and use common counter collection APIs. For details on the APIs including the less commonly used counter collection APIs, see the API library. For fully functional examples of both dispatch and device counting, see `Samples <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk/samples>`_.

Definitions
-----------

**Profile Config**: A configuration to specify the counters to be collected on an agent. This must be supplied to various counter collection APIs to initiate collection of counter data. Profiles are agent-specific and can't be used on different agents.

**Counter ID**: Unique Id (per-architecture) that specifies the counter. The counter Id can be used to fetch counter information, such as its name or expression.

**Instance ID**: Unique record Id that encodes the counter Id and dimension for a collected value.

**Dimension**: Dimensions help to provide context to the raw counter values by identifying *which* physical hardware unit produced a given value. A single counter is usually replicated across many hardware units (for example, one copy per shader engine), so a single counter produces multiple values per collection. Each value is tagged with a coordinate in one or more dimensions, and this coordinate is encoded into the value's instance Id. You can extract the coordinate for an individual dimension using functions in the counter interface (see :ref:`querying_counter_dimensions`).

The following dimensions are supported:

.. list-table::
    :header-rows: 1
    :widths: 30 15 55

    * - Dimension
      - Hardware unit
      - Meaning
    * - ``ROCPROFILER_DIMENSION_XCC``
      - XCD / XCC
      - Index of the Accelerator Complex Die (XCD, sometimes called the XCC) that produced the value. Only present on GPUs that expose more than one compute die to a single agent (for example, the MI300 series). On single-die agents (for example, MI200, where each die is a separate agent) this dimension is absent.
    * - ``ROCPROFILER_DIMENSION_AID``
      - AID
      - Index of the Accelerator Interconnect/IO Die (AID). Present only on multi-die accelerators that expose IO-die-level blocks (for example, the MI300 series). Typically applies to memory- and IO-related blocks.
    * - ``ROCPROFILER_DIMENSION_SHADER_ENGINE``
      - Shader Engine (SE)
      - Index of the shader engine within a die. Most compute (SQ/GRBM-class) counters are replicated per shader engine.
    * - ``ROCPROFILER_DIMENSION_SHADER_ARRAY``
      - Shader Array (SA)
      - Index of the shader array within a shader engine. A shader engine contains one or more shader arrays.
    * - ``ROCPROFILER_DIMENSION_WGP``
      - Workgroup Processor (WGP) / CU
      - Index of the compute unit within a shader array. Named after the RDNA Workgroup Processor (a WGP groups two CUs); on CDNA parts (MI-series) this indexes the compute unit (CU). Present only for counters collected at CU/WGP granularity.
    * - ``ROCPROFILER_DIMENSION_INSTANCE``
      - Block instance
      - Generic per-block instance index used when a block has multiple hardware instances that do not map onto the named geometric dimensions above (for example, multiple memory channels or multiple copies of a fixed-function block). Also used for constant (non-hardware) counters, which report a single instance.
    * - ``ROCPROFILER_DIMENSION_AGENT``
      - Agent
      - Internal use only. This dimension is not set externally and should not be supplied by tools.

.. note::
    The set of dimensions that applies to a counter is **not** fixed by the SDK and is **not** the same for every architecture or every counter. The dimensions (and the size/extent of each dimension) are determined at runtime for the specific agent and the specific counter, based on the hardware block the counter reads from. For example:

    - ``ROCPROFILER_DIMENSION_XCC`` appears only on multi-XCD agents such as the MI300 series; it does not appear on MI200.
    - ``ROCPROFILER_DIMENSION_WGP`` appears only for counters collected at CU/WGP granularity.
    - The extent of ``ROCPROFILER_DIMENSION_SHADER_ENGINE`` (the number of shader engines) differs between architectures.

    Because of this, you should always **query the dimensions of a counter at runtime** rather than assuming a fixed set (see :ref:`querying_counter_dimensions`). When a counter does not have a given dimension, that dimension's coordinate is encoded as ``0`` for every value. As a result, ``reduce`` on an absent dimension is a no-op, but ``select`` on an absent dimension only matches index ``0`` (selecting any non-zero index drops all values for that counter).

.. _querying_counter_dimensions:

Querying counter dimensions
+++++++++++++++++++++++++++

Dimensions depend on the counter and the specific agent it is collected on. To discover which dimensions a counter has (and their sizes), query the counter info with ``rocprofiler_query_counter_info`` using ``ROCPROFILER_COUNTER_INFO_VERSION_1``. This call does not take an agent Id; it selects a representative agent of that counter's architecture internally and reports the dimensions for that agent. The returned ``rocprofiler_counter_info_v1_t`` reports ``dimensions_count`` and, in ``dimensions``, the name and extent of each dimension. To read the coordinate of a specific value within a dimension from a collected record, use ``rocprofiler_query_record_dimension_position`` with the value's instance Id (see the buffered callback example above).

Using the counter collection service
------------------------------------

The setup for dispatch and device counting is similar, with only minor changes needed to adapt code from one to another. Here are the steps required to configure the counter collection services:

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


After creating a context and buffer to store results in ``tool_init``, it's highly recommended but not mandatory for you to construct the profiles for each agent, containing the counters for collection. Avoid profile creation in the time critical dispatch counting callback as it involves validating if the counters can be collected on the agent. After profile setup, you can set up the collection service for dispatch or device counting. To set up either dispatch or device counting (only one can be used at a time), use:

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


Profile setup
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

3. ``rocprofiler_counter_id_t`` is a handle to a counter. To fetch information about the counter, such as its name, use ``rocprofiler_query_counter_info``:

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

    - The created profile is valid only for the agent it was created for.
    - Profiles are immutable. To collect a new counter set, construct a new profile.
    - A single profile can be used multiple times on the same agent.
    - Counter Ids supplied to ``rocprofiler_create_counter_config`` are agent-specific and can't be used to construct profiles for other agents.

Dispatch counting callback
--------------------------

When a kernel is dispatched, a dispatch callback is issued to the tool to allow selection of counters to be collected for the dispatch by supplying a profile.

.. code-block:: cpp

    void
    dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                      rocprofiler_counter_config_id_t*             config,
                      rocprofiler_user_data_t* user_data,
                      void* /*callback_data_args*/)

``dispatch_data`` contains information about the dispatch being launched, such as its name. ``config`` is used by the tool to specify the profile, which allows counter collection for the dispatch. If no profile is supplied, no counters are collected for this dispatch. ``user_data`` contains user data supplied to ``rocprofiler_configure_buffered_dispatch_profile_counting_service``.

Agent set profile callback
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

Data from the collected counter values is returned through a buffered callback. The buffered callback routines are similar for dispatch and device counting except that some data, such as kernel launch Ids is not available in the device counting mode. Here is a sample iteration to print counter collection data:

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

Counter definitions
-------------------

Counters are defined in yaml format in the ``config.yaml`` file. The counter definition has the following format:

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

You can separately define the counters for different architectures, such as gfx90a and gfx1010, as shown in the preceding example. If two or more architectures share the same block, event, or expression definition, they can be specified together using "/" delimiter, for example, "gfx90a/gfx1010:". Hardware metrics have the elements block, event, and description defined. Derived metrics have the element expression defined and can't have block or event defined.

Firmware restrictions
---------------------

ROCprofiler-SDK follows firmware version restrictions to ensure that counter collection operates on devices with compatible firmware. This helps prevent issues that might arise from using outdated firmware versions that could cause device instability or incorrect counter data.

Firmware restrictions in counter definitions file
++++++++++++++++++++++++++++++++++++++++++++++++++

Firmware restrictions are defined alongside counter definitions in the ``config.yaml`` file. The combined file uses the following schema:

.. code-block:: yaml

    rocprofiler-sdk:
      # Counter definitions schema version
      counters-schema-version: 1

      # Counter definitions
      counters:
        - name: ALUStalledByLDS
          description: 'Percentage of GPUTime ALU units are stalled by LDS'
          properties: []
          definitions:
            - architectures: ["gfx908", "gfx90a"]
              expression: 400*reduce(SQ_WAIT_INST_LDS,sum)/reduce(SQ_WAVES,sum)/reduce(GRBM_GUI_ACTIVE,max)

      # Required: Firmware restrictions schema version
      fw_restriction_schema_version: 1

      # List of firmware restrictions
      firmware_restrictions:
        # Example: CP/MEC firmware restrictions
        - firmware_type: CP
          min_version: 199
          reason: "CP firmware below version 199 has critical bugs affecting compute operations"
          affected_architectures:
            - "gfx908"
            - "gfx90a"
            - "gfx940"

        # Example: SDMA firmware restrictions
        - firmware_type: SDMA
          min_version: 150
          reason: "SDMA firmware below 150 causes system instability and data corruption"
          affected_architectures:
            - "gfx1030"
            - "gfx1100"
            - "gfx1101"

**Schema elements:**

- ``fw_restriction_schema_version`` (required): Integer specifying the schema version (currently 1).
- ``firmware_restrictions``: Array of restriction objects consisting of the following fields:

  - ``firmware_type`` (required): Type of firmware being restricted. Supported types include:

    - ``CP`` or ``MEC``: Command Processor or MicroEngine Compute firmware
    - ``SDMA``: System DMA firmware

  - ``min_version`` (required): Integer specifying the minimum firmware version.
  - ``reason`` (optional): Human-readable explanation for the restriction.
  - ``affected_architectures`` (optional): Array of GPU architecture names, such as "gfx940" and "gfx942" liable to meet this restriction. If empty, the restriction applies to all the architectures.

Counter definitions file location
++++++++++++++++++++++++++++++++++

The firmware restrictions are located within the counter definitions file using the following search order:

1. **Environment variable**: If the ``ROCPROFILER_METRICS_PATH`` environment variable is set, ROCprofiler-SDK looks for ``config.yaml`` in that directory.

2. **Installation path**: If no environment variable is set, ROCprofiler-SDK searches in the installation directory at ``<install_path>/share/rocprofiler-sdk/config.yaml``.

**Example:**

.. code-block:: bash

    # Set custom counter definitions path
    export ROCPROFILER_METRICS_PATH=/path/to/custom/counters/

    # ROCprofiler-SDK will now look for:
    # /path/to/custom/counters/config.yaml
    # (which contains both counter definitions and firmware restrictions)

Derived metrics
---------------

Derived metrics are expressions performing computation on collected hardware metrics. These expressions produce result similar to a real hardware counter.

.. code-block:: yaml

    GPU_UTIL:
      architectures:
        gfx942/gfx941/gfx10/gfx1010/gfx1030/gfx1031/gfx11/gfx1032/gfx1102/gfx906/gfx1100/gfx1101/gfx940/gfx908/gfx90a/gfx9:
          expression: 100*GRBM_GUI_ACTIVE/GRBM_COUNT
      description: Percentage of the time that GUI is active

In the preceding example, ``GPU_UTIL`` is a derived metric that uses a mathematical expression to calculate the GPU utilization rate using values of two GRBM hardware counters ``GRBM_GUI_ACTIVE`` and ``GRBM_COUNT``. Expressions support the standard set of math operators (/,*,-,+) and a set of special functions such as, reduce() and accumulate().

Reduce function
++++++++++++++++

.. code-block:: yaml

    Expression: 100*reduce(GL2C_HIT,sum)/(reduce(GL2C_HIT,sum)+reduce(GL2C_MISS,sum))

The reduce() function reduces counter values across all dimensions, such as shader engine, SIMD, and so on, to produce a single output value. This helps to collect and compare values across the entire device. Here are the common reduction operations:

- ``sum``: Sums all values to create a single output. For example, ``reduce(GL2C_HIT,sum)`` sums all ``GL2C_HIT`` hardware register values.
- ``avr``: Calculates the average across all dimensions.
- ``min``: Selects minimum value across all dimensions.
- ``max``: Selects the maximum value across all dimensions.

.. code-block:: yaml

    expression: reduce(X,sum,[DIMENSION_XCC])

Reduce() also supports dimension-wise reduction, when dimension is provided as the third parameter. In the preceding expression, if ``X`` has dimensions ``DIMENSION_XCC``, ``DIMENSION_SHADER_ARRAY``, and ``DIMENSION_WGP``, the reduce is performed across the counter values where ``DIMENSION_SHADER_ARRAY`` and ``DIMENSION_WGP`` dimensions are same, as shown here:

Assuming the DIM sizes of XCC, SHADER_ARRAY(SH), and WGP to be 2, 4, and 4 respectively, the raw counter data in the 3D space is:

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

Reducing XCC[dim] with sum, results to 2D space with only WGP and SH:

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    | SH[0] |  2   |   4  |   6  |   8  |
    | SH[1] |  10  |   12 |   14 |   16 |
    | SH[2] |  18  |   20 |   22 |   24 |
    | SH[3] |  26  |   28 |   30 |   32 |

similarly, ``reduce(X,sum,[DIMENSION_XCC,DIMENSION_SHADER_ARRAY])`` results in only WGP dimension:

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    |       |  56  |  64  |  72  |  80  |

Select function
++++++++++++++++

.. code-block:: yaml

    expression: select(Y, [DIMENSION_XCC=[0],DIMENSION_SHADER_ENGINE=[2]])

The select() function returns only the counter values matching the dimension indices provided in the expression. This operation helps the user select specific dimension's index. Supported dimensions include ``DIMENSION_XCC``, ``DIMENSION_AID``, ``DIMENSION_SHADER_ENGINE``, ``DIMENSION_AGENT``, ``DIMENSION_SHADER_ARRAY``, ``DIMENSION_WGP``, and ``DIMENSION_INSTANCE``. For example ``select(Y, [DIMENSION_XCC=[0],DIMENSION_SHADER_ENGINE=[2]])`` provides counter values from DIMENSION_XCC= 0 and DIMENSION_SHADER_ENGINE= 2 for Y Metric.

.. note::
    Points to note on ``select`` arguments:

    - **Number of dimensions**: There is no fixed limit of one or two. The examples show one and two dimensions only for brevity. You may constrain as many distinct dimensions as the counter actually has, up to all of the user-visible dimensions listed above. Each dimension may appear at most once in the argument list. To find out which dimensions a counter has, see :ref:`querying_counter_dimensions`.
    - **Single index per dimension**: Each dimension takes exactly one index in square brackets (for example, ``DIMENSION_XCC=[0]``). A comma-separated list (for example, ``DIMENSION_XCC=[0,1]``) and range syntax (for example, ``[0:1]``) are **not** supported and raise a runtime error. To keep several indices, apply ``select`` for each index separately.
    - **Index limits**: Each index must be non-negative and fit within the number of bits allocated to the dimension: ``DIMENSION_XCC``, ``DIMENSION_AID``, ``DIMENSION_SHADER_ENGINE``, ``DIMENSION_SHADER_ARRAY``, and ``DIMENSION_WGP`` use 6 bits (indices ``0``-``63``), while ``DIMENSION_INSTANCE`` uses 10 bits (indices ``0``-``1023``). The index must also be within the actual extent of that dimension for the counter.

Assuming that Y has XCC, SHADER_ENGINE (SE), and WGP dimensions with sizes 2, 4, and 4 respectively, the raw counter data in the 3D space is:

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

Selecting XCC=0 results in a 2D space with WGP and SH dimensions, as shown here:

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    | SE[0] |   1  |   2  |   3  |   4  |
    | SE[1] |   5  |   6  |   7  |   8  |
    | SE[2] |   9  |   10 |   11 |   12 |
    | SE[3] |   13 |   14 |   15 |   16 |

Similarly, ``select(Y, [DIMENSION_XCC=[0],DIMENSION_SHADER_ENGINE=[2]])`` results in only WGP dimension with XCC=0 and SE=2.

.. code-block:: text

    |       |WGP[0]|WGP[1]|WGP[2]|WGP[3]|
    |-------|------|------|------|------|
    |       |  9   |  10  |  11  |  12  |

Accumulate function
-------------------

.. code-block:: yaml

    Expression: accumulate(<basic_level_counter>, <resolution>)

The accumulate() function sums the values of a basic level counter over the specified number of cycles. The ``resolution`` parameter helps to control the frequency of the following summing operations:

- ``HIGH_RES``: Sums up the basic level counter every clock cycle. Captures the value every cycle for higher accuracy, supporting fine-grained analysis.
- ``LOW_RES``: Sums up the basic level counter every four clock cycles. Reduces the data points and provides less detailed summing, helping reduce the data volume.
- ``NONE``: Does nothing and is equivalent to collecting basic level counter. Outputs the value of the basic level counter without performing any summing operation.

**Example:**

.. code-block:: yaml

    MeanOccupancyPerCU:
      architectures:
        gfx942/gfx941/gfx940:
          expression: accumulate(SQ_LEVEL_WAVES,HIGH_RES)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM
      description: Mean occupancy per compute unit.

      <metric name="MeanOccupancyPerCU" expr=accumulate(SQ_LEVEL_WAVES,HIGH_RES)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM descr="Mean occupancy per compute unit."></metric>

``MeanOccupancyPerCU``: In the preceding example, the ``MeanOccupancyPerCU`` metric calculates the mean occupancy per compute unit. It uses the accumulate() function with ``HIGH_RES`` to sum up the ``SQ_LEVEL_WAVES`` counter every clock cycle. This sum is then divided by the maximum value of ``GRBM_GUI_ACTIVE`` and the number of compute units ``CU_NUM`` to derive the mean occupancy.

Kernel serialization
---------------------

Counter collection in dispatch counting mode requires serialized execution of kernels on a target device. Kernel serialization isolates kernel executions, helping to collect performance counter data. However, for applications requiring two kernels to execute on the same device simultaneously (co-dependent kernels), kernel serialization leads to deadlock in dispatch counter collection mode. To avoid deadlock in such applications, opt for any of the following options:

- Avoid co-dependent kernels in application.

- Don't collect performance data for co-dependent kernels by using kernel filtration methods in the rocprofv3’s input configuration PMC file.

- Use ROCprofiler-SDK's device-wide counter collection mode to collect performance data. You can use tools such as RDC and PAPI to collect information. Note that the device-wide counter collection captures data for all executions on the device and not specific to the kernels.
