
.. meta::
    :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
    :keywords: ROCprofiler-SDK API reference, Tool library API

.. _tool-library:

ROCprofiler-SDK tool library
============================

The tool library utilizes APIs from ``rocprofiler-sdk`` and ``rocprofiler-register`` libraries for profiling and tracing HIP applications. This document provides information to help you design a tool by utilizing the ``rocprofiler-sdk`` and ``rocprofiler-register`` libraries efficiently. The command-line tool ``rocprofv3`` is also built on ``librocprofiler-sdk-tool.so.X.Y.Z``, which uses these libraries.

ROCm runtimes design
---------------------

The ROCm runtimes are designed to directly communicate with a helper library named ``rocprofiler-register`` during initialization. This library performs cursory checks to find if a tool requires ROCprofiler-SDK services. This detection is based on the presence of one or more instances of ``rocprofiler_configure`` in the tool or ``ROCP_TOOL_LIBRARIES`` environment variable. This design provides drastic improvement over previous designs, which relied solely on a tool racing to set runtime-specific environment variables like ``HSA_TOOLS_LIB`` before the runtime initialization.

Tool library design
--------------------

When ROCprofiler-SDK detects ``rocprofiler_configure`` in a tool's symbol table, ROCprofiler-SDK invokes ``rocprofiler_configure`` with parameters such as ROCprofiler-SDK version that invokes the function, number of tools already invoked, and a unique identifier for the tool. The tool returns a pointer to a ``rocprofiler_tool_configure_result_t`` struct, which, if non-null, provides ROCprofiler-SDK with:

- Function to be called for tool initialization, which is also the opportunity for context creation.
- Function to be called when ROCprofiler-SDK is finalized.
- A pointer to data to be provided to the tool when ROCprofiler-SDK calls the initialization and finalization functions.

ROCprofiler-SDK provides a ``rocprofiler-sdk/registration.h`` header file, which forward declares the ``rocprofiler_configure`` function with the necessary compiler function attributes to ensure that the ``rocprofiler_configure`` symbol is publicly visible.

.. code-block:: cpp

    #include <rocprofiler-sdk/registration.h>

    namespace
    {
    struct ToolData
    {
        uint32_t                              version;
        const char*                           runtime_version;
        uint32_t                              priority;
        rocprofiler_client_id_t               client_id;
    };

    int
    tool_init(rocprofiler_client_finalize_t fini_func,
              void* tool_data_v);

    void
    tool_fini(void* tool_data_v);
    }

    extern "C"
    {
    rocprofiler_tool_configure_result_t*
    rocprofiler_configure(uint32_t                 version,
                          const char*              runtime_version,
                          uint32_t                 priority,
                          rocprofiler_client_id_t* client_id)
    {
        //If not the first tool to register, indicate that the tool doesn't want to do anything
        if(priority > 0) return nullptr;

        // (optional) Provide a name for this tool to rocprofiler
        client_id->name = "ExampleTool";

        // (optional) create configure data
        static auto data = ToolData{ version,
                                 runtime_version,
                                 priority,
                                 client_id };

        // construct configure result
        static auto cfg =
            rocprofiler_tool_configure_result_t{ sizeof(rocprofiler_tool_configure_result_t),
                                             &tool_init,
                                             &tool_fini,
                                             static_cast<void*>(&data) };

        return &cfg;
    }


.. note::
    ROCprofiler-SDK does NOT support calls to any runtime function (HSA, HIP, and so on) during tool initialization.
    Invoking any functions from the runtimes results in a deadlock.

For each tool that contains a ``rocprofiler_configure`` function and returns a non-null pointer to a ``rocprofiler_tool_configure_result_t`` struct, ROCprofiler-SDK invokes the ``initialize`` callback after completing the scan for all ``rocprofiler_configure`` symbols. In other words, ROCprofiler-SDK
collects all ``rocprofiler_tool_configure_result_t`` instances before invoking the ``initialize`` member of any of these instances.
When ROCprofiler-SDK invokes ``initialize`` function in a tool, this is the opportunity to create contexts:

.. code-block:: cpp

    #include <rocprofiler-sdk/rocprofiler.h>

    namespace
    {
    int
    tool_init(rocprofiler_client_finalize_t fini_func,
              void* data_v)
    {
        // create a context
        auto ctx = rocprofiler_context_id_t{0};
        rocprofiler_create_context(&ctx);

        // ... associate services with context ...

        // start the context (optional)
        rocprofiler_start_context(ctx);

        return 0;
    }
    }

Although not mandatory, it is recommended that tools store the context handles to control the data collection for the services associated with the context.

Tool finalization
------------------

When the `initialize` callback is invoked in the tool, ROCprofiler-SDK provides a function pointer of type `rocprofiler_client_finalize_t`.
The tool can invoke this function pointer to explicitly invoke the `finalize` callback from the `rocprofiler_tool_configure_result_t` instance:

.. code-block:: cpp

    #include <rocprofiler-sdk/rocprofiler.h>

    namespace
    {
        int
        tool_init(rocprofiler_client_finalize_t fini_func,
                  void* data_v)
        {
            // ... see initialization section ...

            // function, which
            auto explicit_finalize = [](rocprofiler_client_finalize_t finalizer,
                                        rocprofiler_client_id_t* client_id)
            {
                std::this_thread::sleep_for(std::chrono::seconds{ 10 });
                finalizer(client_id);
            };

            // start the context
            rocprofiler_start_context(ctx);

            // dispatch a background thread to explicitly finalize after 10 seconds
            std::thread{ explicit_finalize, fini_func, static_cast<ToolData*>(data_v)->client_id }.detach();

            return 0;
        }
    }

Otherwise, ROCprofiler-SDK invokes the `finalize` callback via an `atexit` handler.

Full rocprofiler_configure sample
----------------------------------

All the code snippets from the previous sections are combined here to demonstrate complete ROCProfiler configuration.

.. code-block:: cpp

    #include <rocprofiler-sdk/registration.h>

    namespace
    {
    struct rocp_tool_data
    {
        uint32_t                              version;
        const char*                           runtime_version;
        uint32_t                              priority;
        rocprofiler_client_id_t               client_id;
        rocprofiler_client_finalize_t         finalizer;
        std::vector<rocprofiler_context_id_t> contexts;
    };

    void
    tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                        rocprofiler_user_data_t*              user_data,
                        void*                                 callback_data);

    int
    tool_init(rocprofiler_client_finalize_t fini_func,
            void* tool_data_v)
    {
        rocp_tool_data* tool_data = static_cast<rocp_tool_data*>(tool_data_v);

        // Save the finalizer function
        tool_data->finalizer = fini_func;

        // create a context
        auto ctx = rocprofiler_context_id_t{0};
        rocprofiler_create_context(&ctx);

        // Save your contexts
        tool_data->contexts.emplace_back(ctx);

        // Associate code object tracing with this context
        rocprofiler_configure_callback_tracing_service(
            ctx,
            ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
            nullptr,
            0,
            tool_tracing_callback,
            tool_data);

        // ... Associate services with contexts ...

        return 0;
    }

    void
    tool_fini(void* tool_data);
    }

    extern "C"
    {
    rocprofiler_tool_configure_result_t*
    rocprofiler_configure(uint32_t                 version,
                        const char*              runtime_version,
                        uint32_t                 priority,
                        rocprofiler_client_id_t* client_id)
    {
        // (optional) Provide a name for this tool to rocprofiler
        client_id->name = "ExampleTool";

        // Info provided back to tool_init and tool_fini
        auto* my_tool_data = new rocp_tool_data{ version,
                                                runtime_version,
                                                priority,
                                                client_id,
                                                nullptr };

        // Create configure data
        static auto cfg =
            rocprofiler_tool_configure_result_t{ sizeof(rocprofiler_tool_configure_result_t),
                                                &tool_init,
                                                &tool_fini,
                                                my_tool_data };

        return &cfg;
    }
