# Building Tool Library

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

## Rocprofiler and ROCm Runtimes Design

The ROCm runtimes are now designed to directly communicate with a new library called rocprofiler-register during their initialization. This library does cursory checks
for whether any tools have indicated they want rocprofiler support via detection of one or more instances of a symbol named `rocprofiler_configure` (which is provided by
the tool libraries) and/or the `ROCP_TOOL_LIBRARIES` environment variable. This design dramatically improves upon previous designs which relied solely on
a tool racing to set runtime-specific environment variables (e.g. `HSA_TOOLS_LIB`) before the runtime initialization.

## Tool Library Design

When a tool has `rocprofiler_configure` visible in its symbol table, rocprofiler will invoke this function and provide information regarding
the version of rocprofiler which invoking the function, how many tools have already been invoked, and a unique idenitifier for the tool. The tool
returns a pointer to a `rocprofiler_tool_configure_result_t` struct, which, if non-null, can provide rocprofiler with the function it should
call for tool initialization (i.e. the opportunity for context creation), a function is should call when rocprofiler is finalized, and a pointer
to any data that rocprofiler should provide back to the tool when it calls the initialization and finalization functions.

Rocprofiler provides a `rocprofiler/registration.h` header file which forward declares the `rocprofiler_configure` function with the necessary
compiler function attributes to ensure that the symbol is publicly visible.

```cpp
#include <rocprofiler-sdk/registration.h>

namespace
{
// saves the data provided to rocprofiler_configure
struct ToolData
{
    uint32_t                              version;
    const char*                           runtime_version;
    uint32_t                              priority;
    rocprofiler_client_id_t               client_id;
};

// tool initialization function
int
tool_init(rocprofiler_client_finalize_t fini_func,
          void* tool_data_v);

// tool finalization function
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
    // if not first tool to register, indicate tool doesn't want to do anything
    if(priority > 0) return nullptr;

    // (optional) provide a name for this tool to rocprofiler
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
```

## Tool Initialization

> ***NOTE: rocprofiler does NOT support calls to any of the runtime functions (HSA, HIP, etc.) during tool initialization.***
> ***Invoking any functions from the runtimes will result in a deadlock.***

For each tool which contains a `rocprofiler_configure` function and returns a non-null pointer to a `rocprofiler_tool_configure_result_t` struct,
rocprofiler will invoke the `initialize` callback after completing the scan for all `rocprofiler_configure` symbols. In other words, rocprofiler
collects all of the `rocprofiler_tool_configure_result_t` instances before invoking the `initialize` member of any of these instances.
When rocprofiler invokes this function in a tool, this is the opportunity to create contexts:

```cpp
#include <rocprofiler-sdk/rocprofiler.h>

namespace
{
int
tool_init(rocprofiler_client_finalize_t fini_func,
          void* data_v)
{
    // create a context
    auto ctx = rocprofiler_context_id_t{};
    rocprofiler_create_context(&ctx);

    // ... associate services with context ...

    // start the context (optional)
    rocprofiler_start_context(ctx);

    return 0;
}
}
```

Although not strictly necessary, it is recommended that tools store the context handle(s) to control the data collection of the services associated with the context.

## Tool Finalization

In the invocation of the user-provided `initialize` callback, rocprofiler will provide a function pointer of type `rocprofiler_client_finalize_t`.
This function pointer can be invoked by the tool to explicitly invoke the `finalize` callback from the `rocprofiler_tool_configure_result_t` instance:

```cpp
#include <rocprofiler-sdk/rocprofiler.h>

namespace
{
int
tool_init(rocprofiler_client_finalize_t fini_func,
          void* data_v)
{
    // ... see initialization section ...

    // function which finalizes tool after 10 seconds
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
```

Otherwise, rocprofiler will invoke the `finalize` callback via an `atexit` handler.

## Agent Information

## Contexts

## Configuring Services

## Synchronous Callbacks

## Asychronous Callbacks for Buffers

## Recommendations

## Full `rocprofiler_configure` Sample

All of the snippets from the previous sections have been combined here for convenience.

```cpp
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

    // save the finalizer function
    tool_data->finalizer = fini_func;

    // create a context
    auto ctx = rocprofiler_context_id_t{};
    rocprofiler_create_context(&ctx);

    // save your contexts
    tool_data->contexts.emplace_back(ctx);

    // associate code object tracing with this context
    rocprofiler_configure_callback_tracing_service(
        ctx,
        ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
        nullptr,
        0,
        tool_tracing_callback,
        tool_data);

    // ... associate services with contexts ...

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
    // if not first tool to register, indicate tool doesn't want to do anything
    if(priority > 0) return nullptr;

    // (optional) provide a name for this tool to rocprofiler
    client_id->name = "ExampleTool";

    // info provided back to tool_init and tool_fini
    auto* my_tool_data = new rocp_tool_data{ version,
                                             runtime_version,
                                             priority,
                                             client_id,
                                             nullptr };

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{ sizeof(rocprofiler_tool_configure_result_t),
                                             &tool_init,
                                             &tool_fini,
                                             my_tool_data };

    return &cfg;
}
```
