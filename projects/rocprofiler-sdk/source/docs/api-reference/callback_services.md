# Callback tracing services

## Overview

Callback tracing services provide immediate callbacks to a tool on the current CPU thread when a given event occurs.
For example, when tracing an API function, e.g. `hipSetDevice`, callback tracing invokes a user-specified callback
before and after the traced function executes on the thread which is invoking the API function.

## Subscribing to Callback Tracing Services

During tool initialization, tools configure callback tracing via the `rocprofiler_configure_callback_tracing_service`
function:

```cpp
rocprofiler_status_t
rocprofiler_configure_callback_tracing_service(rocprofiler_context_id_t            context_id,
                                               rocprofiler_callback_tracing_kind_t kind,
                                               rocprofiler_tracing_operation_t*    operations,
                                               size_t                              operations_count,
                                               rocprofiler_callback_tracing_cb_t   callback,
                                               void*                               callback_args);
```

The `kind` parameter is a high-level specifier of which service to trace (also known as a "domain").
Domain examples include, but are not limited to, the HIP API, the HSA API, and kernel dispatches.
For each domain, there are (often) various "operations", which can be used to restrict the callbacks
to a subset within the domain. For domains which correspond to APIs, the "operations" are the functions
which compose the API. If all operations in a domain should be traced, the `operations` and `operations_count`
parameters can be set to `nullptr` and `0`, respectively. If the tracing domain should be restricted to a subset
of operations, the tool library should specify a C-array of type `rocprofiler_tracing_operation_t` and the
size of the array for the `operations` and `operations_count` parameter.

`rocprofiler_configure_callback_tracing_service` will return an error if a callback service for given context
and given domain is configured more than once. For example, if one only wanted to trace two functions within
the HIP runtime API, `hipGetDevice` and `hipSetDevice`, the following code would accomplish this objective:

```cpp
{
    auto ctx = rocprofiler_context_id_t{};
    // ... creation of context, etc. ...

    // array of operations (i.e. API functions)
    auto operations = std::array<rocprofiler_tracing_operation_t, 2>{
        ROCPROFILER_HIP_RUNTIME_API_ID_hipSetDevice,
        ROCPROFILER_HIP_RUNTIME_API_ID_hipGetDevice
    };

    rocprofiler_configure_callback_tracing_service(ctx,
                                                   ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                   operations.data(),
                                                   operations.size(),
                                                   callback_func,
                                                   nullptr);
    // ... etc. ...
}
```

But the following code would be invalid:

```cpp
{
    auto ctx = rocprofiler_context_id_t{};
    // ... creation of context, etc. ...

    // array of operations (i.e. API functions)
    auto operations = std::array<rocprofiler_tracing_operation_t, 2>{
        ROCPROFILER_HIP_RUNTIME_API_ID_hipSetDevice,
        ROCPROFILER_HIP_RUNTIME_API_ID_hipGetDevice
    };

    for(auto op : operations)
    {
        // after the first iteration, will return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED
        rocprofiler_configure_callback_tracing_service(ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                       &op,
                                                       1,
                                                       callback_func,
                                                       nullptr);
    }

    // ... etc. ...
}
```

## Callback Tracing Callback Function

Rocprofiler-sdk callback tracing callback functions have the signature:

```cpp
typedef void (*rocprofiler_callback_tracing_cb_t)(rocprofiler_callback_tracing_record_t record,
                                                  rocprofiler_user_data_t*              user_data,
                                                  void* callback_data)
```

The `record` parameter contains the information to uniquely identify a tracing record type and has the
following definition:

```cpp
typedef struct rocprofiler_callback_tracing_record_t
{
    rocprofiler_context_id_t            context_id;
    rocprofiler_thread_id_t             thread_id;
    rocprofiler_correlation_id_t        correlation_id;
    rocprofiler_callback_tracing_kind_t kind;
    uint32_t                            operation;
    rocprofiler_callback_phase_t        phase;
    void*                               payload;
} rocprofiler_callback_tracing_record_t;
```

The underlying type of `payload` field above is typically unique to a domain and, less frequently, an operation.
For example, for the `ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API` and `ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API`,
the payload should be casted to `rocprofiler_callback_tracing_hip_api_data_t*` -- which will contain the arguments
to the function and (in the exit phase) the return value of the function. The payload field will only be a valid
pointer during the invocation of the callback function(s).

The `user_data` parameter can be used to store data in between callback phases. It is a unique for every
instance of an operation. For example, if the tool library wishes to store the timestamp of the
`ROCPROFILER_CALLBACK_PHASE_ENTER` phase for the ensuing `ROCPROFILER_CALLBACK_PHASE_EXIT` callback,
this data can be stored in a method similar to below:

```cpp
void
callback_func(rocprofiler_callback_tracing_record_t record,
              rocprofiler_user_data_t*              user_data,
              void*                                 cb_data)
{
    auto ts = rocprofiler_timestamp_t{};
    rocprofiler_get_timestamp(&ts);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        user_data->value = ts;
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        auto delta_ts = (ts - user_data->value);
        // ... etc. ...
    }
    else
    {
        // ... etc. ...
    }
}
```

The `callback_data` argument will be the value of `callback_args` passed to `rocprofiler_configure_callback_tracing_service`
in [the previous section](#subscribing-to-callback-tracing-services).

## Callback Tracing Record

The name of a tracing kind can be obtained via the `rocprofiler_query_callback_tracing_kind_name` function.
The name of an operation specific to a tracing kind can be obtained via the `rocprofiler_query_callback_tracing_kind_operation_name`
function. One can also iterate over all the callback tracing kinds and operations for each tracing kind via the
`rocprofiler_iterate_callback_tracing_kinds` and `rocprofiler_iterate_callback_tracing_kind_operations` functions.
Lastly, for a given `rocprofiler_callback_tracing_record_t` object, rocprofiler-sdk supports generically iterating over
the arguments of the payload field for many domains.

As mentioned above, within the `rocprofiler_callback_tracing_record_t` object,
an opaque `void* payload` is provided for accessing domain specific information.
The data types generally follow the naming convention of `rocprofiler_callback_tracing_<DOMAIN>_data_t`,
e.g., for the tracing kinds `ROCPROFILER_BUFFER_TRACING_HSA_{CORE,AMD_EXT,IMAGE_EXT,FINALIZE_EXT}_API`,
the payload should be casted to `rocprofiler_callback_tracing_hsa_api_data_t*`:

```cpp
void
callback_func(rocprofiler_callback_tracing_record_t record,
              rocprofiler_user_data_t*              user_data,
              void*                                 cb_data)
{
    static auto hsa_domains = std::unordered_set<rocprofiler_buffer_tracing_kind_t>{
        ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
        ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
        ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
        ROCPROFILER_BUFFER_TRACING_HSA_FINALIZER_API};

    if(hsa_domains.count(record.kind) > 0)
    {
        auto* payload = static_cast<rocprofiler_callback_tracing_hsa_api_data_t*>(record.payload);

        hsa_status_t status = payload->retval.hsa_status_t_retval;
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT && status != HSA_STATUS_SUCCESS)
        {
            const char* _kind = nullptr;
            const char* _operation = nullptr;

            rocprofiler_query_callback_tracing_kind_name(record.kind, &_kind, nullptr);
            rocprofiler_query_callback_tracing_kind_operation_name(
                record.kind, record.operation, &_operation, nullptr);

            // message that
            fprintf(stderr, "[domain=%s] %s returned a non-zero exit code: %i\n", _kind, _operation, status);
        }
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
    {
        auto delta_ts = (ts - user_data->value);
        // ... etc. ...
    }
    else
    {
        // ... etc. ...
    }
}
```

### Sample `rocprofiler_iterate_callback_tracing_kind_operation_args`

```cpp
int
print_args(rocprofiler_callback_tracing_kind_t domain_idx,
           uint32_t                            op_idx,
           uint32_t                            arg_num,
           const void* const                   arg_value_addr,
           int32_t                             arg_indirection_count,
           const char*                         arg_type,
           const char*                         arg_name,
           const char*                         arg_value_str,
           int32_t                             arg_dereference_count,
           void*                               data)
{
    if(arg_num == 0)
    {
        const char* _kind      = nullptr;
        const char* _operation = nullptr;

        rocprofiler_query_callback_tracing_kind_name(domain_idx, &_kind, nullptr);
        rocprofiler_query_callback_tracing_kind_operation_name(
            domain_idx, op_idx, &_operation, nullptr);

        fprintf(stderr, "\n[%s] %s\n", _kind, _operation);
    }

    char* _arg_type = abi::__cxa_demangle(arg_type, nullptr, nullptr, nullptr);

    fprintf(stderr, "    %u: %-18s %-16s = %s\n", arg_num, _arg_type, arg_name, arg_value_str);

    free(_arg_type);

    // unused in example
    (void) arg_value_addr;
    (void) arg_indirection_count;
    (void) arg_dereference_count;
    (void) data;

    return 0;
}

void
callback_func(rocprofiler_callback_tracing_record_t record,
              rocprofiler_user_data_t*              user_data,
              void*                                 cb_data)
{
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
       record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API &&
       (record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel ||
        record.operation == ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync))
    {
        rocprofiler_iterate_callback_tracing_kind_operation_args(
                             record, print_args, record.phase, nullptr));
    }
}
```

Sample Output:

```console

[HIP_RUNTIME_API] hipLaunchKernel
    0: void const*        function_address = 0x219308
    1: rocprofiler_dim3_t numBlocks        = {z=1, y=310, x=310}
    2: rocprofiler_dim3_t dimBlocks        = {z=1, y=32, x=32}
    3: void**             args             = 0x7ffe6d8dd3c0
    4: unsigned long      sharedMemBytes   = 0
    5: hipStream_t*      stream           = 0x17b40c0

[HIP_RUNTIME_API] hipMemcpyAsync
    0: void*              dst              = 0x7f06c7bbb010
    1: void const*        src              = 0x7f0698800000
    2: unsigned long      sizeBytes        = 393625600
    3: hipMemcpyKind      kind             = DeviceToHost
    4: hipStream_t*      stream           = 0x25dfcf0
```

## Code Object Tracing

The code object tracing service is a critical component for obtaining information regarding
asynchronous activity on the GPU. The `rocprofiler_callback_tracing_code_object_load_data_t`
payload (kind=`ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`, operation=`ROCPROFILER_CODE_OBJECT_LOAD`)
provides a unique identifier for a bundle of one or more GPU kernel symbols which have been loaded
for a specific GPU agent. For example, if your application is leveraging a multi-GPU system system
containing 4 Vega20 GPUs and 4 MI100 GPUs, there will at least 8 code objects loaded: one code
object for each GPU. Each code object will be associated with a set of kernel symbols:
the `rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t` payload
(kind=`ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`, operation=`ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER`)
provides a globally unique identifier for the specific kernel symbol along with the kernel name and
several other static properties of the kernel (e.g. scratch size, scalar general purpose register count, etc.).
Note: two otherwise identical kernel symbols (same kernel name, scratch size, etc.) which are part of
otherwise identical code objects but the code objects are loaded for different GPU agents ***will*** have unique
kernel identifiers. Furthermore, if the same code object (and it's kernel symbols) are unloaded and then
re-loaded, that code object and all of it's kernel symbols ***will*** be given new unique identifiers.

In general, when a code object is loaded and unloaded, here is the sequence of events:

1. Callback: code object load
    - kind=`ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`
    - operation=`ROCPROFILER_CODE_OBJECT_LOAD`
    - phase=`ROCPROFILER_CALLBACK_PHASE_LOAD`
2. Callback: kernel symbol load
    - kind=`ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`
    - operation=`ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER`
    - phase=`ROCPROFILER_CALLBACK_PHASE_LOAD`
    - Repeats for each kernel symbol in code object
3. Application Execution
4. Callback: kernel symbol unload
    - kind=`ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`
    - operation=`ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER`
    - phase=`ROCPROFILER_CALLBACK_PHASE_UNLOAD`
    - Repeats for each kernel symbol in code object
5. Callback: code object unload
    - kind=`ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT`
    - operation=`ROCPROFILER_CODE_OBJECT_LOAD`
    - phase=`ROCPROFILER_CALLBACK_PHASE_UNLOAD`

Note: rocprofiler-sdk does not provide an interface to query this information outside of the
code object tracing service. If you wish to be able to associate kernel names with kernel tracing records,
a tool is personally responsible for making a copy of the relevant information when the code objects and
kernel symbol are loaded (however, any constant string fields like the (`const char* kernel_name` field)
need not to be copied, these are guaranteed to be valid pointers until after rocprofiler-sdk finalization).
If a tool decides to delete their copy of the data associated with a given code object or kernel symbol
identifier when the code object and kernel symbols are unloaded, it is highly recommended to flush
any/all buffers which might contain references to that code object or kernel symbol identifiers before
deleting the associated data.

For a sample of code object tracing, please see the `samples/code_object_tracing` example in the
[rocprofiler-sdk GitHub repository](https://github.com/ROCm/rocprofiler-sdk).
