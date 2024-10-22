# Buffered services

For the buffered approach, supported buffer record categories are enumerated in `rocprofiler_buffer_category_t` category field.

## Overview

In buffered approach, callbacks are received for batches of records from an internal (background) thread.
Supported buffered tracing services are enumerated in  `rocprofiler_buffer_tracing_kind_t`. Configuring
a buffer tracing service requires the creation of a buffer. When the buffer is "flushed", either implicitly
or explicitly, a callback to the tool will be invoked which provides an array of one or more buffer records.
A buffer can be explicitly flushed via the `rocprofiler_flush_buffer` function.

## Subscribing to Buffer Tracing Services

During tool initialization, tools configure callback tracing via the `rocprofiler_configure_buffer_tracing_service`
function. However, before invoking `rocprofiler_configure_buffer_tracing_service`, the tool must create a buffer
for the tracing records.

### Creating a Buffer

```cpp
rocprofiler_status_t
rocprofiler_create_buffer(rocprofiler_context_id_t        context,
                          size_t                          size,
                          size_t                          watermark,
                          rocprofiler_buffer_policy_t     policy,
                          rocprofiler_buffer_tracing_cb_t callback,
                          void*                           callback_data,
                          rocprofiler_buffer_id_t*        buffer_id);
```

The `size` parameter is the size of the buffer in bytes and will be rounded up to the nearest
memory page size (defined by `sysconf(_SC_PAGESIZE)`); the default memory page size on Linux
is 4096 bytes (4 KB).

The `watermark` parameter specifies the number of bytes at which
the buffer should be "flushed", i.e. when the records in the buffer should invoke the
`callback` parameter to deliver the records to the tool. For example, if a buffer has a size
of 4096 bytes and the watermark is set to 48 bytes, six 8-byte records can be placed in the
buffer before `callback` is invoked. However, every 64-byte record that is placed in the
buffer will trigger a flush. It is safe to set the `watermark` to any value between
zero and the buffer size.

The `policy` parameter specifies the behavior for when a record is larger than the
amount of free space in the current buffer. For example, if a buffer has a size of
4000 bytes with a watermark set to 4000 bytes and 3998 of the bytes in the buffer
have been populated with records, the `policy` dictates how to handle an incoming record >
2 bytes. The `ROCPROFILER_BUFFER_POLICY_DISCARD` policy dictates that all records greater
than should 2 bytes should be dropped until the tool _explicitly_ flushes the buffer via
a `rocprofiler_flush_buffer` function call whereas the `ROCPROFILER_BUFFER_POLICY_LOSSLESS`
policy dictates that the current buffer should be swapped out for an empty buffer and placed
in that new buffer and former (full) buffer should be _implicitly_ flushed.

The `callback` parameter is the function that rocprofiler-sdk should invoke when flushing
the buffer; the value of the `callback_data` parameter will be passed as one of the arguments
to the `callback` function.

The `buffer_id` parameter is an output parameter for the function call and will have a
non-zero handle field after successful buffer creation.

### Creating a Dedicated Thread for Buffer Callbacks

By default, all buffers will use the same (default) background thread created by rocprofiler-sdk to
invoke their callback. However, rocprofiler-sdk provides an interface for tools to specify the
creation of an additional background thread for one or more of their buffers.

Callback threads for buffers are created via the `rocprofiler_create_callback_thread` function:

```cpp
rocprofiler_status_t
rocprofiler_create_callback_thread(rocprofiler_callback_thread_t* cb_thread_id);
```

Buffers are assigned to that callback thread via the `rocprofiler_assign_callback_thread` function:

```cpp
rocprofiler_status_t
rocprofiler_assign_callback_thread(rocprofiler_buffer_id_t       buffer_id,
                                   rocprofiler_callback_thread_t cb_thread_id);
```

#### Buffer Callback Thread Creation and Assignment Example

```cpp
{
    // create a context
    auto context_id = rocprofiler_context_id_t{0};
    rocprofiler_create_context(&context_id);

    // create a buffer associated with the context
    auto buffer_id  = rocprofiler_buffer_id_t{};
    rocprofiler_create_buffer(context_id, ..., &buffer_id);

    // specify that a new callback thread should be created and provide
    // and assign the identifier for it to the "thr_id" variable
    auto thr_id = rocprofiler_callback_thread_t{};
    rocprofiler_create_callback_thread(&thr_id);

    // assign the buffer callback to be delivered on this thread
    rocprofiler_assign_callback_thread(buffer_id, thr_id);
}
```

### Configuring Buffer Tracing Services

```cpp
rocprofiler_status_t
rocprofiler_configure_buffer_tracing_service(rocprofiler_context_id_t          context_id,
                                             rocprofiler_buffer_tracing_kind_t kind,
                                             rocprofiler_tracing_operation_t*  operations,
                                             size_t                            operations_count,
                                             rocprofiler_buffer_id_t           buffer_id);
```

The `kind` parameter is a high-level specifier of which service to trace (also known as a "domain").
Domain examples include, but are not limited to, the HIP API, the HSA API, and kernel dispatches.
For each domain, there are (often) various "operations", which can be used to restrict the callbacks
to a subset within the domain. For domains which correspond to APIs, the "operations" are the functions
which compose the API. If all operations in a domain should be traced, the `operations` and `operations_count`
parameters can be set to `nullptr` and `0`, respectively. If the tracing domain should be restricted to a subset
of operations, the tool library should specify a C-array of type `rocprofiler_tracing_operation_t` and the
size of the array for the `operations` and `operations_count` parameter.

Similar to `rocprofiler_configure_callback_tracing_service`,
`rocprofiler_configure_buffer_tracing_service` will return an error if a buffer service for given context
and given domain is configured more than once.

#### Example

```cpp
{
    auto ctx = rocprofiler_context_id_t{};
    // ... creation of context, etc. ...

    // buffer parameters
    constexpr auto KB          = 1024;  // 1024 bytes
    constexpr auto buffer_size = 16 * KB;
    constexpr auto watermark   = 15 * KB;
    constexpr auto policy      = ROCPROFILER_BUFFER_POLICY_LOSSLESS;

    // buffer handle
    auto buffer_id = rocprofiler_buffer_id_t{};

    // create a buffer associated with the context
    rocprofiler_create_buffer(
        context_id, buffer_size, watermark, policy, callback_func, nullptr, &buffer_id);

    // configure HIP runtime API function records to be placed in buffer
    rocprofiler_configure_buffer_tracing_service(
        ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0, buffer_id);

    // configure kernel dispatch records to be placed in buffer
    // (more than one service can use the same buffer)
    rocprofiler_configure_buffer_tracing_service(
        ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, buffer_id);

    // ... etc. ...
}
```

## Buffer Tracing Callback Function

Rocprofiler-sdk buffer tracing callback functions have the signature:

```cpp
typedef void (*rocprofiler_buffer_tracing_cb_t)(rocprofiler_context_id_t      context,
                                                rocprofiler_buffer_id_t       buffer_id,
                                                rocprofiler_record_header_t** headers,
                                                size_t                        num_headers,
                                                void*                         data,
                                                uint64_t                      drop_count);
```

The `rocprofiler_record_header_t` data type provides three pieces of information:

1. Category (`rocprofiler_buffer_category_t`)
2. Kind
3. Payload

The category is used to distinguish the classification of the buffer record. For all
services configured via `rocprofiler_configure_buffer_tracing_service`, the category will
be equal to the value of `ROCPROFILER_BUFFER_CATEGORY_TRACING`. The meaning of the kind
field is dependent on the category but when the category is `ROCPROFILER_BUFFER_CATEGORY_TRACING`,
the kind value will be equivalent to the  is used
to distinguish the `rocprofiler_buffer_tracing_kind_t` value passed to
`rocprofiler_configure_buffer_tracing_service`, e.g. `ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH`.
Once the category and kind have been determined, the payload can be casted:

```cpp
{
    if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
        header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
    {
        auto* record =
            static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);

        // ... etc. ...
    }
}
```

### Buffer Tracing Callback Function Example

```cpp
void
buffer_callback_func(rocprofiler_context_id_t      context,
                     rocprofiler_buffer_id_t       buffer_id,
                     rocprofiler_record_header_t** headers,
                     size_t                        num_headers,
                     void*                         user_data,
                     uint64_t                      drop_count)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
           header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);

            // ... etc. ...
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);

            // ... etc. ...
        }
        else
        {
            throw std::runtime_error{"unhandled record header category + kind"};
        }
    }
}
```

## Buffer Tracing Record

Unlike callback tracing records, there is no common set of data for each buffer tracing record. However,
many buffer tracing records contain a `kind` field and an `operation` field.
The name of a tracing kind can be obtained via the `rocprofiler_query_buffer_tracing_kind_name` function.
The name of an operation specific to a tracing kind can be obtained via the `rocprofiler_query_buffer_tracing_kind_operation_name`
function. One can also iterate over all the buffer tracing kinds and operations for each tracing kind via the
`rocprofiler_iterate_buffer_tracing_kinds` and `rocprofiler_iterate_buffer_tracing_kind_operations` functions.

The buffer tracing record data types can be found in the `rocprofiler-sdk/buffer_tracing.h` header
(`source/include/rocprofiler-sdk/buffer_tracing.h` in the [rocprofiler-sdk GitHub repository](https://github.com/ROCm/rocprofiler-sdk)).
