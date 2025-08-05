
.. meta::
    :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
    :keywords: ROCprofiler-SDK API reference, Buffered services API

.. _buffered-services:

ROCprofiler-SDK buffered services
=================================

In the buffered approach, the internal (background) thread sends callbacks for batches of records.
Supported buffer record categories are enumerated in ``rocprofiler_buffer_category_t`` category field and supported buffer tracing services are enumerated in  ``rocprofiler_buffer_tracing_kind_t``. Configuring
a buffered tracing service requires buffer creation. Flushing the buffer implicitly or explicitly invokes a callback to the tool, which provides an array of one or more buffer records.
To flush a buffer explicitly, use ``rocprofiler_flush_buffer`` function.

Subscribing to buffer tracing services
--------------------------------------

During tool initialization, the tool configures callback tracing using ``rocprofiler_configure_buffer_tracing_service``
function. However, before invoking ``rocprofiler_configure_buffer_tracing_service``, the tool must create a buffer for the tracing records as shown in the following section.

Creating a buffer
-----------------

.. code-block:: cpp

    rocprofiler_status_t
    rocprofiler_create_buffer(rocprofiler_context_id_t        context,
                              size_t                          size,
                              size_t                          watermark,
                              rocprofiler_buffer_policy_t     policy,
                              rocprofiler_buffer_tracing_cb_t callback,
                              void*                           callback_data,
                              rocprofiler_buffer_id_t*        buffer_id);

Here are the parameters required to create a buffer:

- ``size``: Size of the buffer in bytes, which is rounded up to the nearest
  memory page size (defined by ``sysconf(_SC_PAGESIZE)``). The default memory page size on Linux
  is 4096 bytes (4 KB).

- ``watermark``: Specifies the number of bytes at which the buffer should be flushed. To flush the buffer, the records in the buffer must invoke the ``callback`` parameter to deliver the records to the tool. For example, for a buffer of size 4096 bytes with the watermark set to 48 bytes, six 8-byte records can be placed in the
  buffer before ``callback`` is invoked. However, every 64-byte record that is placed in the
  buffer will trigger a flush. It is safe to set the ``watermark`` to any value between
  zero and the buffer size.

- ``policy``: Specifies the behavior when a record is larger than the
  amount of free space in the current buffer. For example, for a buffer of size 4000 bytes with the watermark set to 4000 bytes and 3998 bytes populated with records, the ``policy`` dictates how to handle an incoming record greater than 2 bytes. If the environment variable ``ROCPROFILER_BUFFER_POLICY_DISCARD`` is enabled, all records greater than 2 bytes are dropped until the tool _explicitly_ flushes the buffer using ``rocprofiler_flush_buffer`` function call whereas, if the environment variable ``ROCPROFILER_BUFFER_POLICY_LOSSLESS`` is enabled, the current buffer is swapped out for an empty buffer and placed in the new buffer while the former (full) buffer is _implicitly_ flushed.

- ``callback``: Invoked to flush the buffer.

- ``callback_data``: Value passed as one of the arguments to the ``callback`` function.

- ``buffer_id``: Output parameter for the function call to contain a
  non-zero handle field after successful buffer creation.

Creating a dedicated thread for buffer callbacks
------------------------------------------------

By default, all buffers use the same (default) background thread created by ROCprofiler-SDK to
invoke their callback. However, ROCprofiler-SDK provides an interface to allow the tools to create an additional background thread for one or more of their buffers.

To create callback threads for buffers, use ``rocprofiler_create_callback_thread`` function:

.. code-block:: cpp

    rocprofiler_status_t
    rocprofiler_create_callback_thread(rocprofiler_callback_thread_t* cb_thread_id);

To assign buffers to that callback thread, use ``rocprofiler_assign_callback_thread`` function:

.. code-block:: cpp

    rocprofiler_status_t
    rocprofiler_assign_callback_thread(rocprofiler_buffer_id_t       buffer_id,
                                       rocprofiler_callback_thread_t cb_thread_id);

**Example:**

.. code-block:: cpp

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

Configuring buffer tracing services
-----------------------------------

To configure buffer tracing services, use:

.. code-block:: cpp

    rocprofiler_status_t
    rocprofiler_configure_buffer_tracing_service(rocprofiler_context_id_t          context_id,
                                                 rocprofiler_buffer_tracing_kind_t kind,
                                                 rocprofiler_tracing_operation_t*  operations,
                                                 size_t                            operations_count,
                                                 rocprofiler_buffer_id_t           buffer_id);

Here are the parameters required to configure buffer tracing services:

- ``kind``: A high-level specification of the services to be traced. This parameter is also known as "domain".
  Domain examples include, but not limited to, the HIP API, HSA API, and kernel dispatches.

- ``operations``: For each domain, there are often various ``operations`` that can be used to restrict the callbacks to a subset within the domain. For domains corresponding to APIs, the ``operations`` are the functions
  composing the API. To trace all operations in a domain, set the ``operations`` and ``operations_count``
  parameters to ``nullptr`` and ``0`` respectively. To restrict the tracing domain to a subset
  of operations, the tool library must specify a C-array of type ``rocprofiler_tracing_operation_t`` for ``operations`` and size of the array for the ``operations_count`` parameter.

Similar to the ``rocprofiler_configure_callback_tracing_service``,
``rocprofiler_configure_buffer_tracing_service`` returns an error if a buffer service for the specified context
and domain is configured more than once.

**Example:**

.. code-block:: cpp

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

Buffer tracing callback function
--------------------------------

Here is the buffer tracing callback function:

.. code-block:: cpp

    typedef void (*rocprofiler_buffer_tracing_cb_t)(rocprofiler_context_id_t      context,
                                                    rocprofiler_buffer_id_t       buffer_id,
                                                    rocprofiler_record_header_t** headers,
                                                    size_t                        num_headers,
                                                    void*                         data,
                                                    uint64_t                      drop_count);

The ``rocprofiler_record_header_t`` data type contains the following information:

- ``category`` (``rocprofiler_buffer_category_t``): The ``category`` is used to classify the buffer record. For all
  services configured via ``rocprofiler_configure_buffer_tracing_service``, the ``category`` is equal to the value of ``ROCPROFILER_BUFFER_CATEGORY_TRACING``. The other available categories are ``ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING`` and ``ROCPROFILER_BUFFER_CATEGORY_COUNTERS``.

- ``kind``: The ``kind`` field is dependent on the ``category``. For example, for ``category`` ``ROCPROFILER_BUFFER_CATEGORY_TRACING``, the value of ``kind`` depicts the tracing type such as HSA core API in ``ROCPROFILER_BUFFER_TRACING_HSA_CORE_API``.

- ``payload``: The ``payload`` is casted after the category and kind have been determined.

.. code-block:: cpp

    {
        if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
            header->kind == ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_hip_api_record_t*>(header->payload);

            // ... etc. ...
        }
    }

**Example:**

.. code-block:: cpp

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

Buffer tracing record
---------------------

Unlike callback tracing records, there is no common set of data for each buffer tracing record. However,
many buffer tracing records contain a ``kind`` and an ``operation`` field.
You can obtain the value for the ``kind`` of tracing using ``rocprofiler_query_buffer_tracing_kind_name`` function and the value for the ``operation`` specific to a tracing kind using the ``rocprofiler_query_buffer_tracing_kind_operation_name``
function. You can also iterate over all the buffer tracing ``kinds`` and ``operations`` for each tracing kind using the
``rocprofiler_iterate_buffer_tracing_kinds`` and ``rocprofiler_iterate_buffer_tracing_kind_operations`` functions.

The buffer tracing record data types are available in the ``rocprofiler-sdk/buffer_tracing.h`` header.
