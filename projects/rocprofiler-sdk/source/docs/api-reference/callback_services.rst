
.. meta::
    :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
    :keywords: ROCprofiler-SDK API reference, ROCprofiler-SDK callback services, Callback services API

.. _rocprofiler_sdk_callback_tracing_services:

ROCprofiler-SDK callback tracing services
=========================================

Callback tracing services provide immediate callbacks to a tool on the current CPU thread on the occurrence of an event.
For example, when tracing an API function such as ``hipSetDevice``, callback tracing invokes a user-specified callback
before and after the traced function executes on the thread invoking the API function.

Subscribing to callback tracing services
----------------------------------------

During tool initialization, tools configure callback tracing using:

.. code-block:: cpp

    rocprofiler_status_t
    rocprofiler_configure_callback_tracing_service(rocprofiler_context_id_t            context_id,
                                                   rocprofiler_callback_tracing_kind_t kind,
                                                   rocprofiler_tracing_operation_t*    operations,
                                                   size_t                              operations_count,
                                                   rocprofiler_callback_tracing_cb_t   callback,
                                                   void*                               callback_args);

Here are the parameters required to configure callback tracing services:

- ``kind``: A high-level specification of the services to be traced. This parameter is also known as "domain".
  Domain examples include, but not limited to, the HIP API, HSA API, and kernel dispatches.

- ``operations``: For each domain, there are often various ``operations`` that can be used to restrict the callbacks to a subset within the domain. For domains corresponding to APIs, the ``operations`` are the functions
  composing the API. To trace all operations in a domain, set the ``operations`` and ``operations_count``
  parameters to ``nullptr`` and ``0`` respectively. To restrict the tracing domain to a subset
  of operations, the tool library must specify a C-array of type ``rocprofiler_tracing_operation_t`` for ``operations`` and size of the array for the ``operations_count`` parameter.

``rocprofiler_configure_callback_tracing_service`` returns an error if a callback service for the specified context and domain is configured more than once.

**Example:** To trace only two functions within
the HIP runtime API, ``hipGetDevice`` and ``hipSetDevice``:

.. code-block:: cpp

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

The following code returns error ``ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED`` as the callback service is already configured:

.. code-block:: cpp

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
            // after the first iteration, returns ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED
            rocprofiler_configure_callback_tracing_service(ctx,
                                                           ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                           &op,
                                                           1,
                                                           callback_func,
                                                           nullptr);
        }

        // ... etc. ...
    }

Callback tracing callback function
----------------------------------

Here is the callback tracing callback function:

.. code-block:: cpp

    typedef void (*rocprofiler_callback_tracing_cb_t)(rocprofiler_callback_tracing_record_t record,
                                                      rocprofiler_user_data_t*              user_data,
                                                      void* callback_data)

The parameters ``record`` and ``user_data`` are discussed here:

- ``record``: Contains the information to uniquely identify a tracing record type. Here is the definition:

  .. code-block:: cpp

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

  The underlying type of ``payload`` field is typically unique to a domain and, less frequently, an operation.
  For example, for the ``ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API`` and ``ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API``,
  the payload must be casted to ``rocprofiler_callback_tracing_hip_api_data_t*``, which contains the arguments
  to the function and the return value when exiting the function. The payload field is a valid
  pointer only during the invocation of the callback function(s).

- ``user_data``: Stores data in between callback phases. This value is unique for every
  instance of an operation. For example, for a tool library to store the timestamp of the
  ``ROCPROFILER_CALLBACK_PHASE_ENTER`` phase for the ensuing ``ROCPROFILER_CALLBACK_PHASE_EXIT`` callback,
  the data can be stored using:

  .. code-block:: cpp

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

  The `callback_data` is passed to `rocprofiler_configure_callback_tracing_service` as the value of `callback_args` to :ref:`subscribe to callback tracing services <rocprofiler_sdk_callback_tracing_services>`.

Callback tracing record
-----------------------

To obtain the name of the ``kind`` of tracing, you can use ``rocprofiler_query_callback_tracing_kind_name`` function and to obtain the name of an ``operation`` specific to a tracing kind, use ``rocprofiler_query_callback_tracing_kind_operation_name``
function. To iterate over all the callback tracing kinds and operations for each tracing kind, use ``rocprofiler_iterate_callback_tracing_kinds`` and ``rocprofiler_iterate_callback_tracing_kind_operations`` functions.

Lastly, for a specified ``rocprofiler_callback_tracing_record_t`` object, ROCprofiler-SDK supports generically iterating over the arguments of the payload field for many domains. Within the ``rocprofiler_callback_tracing_record_t`` object, the domain-specific information is available in
an opaque ``void* payload``.
The data types generally follow the naming convention of ``rocprofiler_callback_tracing_<DOMAIN>_data_t``. For example, for the tracing kinds ``ROCPROFILER_BUFFER_TRACING_HSA_{CORE,AMD_EXT,IMAGE_EXT,FINALIZE_EXT}_API``,
cast the payload to ``rocprofiler_callback_tracing_hsa_api_data_t*``:

.. code-block:: cpp

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

**Example:** Iterating over all the callback tracing kinds and operations for each tracing kind using ``rocprofiler_iterate_callback_tracing_kind_operation_args``:

.. code-block:: cpp

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

**Sample output:**

.. code-block:: console

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

Code object tracing
-------------------

The code object tracing service is a critical component for obtaining information regarding
asynchronous activity on the GPU. The ``rocprofiler_callback_tracing_code_object_load_data_t``
payload (kind=``ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT``, operation=``ROCPROFILER_CODE_OBJECT_LOAD``)
provides a unique identifier for a bundle of one or more GPU kernel symbols that are loaded
for a specific GPU agent. For example, if your application leverages a multi-GPU system
consisting of four Vega20 GPUs and four MI100 GPUs, at least eight code objects will be loaded: one code
object for each GPU. Each code object will be associated with a set of kernel symbols.
The ``rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t`` payload
(kind=``ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT``, operation=``ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER``)
provides a globally unique identifier for the specific kernel symbol along with the kernel name and
several other static properties of the kernel such as scratch size, scalar general purpose register count, and so on.

.. note::
    The kernel identifiers for two identical kernel symbols with the same properties (kernel name, scratch size, and so on) that are part of similar code objects loaded for different GPU agents will still be unique. Furthermore, the identifier for a code object and its kernel symbols after being unloaded and then
    reloaded, will also be unique.

Here is the general sequence of events when a code object is loaded and unloaded:

1. Callback: load code object
   - kind= ``ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT``
   - operation= ``ROCPROFILER_CODE_OBJECT_LOAD``
   - phase= ``ROCPROFILER_CALLBACK_PHASE_LOAD``
2. Callback: load kernel symbol
   - kind= ``ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT``
   - operation= ``ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER``
   - phase= ``ROCPROFILER_CALLBACK_PHASE_LOAD``
   - Repeats for each kernel symbol in code object
3. Execute application
4. Callback: unload kernel symbol
   - kind= ``ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT``
   - operation= ``ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER``
   - phase= ``ROCPROFILER_CALLBACK_PHASE_UNLOAD``
   - Repeats for each kernel symbol in code object
5. Callback: unload code object
   - kind= ``ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT``
   - operation= ``ROCPROFILER_CODE_OBJECT_LOAD``
   - phase= ``ROCPROFILER_CALLBACK_PHASE_UNLOAD``

.. note::
    ROCprofiler-SDK doesn't provide an interface to query information outside of the
    code object tracing service. If you wish to associate kernel names with kernel tracing records,
    the tool must be configured to create a copy of the relevant information when the code objects and
    kernel symbol are loaded. However, any constant string fields like ``const char* kernel_name``
    don't need to be copied as these are guaranteed to be valid pointers until after ROCprofiler-SDK finalization.
    If a tool decides to delete its copy of the data associated with a code object or kernel symbol
    identifier when the code object and kernel symbols are unloaded, it is highly recommended to flush
    all buffers that might contain references to that code object or kernel symbol identifier before
    deleting the associated data.

For a sample of code object tracing, see `samples/code_object_tracing <https://github.com/ROCm/rocprofiler-sdk/tree/amd-mainline/samples/code_object_tracing>`_.
