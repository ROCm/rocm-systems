
.. meta::
    :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
    :keywords: ROCprofiler-SDK API reference, ROCprofiler-SDK intercept table, Intercept table API

.. _runtime-intercept-tables:

Runtime intercept tables
=========================

While tools commonly leverage the callback or buffer tracing services for tracing the HIP, HSA, and ROCTx
APIs, ROCprofiler-SDK also provides access to the raw API dispatch tables.

Forward declaration of public C API function
----------------------------------------------

All the aforementioned APIs are designed similar to the following sample:

.. code-block:: cpp

    extern "C"
    {
    // forward declaration of public C API function
    int
    foo(int) __attribute__((visibility("default")));
    }

Internal implementation of API function
-----------------------------------------

.. code-block:: cpp

    namespace impl
    {
    int
    foo(int val)
    {
        // real implementation
        return (2 * val);
    }
    }

Dispatch table implementation
-------------------------------

.. code-block:: cpp

    namespace impl
    {
    struct dispatch_table
    {
        int (*foo_fn)(int) = nullptr;
    };

    // Invoked once: populates the dispatch_table with function pointers to implementation
    dispatch_table*&
    construct_dispatch_table()
    {
        static dispatch_table* tbl = new dispatch_table{};
        tbl->foo_fn                = impl::foo;

        // In between, ROCprofiler-SDK gets passed the pointer
        // to the dispatch table and has the opportunity to wrap the function
        // pointers for interception

        return tbl;
    }

    // Constructs dispatch table and stores it in static variable
    dispatch_table*
    get_dispatch_table()
    {
        static dispatch_table*& tbl = construct_dispatch_table();
        return tbl;
    }
    }  // namespace impl

Implementation of public C API function
-----------------------------------------

.. code-block:: cpp

    extern "C"
    {
    // implementation of public C API function
    int
    foo(int val)
    {
        return impl::get_dispatch_table()->foo_fn(val);
    }
    }

Dispatch table chaining
-------------------------

ROCprofiler-SDK can save the original values of the function pointers such as ``foo_fn`` in ``impl::construct_dispatch_table()`` and install its own function pointers in its place. This results in the public C API function ``foo`` calling into the ROCprofiler-SDK function pointer, which in turn, calls the original function pointer to ``impl::foo``. This phenomenon is named chaining. Once ROCprofiler-SDK
makes necessary modifications to the dispatch table, tools requesting access to the raw dispatch table via ``rocprofiler_at_intercept_table_registration`` are provided the pointer to the dispatch table.

For examples on dispatch table chaining, see `samples/intercept_table <https://github.com/ROCm/rocprofiler-sdk/tree/amd-staging/samples/intercept_table>`_.
