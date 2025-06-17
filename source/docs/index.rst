.. meta::
  :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
  :keywords: ROCprofiler-SDK tool, ROCprofiler-SDK library, rocprofv3, ROCprofiler-SDK API, ROCprofiler-SDK documentation

.. _index:

********************************
ROCprofiler-SDK documentation
********************************

ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software.
It supports application tracing to provide a big picture of the GPU application execution and kernel counter collection to provide low-level hardware details from the performance counters.
The ROCprofiler-SDK library provides runtime-independent APIs for tracing runtime calls and asynchronous activities such as GPU kernel dispatches and memory moves. The tracing includes callback APIs for runtime API tracing and activity APIs for asynchronous activity records logging.

In summary, ROCprofiler-SDK combines `ROCProfiler <https://rocm.docs.amd.com/projects/rocprofiler/en/latest/index.html>`_ and `ROCTracer <https://rocm.docs.amd.com/projects/roctracer/en/latest/index.html>`_.
You can utilize the ROCprofiler-SDK to develop a tool for profiling and tracing HIP applications on ROCm software.

The code is open and hosted at `<https://github.com/ROCm/rocprofiler-sdk>`_.


The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :ref:`installing-rocprofiler-sdk`

  .. grid-item-card:: How to

    * :doc:`Samples <how-to/samples>`
    * :ref:`using-rocprofv3`
    * :ref:`using-rocprofv3-avail`
    * :ref:`using-rocpd-output-format`
    * :ref:`using-rocprofiler-sdk-roctx`
    * :ref:`using-rocprofv3-with-mpi`
    * :ref:`using-pc-sampling`
    * :ref:`using-thread-trace`

  .. grid-item-card:: API reference

    * :doc:`Tool library <api-reference/tool_library>`
    * :ref:`runtime-intercept-tables`
    * :doc:`Buffered services <api-reference/buffered_services>`
    * :doc:`Callback services <api-reference/callback_services>`
    * :doc:`Counter collection services <api-reference/counter_collection_services>`
    * :doc:`PC sampling <api-reference/pc_sampling>`
    * :doc:`ROCprof Trace Decoder <api-reference/thread_trace>`
    * :doc:`ROCprofiler-SDK API <api-reference/rocprofiler-sdk_api_reference>`
    * :doc:`ROCTx API <api-reference/rocprofiler-sdk-roctx_api_reference>`

  .. grid-item-card:: Conceptual

    * :ref:`comparing-with-legacy-tools`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
