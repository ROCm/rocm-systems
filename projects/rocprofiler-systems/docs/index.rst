.. meta::
   :description: ROCm Systems Profiler documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

***********************************
ROCm Systems Profiler documentation
***********************************

ROCm Systems Profiler is designed for the high-level profiling and comprehensive tracing
of applications running on the CPU or the CPU and GPU. It supports dynamic binary
instrumentation, call-stack sampling, and various other features for determining
which function and line number are currently executing. To learn more, see :doc:`what-is-rocprof-sys`

ROCm Systems Profiler is open source and hosted at `<https://github.com/ROCm/rocprofiler-systems>`__.
It is the successor to `<https://github.com/ROCm/omnitrace>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Quick start <./install/quick-start>`
    * :doc:`ROCm Systems Profiler installation <./install/install>`

Use the following topics to learn more about the advantages of ROCm Systems Profiler in application
profiling, how it supports performance analysis, and how to leverage its capabilities in practice:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: How to

    * :doc:`Configuring the environment <./how-to/configuring-validating-environment>`

      * :doc:`Configuring runtime options <./how-to/configuring-runtime-options>`

    * :doc:`Profiling <./how-to/general-tips-using-rocprof-sys>`

      * :doc:`Sampling the call stack <./how-to/sampling-call-stack>`
      * :doc:`Instrumenting and rewriting a binary application <./how-to/instrumenting-rewriting-binary-application>`
      * :doc:`Performing causal profiling <./how-to/performing-causal-profiling>`
      * :doc:`Profiling Python scripts <./how-to/profiling-python-scripts>`
      * :doc:`Network performance profiling <./how-to/nic-profiling>`
      * :doc:`VCN and JPEG sampling and tracing <./how-to/vcn-jpeg-sampling>`

    * :doc:`Understanding the output <./how-to/understanding-rocprof-sys-output>`
    * :doc:`Using the ROCm Systems Profiler API <./how-to/using-rocprof-sys-api>`

  .. grid-item-card:: Conceptual

    * :doc:`Data collection modes <./conceptual/data-collection-modes>`
    * :doc:`Features and use cases <./conceptual/rocprof-sys-feature-set>`

  .. grid-item-card:: Reference

    * :doc:`Development guide <./reference/development-guide>`
    * :doc:`Glossary <./reference/rocprof-sys-glossary>`
    * :doc:`API library <./doxygen/html/files>`
    * :doc:`Class member functions <./doxygen/html/functions>`
    * :doc:`Globals <./doxygen/html/globals>`
    * :doc:`Classes, structures, and interfaces <./doxygen/html/annotated>`

  .. grid-item-card:: Tutorials

    * `GitHub examples <https://github.com/ROCm/rocprofiler-systems/tree/amd-mainline/examples>`_
    * :doc:`Video tutorials <./tutorials/video-tutorials>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
