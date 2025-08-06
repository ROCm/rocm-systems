.. meta::
   :description: ROCm Compute Profiler documentation and reference
   :keywords: Omniperf, ROCm, profiler, tool, Instinct, accelerator, AMD

***********************************
ROCm Compute Profiler documentation
***********************************

This documentation provides a comprehensive overview of the ROCm Compute
Profiler tool. In addition to a full deployment guide with installation
instructions, this documentation also explains the ideas motivating the design
behind the tool and its components.

If you're new to ROCm Compute Profiler, familiarize yourself with the tool by reviewing the
chapters that follow and gradually learn its more advanced features. To get
started, see :doc:`What is ROCm Compute Profiler? <what-is-rocprof-compute>`.

ROCm Compute Profiler is open source and hosted at `<https://github.com/ROCm/rocprofiler-compute>`__.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Installation and deployment <install/core-install>`
      * :doc:`Grafana server for ROCm Compute Profiler <install/grafana-setup>`

   .. grid-item::

Use the following topics to learn more about the advantages of ROCm Compute Profiler in your
development toolkit, how it aims to model performance, and how to use ROCm Compute Profiler
in practice.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: How to

      * :doc:`how-to/use`

      * :doc:`how-to/profile/mode`

      * :doc:`how-to/analyze/mode`

        * :doc:`how-to/analyze/cli`

        * :doc:`how-to/analyze/grafana-gui`

        * :doc:`how-to/analyze/standalone-gui`

        * :doc:`how-to/analyze/tui`

   .. grid-item-card:: Conceptual

      * :doc:`conceptual/performance-model`

        * :doc:`conceptual/compute-unit`

        * :doc:`conceptual/l2-cache`

        * :doc:`conceptual/shader-engine`

        * :doc:`conceptual/command-processor`

        * :doc:`conceptual/system-speed-of-light`

      * :doc:`conceptual/definitions`

        * :ref:`normalization-units`

   .. grid-item-card:: Tutorials

      * :doc:`tutorial/profiling-by-example`

      * :doc:`Learning resources <tutorial/learning-resources>`

   .. grid-item-card:: Reference

      * :doc:`reference/compatible-accelerators`

      * :doc:`reference/faq`

This project is proudly open source. For more details on how to contribute,
refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

Find ROCm licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
