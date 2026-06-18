.. meta::
  :description: AMD Infinity Storage library that supports IO directly to the GPU
  :keywords: hipFile, ROCm, storage, library, API, DMA

*********************
hipFile documentation
*********************

hipFile is an AMD Infinity Storage library that supports IO directly to the GPU. This documentation covers how to build and use hipFile, and provides a reference for its public APIs. For more information, see :doc:`introduction`.

hipFile is a part of the rocm-systems repository. The code can be found at `<https://github.com/ROCm/rocm-systems/tree/develop/projects/hipfile>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Build and install

      * :doc:`Configure and build <./config_build>`
      * :doc:`Sanitizers <./sanitizers>`

  .. grid-item-card:: Using hipFile

      * :doc:`hipify <./hipify>`
      * :doc:`NVIDIA compatibility <./nvidia_compat>`
      * :doc:`fio <./fio>`
      * :doc:`Collecting statistics <./stats_collection>`

  .. grid-item-card:: API reference

      * :doc:`Errors <./api/errors>`
      * :doc:`Core API <./api/core>`
      * :doc:`Drivers <./api/driver>`
      * :doc:`File operations <./api/file>`
      * :doc:`Asynchronous operations <./api/async>`
      * :doc:`Batch operations <./api/batch>`
      * :doc:`RDMA <./api/rdma>`
      * :doc:`Other <./api/misc_api>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
