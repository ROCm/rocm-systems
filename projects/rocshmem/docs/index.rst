.. meta::
  :description: rocSHMEM is a runtime that provides GPU-centric networking through an OpenSHMEM-like interface.
  :keywords: rocSHMEM, ROCm, OpenSHMEM, library, API, IPC, RO

****************************
rocSHMEM documentation
****************************

The ROCm OpenSHMEM (rocSHMEM) is an intra-kernel networking library that provides GPU-centric networking through an OpenSHMEM-like interface. It simplifies application code complexity and enables finer communication and computation overlap than traditional host-driven networking. rocSHMEM uses a single symmetric heap allocated to GPU memories. For more information, see :doc:`introduction`.

The rocSHMEM public repository is located within the ROCm Systems Super Repo at `<https://github.com/ROCm/rocm-systems/tree/develop/projects/rocshmem>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

      * :doc:`Install rocSHMEM <./install>`

  .. grid-item-card:: How to

      * :doc:`Compile and run applications <./compile_and_run>`

  .. grid-item-card:: API reference

      * :doc:`Library setup, exit, and query routines <./api/init>`
      * :doc:`Memory management routines <./api/memory_management>`
      * :doc:`Team management routines <./api/teams>`
      * :doc:`Context management routines <./api/ctx>`
      * :doc:`Environment variables <./api/env_variables>`
      * :doc:`Remote memory access routines <./api/rma>`
      * :doc:`Atomic memory operations <./api/amo>`
      * :doc:`Signaling operations <./api/sigops>`
      * :doc:`Collective routines <./api/coll>`
      * :doc:`Point-to-point synchronization routines <./api/pt2pt_sync>`
      * :doc:`Memory ordering routines <./api/memory_ordering>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
