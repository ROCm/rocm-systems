.. meta::
  :description: The Component Unified ID (CUID) generates a deterministic unique ID for various devices such as GPUs, CPUs, NICs, and platforms in a data center environment.
  :keywords: CUID docs, Data center standardization, GPU index, Data center device management, CUUID

.. _cuid-index:

*********************
CUID documentation
*********************

In a data center environment, a variety of devices, including GPUs, CPUs, and NICs are managed using different tools. Each of these tools uses distinct methods to identify devices, complicating correlation efforts. For example, the :doc:`AMD SMI <amdsmi:index>` tool uses BDF as the GPU index, while :doc:`ROCm Validation Suite <rocmvalidationsuite:index>` utilizes the ``gpu_id`` from KFD as the GPU index. This inconsistency makes device management and standardization particularly challenging, especially when external vendors are involved.

The Component Unified Identifier (CUID or CUUID) library solves this problem by generating a stable, unique ID (CUID) for various AMD hardware devices including GPUs, CPUs, NICs, and platform devices in a deterministic manner. This unique ID acts as a bridge, facilitating seamless integration of different descriptions across the ecosystem. CUIDs are formatted as UUIDv8 values and derived from hardware fingerprints using Keyed-Hash Message Authentication Code (HMAC), enabling consistent device identification across reboots and driver upgrades. To learn more about CUID, see :ref:`what-is-cuid`

The CUID library also provides a command-line (CLI) tool to detect devices and generate CUIDs. The CLI tool can help administrators simplify device management and integration.

The code is open and hosted at `<https://github.com/ROCm/rocm-systems/blob/develop/projects/cuid>`_.

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :ref:`Building-cuid`

  .. grid-item-card:: How to

    * :ref:`cuid-cli-tool`
    * `Sample program using CUID API <https://github.com/ROCm/rocm-systems/blob/develop/projects/cuid/example/main.cc>`_

  .. grid-item-card:: API reference

    * :doc:`CUID API library <reference/amdcuid-c-api>`

  .. grid-item-card:: Conceptual

    * :ref:`cuid-file`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
