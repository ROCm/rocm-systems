.. meta::
  :description: The Component Unified ID (CUID) generates a deterministic unique ID for various devices such as GPUs, CPUs, NICs, and platforms in a data center environment.
  :keywords: CUID docs, Data center standardization, GPU index, Data center device management, Component Unified ID

.. _cuid-index:

*********************
CUID documentation
*********************

In a data center environment, a variety of devices, including GPUs, CPUs, and NICs are managed using different tools. Each of these tools uses distinct methods to identify devices, complicating correlation efforts. For example, the :doc:`AMD SMI <amdsmi:index>` tool uses BDF as the GPU index, while :doc:`ROCm Validation Suite <rocmvalidationsuite:index>` utilizes the ``gpu_id`` from KFD as the GPU index. This inconsistency makes device management and standardization particularly challenging, especially when external vendors are involved.

The Component Unified ID (CUID) library solves this problem by giving each component (the platform, AMD and Intel CPU packages, AMD GPUs and GPU partitions, AMD NPUs and network PCI functions) a deterministic ID. CUIDs are formatted as UUIDv8 values. This release produces temporary CUIDs: an HMAC (Keyed-Hash Message Authentication Code) of the host's machine-id and the component's PCIe address, IDs and UnitID. They read no hardware serial, are the same for every caller, and survive reboots and driver upgrades, but change when the machine-id changes, the component moves to another slot, or a GPU's compute partition mode changes. To learn more about CUID, see :ref:`what-is-cuid`.

Administrators set the node key and list CUIDs with amd-smi; programs link the static library.

The code is open and hosted at `<https://github.com/ROCm/rocm-systems/blob/develop/shared/cuid>`_.

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :ref:`Building-cuid`

  .. grid-item-card:: How to

    * :ref:`manage-node-key`
    * `Sample program using CUID API <https://github.com/ROCm/rocm-systems/blob/develop/shared/cuid/example/main.cc>`_

  .. grid-item-card:: API reference

    * :doc:`CUID API library <reference/amdcuid-c-api>`

  .. grid-item-card:: Conceptual

    * :ref:`what-is-cuid`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
