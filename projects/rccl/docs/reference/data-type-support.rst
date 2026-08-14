.. meta::
   :description: Reference table of data types supported by RCCL, including float, bfloat16, and FP8, and their full or partial support status for collective operations.
   :keywords: RCCL, ROCm, data type support, precision, ncclDataType_t, float, bfloat16, FP8, fp8_e4m3, rccl_bfloat8, half, int32, uint8, collective operations

.. _rccl-data-type-support:

**********************
RCCL precision support
**********************

This topic lists the data type support for the RCCL library on AMD GPUs.

This page lists the data types supported by the library itself and does not
indicate hardware support. A type listed here is only usable if the GPU
architecture also supports it; otherwise it is unsupported. For data type support
across the other ROCm libraries and by GPU architecture, see the
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

.. _rccl-input-output-type-support:

Supported data types
====================

The following table lists the data types supported by RCCL. For the corresponding
``ncclDataType_t`` enumeration, see the
:ref:`Library specification <rccl-supported-data-types>`.

.. list-table::
    :header-rows: 1

    *
      - Icon
      - Definition
    *
      - ✅
      - Fully supported as both an input and output type.
    *
      - ⚠️
      - Partially supported as an input or output type.

Data types not listed in the table below are not supported.

.. datatemplate:yaml:: /data/reference/precision-support.yaml

    .. list-table::
        :header-rows: 1
        :widths: 70, 30

        *
            - Data type
            - Support
    {% for data_type in data.data_types %}
        *
            - {{ data_type.type }}
            - {{ data_type.support }}
    {% endfor %}
