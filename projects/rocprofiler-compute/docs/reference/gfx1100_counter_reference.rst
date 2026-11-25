.. Copyright Advanced Micro Devices, Inc., or its affiliates.
.. SPDX-License-Identifier: MIT

gfx1100 counter coverage
================================

The Compute SDK override restores 84 historical gfx1100 definitions: 41 raw
selectors and 43 derived metrics or constants. The snapshot is the SDK
configuration immediately before the gfx1250 architecture cleanup in #9541.
The historical Compute configuration contained 65 of these names; the
remaining 19 were already present in that same SDK snapshot.

Compute writes its own configuration to ``ROCPROFILER_METRICS_PATH``. An
installed SDK definition cannot fill a missing entry in an existing override.
This change restores the configuration contract. Default gfx1100 SoC routing
requires separate enablement.

Selector reference
------------------

The gfx11 definition layer includes ``soc21_enum.h``. All 41 raw selector
values match that catalog, including the explicit TA name mappings.
The catalog's wave32 instruction names and its wave32 marker names are
different events and must not be interchanged.

The generic soc24 values belong to another SoC reference. Their differences
from gfx1100 are not evidence that soc24 needs correction. This restoration
does not introduce new scoped selector aliases.

Runtime validation
------------------

The recorded AMD Radeon Pro W7900 Dual Slot run used BF16 A/B/C/D, FP32
compute, transpose A=T and B=N, batch=1, alpha=1 and beta=0. hipBLASLt and its
gfx1100 device library were built from rocm-libraries develop at
``a0c693f6e0c001e0a69dce9e2ac7c0b30c41feb7`` using the AutoTune
``jinzh_dev`` initialization workflow at ``558caf9``.

Each metric has 300 target-dispatch samples per GEMM: 100 cold and 200
measured, with cache flushing disabled. Nine validated pass groups produced
18 profiling runs and 5,400 target dispatches. The 16,200 same-pass formula,
aggregation and bound checks had no mismatches.

That recorded run used a gfx1100-only SDK configuration. It did not validate
every mixed-architecture parser path in older SDK versions. The regression
suite checks the restored definitions in the shipped mixed-architecture YAML.

.. list-table:: GEMM output checks
   :header-rows: 1

   * - M x N x K
     - Full-output norm
     - Nonzero metrics
     - Zero metrics
   * - 24576 x 16 x 4096
     - 0.000174113
     - 72
     - 12
   * - 32768 x 4096 x 256
     - 0.0000274849
     - 71
     - 13

An independent launch-geometry check validated ``SQ_WAVES`` on all 600
samples. The expected values are 768 for the small-N kernel and 43,776 for
the large-N kernel. These use the macro tiles, workgroup dimensions and
wave32 mode of the actual kernels.

Zero observations do not establish support or non-support. Same-pass formula
checks establish configuration and export relationships, not every counter's
hardware semantics. Nonzero background utilization was observed despite no
reported KFD/AMD-SMI process, and some sysfs queries returned EBUSY. Those
conditions limit attribution of all window traffic solely to GEMM.

Raw selectors
-------------

Means below are instance-aggregated values per dispatch. They are not total
counts over the complete benchmark. Selector agreement is a reference check;
most raw instruction and traffic events have no independent numeric oracle
in these two GEMMs.

.. list-table:: Raw selector observations
   :header-rows: 1

   * - Counter
     - Selector
     - Small-N mean
     - Large-N mean
     - Validation scope

   * - ``GL2C_EA_RDREQ_128B``
     - GL2C/102
     - 1575501.9333333333
     - 542447.16666666663
     - soc21 match; 300 samples per case

   * - ``GL2C_EA_RDREQ_32B``
     - GL2C/99
     - 0
     - 0
     - soc21 match; observed zero, unproven

   * - ``GL2C_EA_RDREQ_64B``
     - GL2C/100
     - 0.59333333333333338
     - 0.66333333333333333
     - soc21 match; 300 samples per case

   * - ``GL2C_EA_RDREQ_96B``
     - GL2C/101
     - 0
     - 0
     - soc21 match; observed zero, unproven

   * - ``GL2C_EA_WRREQ_64B``
     - GL2C/85
     - 6859.833333333333
     - 4132680.7933333335
     - soc21 match; 300 samples per case

   * - ``GL2C_HIT``
     - GL2C/42
     - 193523.79999999999
     - 12897254.063333333
     - soc21 match; 300 samples per case

   * - ``GL2C_MC_RDREQ``
     - GL2C/96
     - 1575728.0933333333
     - 542388.41333333333
     - soc21 match; 300 samples per case

   * - ``GL2C_MC_WRREQ``
     - GL2C/83
     - 6859.9300000000003
     - 4132686.6533333333
     - soc21 match; 300 samples per case

   * - ``GL2C_MC_WRREQ_STALL``
     - GL2C/88
     - 0.95333333333333337
     - 5131.7666666666664
     - soc21 match; 300 samples per case

   * - ``GL2C_MISS``
     - GL2C/43
     - 1581987.5966666667
     - 2639624.2066666665
     - soc21 match; 300 samples per case

   * - ``GRBM_COUNT``
     - GRBM/0
     - 811978.73999999999
     - 2671790.1633333336
     - soc21 match; 300 samples per case

   * - ``GRBM_GUI_ACTIVE``
     - GRBM/2
     - 811978.73999999999
     - 2671790.1633333336
     - soc21 match; 300 samples per case

   * - ``SQC_LDS_BANK_CONFLICT``
     - SQ/256
     - 0
     - 0
     - soc21 match; observed zero, unproven

   * - ``SQC_LDS_IDX_ACTIVE``
     - SQ/261
     - 8262144
     - 49643520
     - soc21 match; 300 samples per case

   * - ``SQ_ACCUM_PREV``
     - SQ/1
     - 206158430160
     - 0
     - soc21 match; 300 samples per case

   * - ``SQ_BUSY_CYCLES``
     - SQ/3
     - 33298123.399999999
     - 115033214.68000001
     - soc21 match; 300 samples per case

   * - ``SQ_INSTS_FLAT``
     - SQ/56
     - 0
     - 0
     - soc21 match; observed zero, unproven

   * - ``SQ_INSTS_GDS``
     - SQ/54
     - 0
     - 0
     - soc21 match; observed zero, unproven

   * - ``SQ_INSTS_LDS``
     - SQ/57
     - 2068992
     - 12871680
     - soc21 match; 300 samples per case

   * - ``SQ_INSTS_SALU``
     - SQ/58
     - 888576
     - 31322112
     - soc21 match; 300 samples per case

   * - ``SQ_INSTS_SMEM``
     - SQ/59
     - 6912
     - 393984
     - soc21 match; 300 samples per case

   * - ``SQ_INSTS_TEX_LOAD``
     - SQ/66
     - 493056
     - 2539008
     - soc21 match; 300 samples per case

   * - ``SQ_INSTS_TEX_STORE``
     - SQ/67
     - 24576
     - 4202496
     - soc21 match; 300 samples per case

   * - ``SQ_INSTS_VALU``
     - SQ/62
     - 1333248
     - 56959992
     - soc21 match; 300 samples per case

   * - ``SQ_INSTS_WAVE32``
     - SQ/70
     - 3919872
     - 76573176
     - soc21 match; 300 samples per case

   * - ``SQ_INSTS_WAVE32_LDS``
     - SQ/72
     - 2068992
     - 12871680
     - soc21 match; 300 samples per case

   * - ``SQ_INSTS_WAVE32_VALU``
     - SQ/73
     - 1333248
     - 56959992
     - soc21 match; 300 samples per case

   * - ``SQ_INST_CYCLES_VALU``
     - SQ/103
     - 1741056
     - 80205048
     - soc21 match; 300 samples per case

   * - ``SQ_INST_CYCLES_VMEM``
     - SQ/106
     - 4578816
     - 33709056
     - soc21 match; 300 samples per case

   * - ``SQ_INST_LEVEL_GDS``
     - SQ/87
     - 0
     - 0
     - soc21 match; observed zero, unproven

   * - ``SQ_INST_LEVEL_LDS``
     - SQ/88
     - 0
     - 0
     - soc21 match; observed zero, unproven

   * - ``SQ_WAIT_ANY``
     - SQ/35
     - 263602300.64666668
     - 1183895211.5833333
     - soc21 match; 300 samples per case

   * - ``SQ_WAIT_INST_ANY``
     - SQ/26
     - 716616.02666666661
     - 514512408.99000001
     - soc21 match; 300 samples per case

   * - ``SQ_WAIT_INST_LDS``
     - SQ/29
     - 194377.48666666666
     - 5290983.2833333332
     - soc21 match; 300 samples per case

   * - ``SQ_WAVE32_INSTS``
     - SQ/82
     - 6060288
     - 118927480
     - soc21 match; 300 samples per case

   * - ``SQ_WAVE64_INSTS``
     - SQ/83
     - 0
     - 0
     - soc21 match; observed zero, unproven

   * - ``SQ_WAVES``
     - SQ/4
     - 768
     - 43776
     - launch geometry: 600 exact matches

   * - ``SQ_WAVE_CYCLES``
     - SQ/24
     - 271672255.37333333
     - 1816964716.3766668
     - soc21 match; 300 samples per case

   * - ``TA_BUFFER_LOAD_WAVEFRONTS``
     - TA/45
     - 493056
     - 2539017.2799999998
     - soc21 match; 300 samples per case

   * - ``TA_BUFFER_STORE_WAVEFRONTS``
     - TA/46
     - 24576
     - 4202496
     - soc21 match; 300 samples per case

   * - ``TA_TA_BUSY``
     - TA/15
     - 63908698.413333334
     - 224828860.66
     - soc21 match; 300 samples per case

References
----------

* `Historical SDK configuration before #9541 <https://github.com/ROCm/rocm-systems/blob/e92aa6ee8c01f991c9da08ea386330a6374e0f4a/projects/rocprofiler-sdk/source/share/rocprofiler-sdk/config.yaml>`_.
* `Architecture cleanup #9541 <https://github.com/ROCm/rocm-systems/commit/9d29a1da312470ebfb1cc3d862e607d9a1960559>`_.
* `gfx11 definition-layer reference <https://github.com/ROCm/rocm-systems/blob/e8da3e5814e887580eb45ab5141a80104d58291c/projects/aqlprofile/src/def/gfx11_def.h>`_.
* `soc21 selector reference <https://github.com/ROCm/rocm-systems/blob/e8da3e5814e887580eb45ab5141a80104d58291c/projects/aqlprofile/linux/soc21_enum.h>`_.
