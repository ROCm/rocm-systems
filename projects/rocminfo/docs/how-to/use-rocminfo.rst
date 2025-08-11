.. meta::
  :description: Using ROCmInfo
  :keywords: rocminfo, enumerator, info, AMD, ROCm, HSA, hsa


================
Using ROCmInfo
================

The ROCmInfo command provides information about the Heterogenous System Architecture (HSA) system attributes and agents. Each agent represents a device and a device can be a CPU or a GPU.

The output has the following two sections:

* HSA System Attributes - List of general information of the system.

* HSA agents - List of devices in the system.

See the following example output of the ROCmInfo command on a system with MI300X:

.. code-block::

        HSA System Attributes 
        =====================
        Runtime Version:         1.1
        Runtime Ext Version:     1.6
        System Timestamp Freq.:  1000.000000MHz
        Sig. Max Wait Duration:  18446744073709551615 (0xFFFFFFFFFFFFFFFF) (timestamp count)
        Machine Model:           LARGE
        System Endianness:       LITTLE
        Mwaitx:                  DISABLED
        DMAbuf Support:          YES
        ==========
        HSA Agents
        ==========
        *******
        Agent 1
        *******
          Name:                    AMD Ryzen 9 7950X 16-Core Processor
          Uuid:                    CPU-XX
          Marketing Name:          AMD Ryzen 9 7950X 16-Core Processor\
          Vendor Name:             CPU\
          Feature:                 None specified
          Profile:                 FULL_PROFILE
          Float Round Mode:        NEAR
          Max Queue Number:        0(0x0)
          Queue Min Size:          0(0x0)\
          Queue Max Size:          0(0x0)
          Queue Type:              MULTI
          Node:                    0
          Device Type:             CPU
          Cache Info:
            L1:                      32768(0x8000) KB
          Chip ID:                 0(0x0)
          ASIC Revision:           0(0x0)
          Cacheline Size:          64(0x40)
          Max Clock Freq. (MHz):   4500
          BDFID:                   0
          Internal Node ID:        0
          Compute Unit:            32
          SIMDs per CU:            0
          Shader Engines:          0
          Shader Arrs. per Eng.:   0
          WatchPts on Addr. Ranges:1
          Memory Properties:
          Features:                None
          Pool Info:
            Pool 1
              Segment:                 GLOBAL; FLAGS: FINE GRAINED
              Size:                    65111316(0x3e18514) KB
              Allocatable:             TRUE
              Alloc Granule:           4KB
              Alloc Recommended Granule:4KB
              Alloc Alignment:         4KB
              Accessible by all:       TRUE
            Pool 2
              Segment:                 GLOBAL; FLAGS: KERNARG, FINE GRAINED
              Size:                    65111316(0x3e18514) KB
              Allocatable:             TRUE
              Alloc Granule:           4KB
              Alloc Recommended Granule:4KB
              Alloc Alignment:         4KB
              Accessible by all:       TRUE
            Pool 3
              Segment:                 GLOBAL; FLAGS: COARSE GRAINED
              Size:                    65111316(0x3e18514) KB
              Allocatable:             TRUE
              Alloc Granule:           4KB
              Alloc Recommended Granule:4KB
              Alloc Alignment:         4KB
              Accessible by all:       TRUE
          ISA Info:
        *******
        Agent 2
        *******
          Name:                    gfx941
          Uuid:                    GPU-a8673551b40c6374
          Marketing Name:          AMD Instinct MI300X
          Vendor Name:             AMD
          Feature:                 KERNEL_DISPATCH
          Profile:                 BASE_PROFILE
          Float Round Mode:        NEAR
          Max Queue Number:        128(0x80)
          Queue Min Size:          64(0x40)
          Queue Max Size:          131072(0x20000)
          Queue Type:              MULTI
          Node:                    1
          Device Type:             GPU
          Cache Info:
            L1:                      32(0x20) KB
            L2:                      4096(0x1000) KB
            L3:                      262144(0x40000) KB
          Chip ID:                 29857(0x74a1)
          ASIC Revision:           0(0x0)
          Cacheline Size:          64(0x40)
          Max Clock Freq. (MHz):   1800
          BDFID:                   768
          Internal Node ID:        1
          Compute Unit:            304
          SIMDs per CU:            4
          Shader Engines:          32
          Shader Arrs. per Eng.:   1
          WatchPts on Addr. Ranges:4
          Coherent Host Access:    FALSE
          Memory Properties:
          Features:                KERNEL_DISPATCH
          Fast F16 Operation:      TRUE
          Wavefront Size:          64(0x40)
          Workgroup Max Size:      1024(0x400)
          Workgroup Max Size per Dimension:
            x                        1024(0x400)
            y                        1024(0x400)
            z                        1024(0x400)
          Max Waves Per CU:        32(0x20)
          Max Work-item Per CU:    2048(0x800)
          Grid Max Size:           4294967295(0xffffffff)
          Grid Max Size per Dimension:
            x                        4294967295(0xffffffff)
            y                        4294967295(0xffffffff)
            z                        4294967295(0xffffffff)
          Max fbarriers/Workgrp:   32
          Packet Processor uCode:: 141
          SDMA engine uCode::      19
          IOMMU Support::          None
          Pool Info:
            Pool 1
              Segment:                 GLOBAL; FLAGS: COARSE GRAINED
              Size:                    134201344(0x7ffc000) KB
              Allocatable:             TRUE
              Alloc Granule:           4KB
              Alloc Recommended Granule:2048KB
              Alloc Alignment:         4KB
              Accessible by all:       FALSE
            Pool 2
              Segment:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
              Size:                    134201344(0x7ffc000) KB
              Allocatable:             TRUE
              Alloc Granule:           4KB
              Alloc Recommended Granule:2048KB
              Alloc Alignment:         4KB
              Accessible by all:       FALSE
            Pool 3
              Segment:                 GROUP
              Size:                    64(0x40) KB
              Allocatable:             FALSE
              Alloc Granule:           0KB
              Alloc Recommended Granule:0KB
              Alloc Alignment:         0KB
              Accessible by all:       FALSE
          ISA Info:
            ISA 1
              Name:                    amdgcn-amd-amdhsa--gfx941:sramecc+:xnack-
              Machine Models:          HSA_MACHINE_MODEL_LARGE
              Profiles:                HSA_PROFILE_BASE
              Default Rounding Mode:   NEAR
              Default Rounding Mode:   NEAR
              Fast f16:                TRUE
              Workgroup Max Size:      1024(0x400
        
              Workgroup Max Size per Dimension:
                x                        1024(0x400)
                y                        1024(0x400)
                z                        1024(0x400)
              Grid Max Size:           4294967295(0xffffffff)
              Grid Max Size per Dimension:
                x                        4294967295(0xffffffff)
                y                        4294967295(0xffffffff)
                z                        4294967295(0xffffffff)
        
        *** Done ***

