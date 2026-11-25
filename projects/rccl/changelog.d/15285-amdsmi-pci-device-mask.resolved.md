* Fixed PCI device lookup with `RCCL_USE_AMD_SMI_LIB=1` for device numbers `0x10` through `0x1f`, which previously lost their high bit and could select the wrong GPU or fail lookup.
