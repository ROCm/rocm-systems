## 1. Kernel

- [x] 1.1 Identity series: two xcp sysfs fixes; amdgpu `cuid_primary` and
      `cuid_unit_id` for devices and partitions, with no key or HMAC; KUnit
      for packing, UUIDv8 and UnitID; ABI document.
- [x] 1.2 Node-key series on top, amdgpu only: `add-volatile-node-key` 1.1
      to 1.4.
- [x] 1.3 Every commit builds with W=1; checkpatch --strict; KUnit at both heads.

## 2. Library and amd-smi

- [x] 2.1 Identity series: driver marker `cuid_unit_id`; GPU derived CUID falls
      back to a temporary CUID without a key; remove the daemon, key file,
      record store and default seed; no key setter.
- [x] 2.2 Node-key series on top: `add-volatile-node-key` 2.1 to 2.4.
- [x] 2.3 ctest, amd-smi gtest and pytest at both heads.

## 3. Verification

- [x] 3.1 Fixture sysfs (`cuid_gpu_paths`): one library serves a kernel from
      either series, including `cuid_derived` failing with `ENODATA`.
- [x] 3.2 MI350X, identity kernel: primary and UnitID for every partition in
      SPX/DPX/QPX/CPX; amd-smi with the identity library.
- [x] 3.3 MI350X, node-key kernel: `add-volatile-node-key` 4.2.
- [x] 3.4 Radeon PRO W6800 with a key-store kernel and the identity library:
      root reads the driver's `cuid_primary`; every derived CUID is temporary
      and the same for root and other users.
- [x] 3.5 Instinct MI350X host NICs: every PCI function listed for root and
      other users alike, with distinct primary CUIDs (UnitID = function).
