## 1. Kernel

- [ ] 1.1 Identity series: two xcp sysfs fixes; amdgpu `cuid_primary` and
      `cuid_unit_id` for devices and partitions, with no key or HMAC; KUnit
      for packing, UUIDv8 and UnitID; ABI document.
- [ ] 1.2 Key-store series on top: efivarfs secret table with
      `LoaderSystemToken`; `AmdCuidKey` entry and `AMD_CUID_EFI_GUID` in
      `include/linux/efi.h`; amdgpu derived CUIDs, `cuid_seed`,
      `cuid_seed_state`; KUnit HMAC vectors.
- [ ] 1.3 Every commit builds with W=1; checkpatch --strict; KUnit at both heads.

## 2. Library and amd-smi

- [x] 2.1 Identity series: driver marker `cuid_unit_id`; GPU derived CUID falls
      back to a temporary CUID without a key; remove the daemon, key file,
      record store and default seed; no key setter.
- [ ] 2.2 Key-store series on top: key read from `cuid_seed` or efivarfs,
      `amdcuid_set_hash_key()`, `amd-smi set --cuid-seed`, tmpfiles.d entry,
      `adopt-uefi-key-store`.
- [ ] 2.3 ctest, amd-smi gtest and pytest at both heads.

## 3. Verification

- [ ] 3.1 efivarfs under OVMF: table entries 0600 at mount, resync and create;
      other variables 0644.
- [ ] 3.2 MI350X, identity kernel: primary and UnitID for every partition in
      SPX/DPX/QPX/CPX; amd-smi with the identity library.
- [ ] 3.3 MI350X, key-store kernel: variable created and 0600, `cuid_seed`
      re-key persists across reboot, derived CUIDs distinct per partition.
- [x] 3.4 Radeon PRO W6800 with a key-store kernel and the identity library:
      root reads the driver's `cuid_primary`; every derived CUID is temporary
      and the same for root and other users.
