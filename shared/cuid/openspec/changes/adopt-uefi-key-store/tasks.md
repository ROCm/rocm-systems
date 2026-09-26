## 1. Kernel (amdgpu)

- [x] 1.1 Load or create `AmdCuidKey` once per module lifetime, outside
      `cuid_seed_lock`, guarded by `IS_ENABLED(CONFIG_EFI)`; `MODULE_IMPORT_NS`.
- [x] 1.2 Publish nothing when there is no key; remove `CUID_DEFAULT_SEED` and
      the default-restore branch.
- [x] 1.3 `cuid_seed` store persists to the variable (flags = provisioned)
      before installing the key; `-EIO` and no change on failure.
- [x] 1.4 `cuid_seed_state`: `unprovisioned` / `provisioned`.
- [x] 1.5 Partition UnitID = logical XCC range; device serial for partitions;
      refuse a non-contiguous mask.
- [x] 1.6 `cuid_unit_id` attribute (0444) on device and partitions.
- [x] 1.7 ABI document, selftests, KAT and lifecycle shims, vectors.
- [x] 1.8 efivarfs: CUID GUID entries 0600; GUID in `include/linux/efi.h`.
- [x] 1.9 Compare the UnitID with the buddy-table proposal; keep the XCC range
      and specify the closed form for tools outside the driver.

## 2. Kernel backport (< 6.17)

- [x] 2.1 Port 1.1–1.7, with the efivar API differences of older kernels.
- [x] 2.2 Build against a distribution DKMS source and run the MI350X
      partition matrix.

## 3. Library, amd-smi, packaging

- [x] 3.1 Remove the daemon, service, udev and maintainer-script hooks, IPC,
      reconciliation, key file and lock, record store, default seed and the
      TPM-sealed variable script, with their tests.
- [x] 3.2 Key source: `cuid_seed`, else the efivarfs payload (validated), else
      none; temporary CUIDs otherwise and for non-root callers.
- [x] 3.3 Key setting in `amdcuid_set_hash_key()`, used by amd-smi; exactly 32
      octets; refuse all-equal and public constants; sysfs first, efivarfs
      without amdgpu. No standalone tool.
- [x] 3.4 Temporary-CUID construction per design.md; regenerate A-* vectors.
- [x] 3.5 tmpfiles.d entry for the variable.
- [x] 3.6 amd-smi seed-state strings and sources.
- [ ] 3.7 An automated OVMF test for a driver-created variable and for the
      efivarfs path of `amdcuid_set_hash_key()`. Both are covered on hardware
      only.
- [x] 3.8 Library minor version 2.2, one key read per enumeration, AMD-only GPU
      enumeration, source and temporary flag in `amd-smi list`, tmpfiles.d
      install from the amd-smi build.
- [x] 3.9 `amdsmi_get_cuid_components()` and `amd-smi node --cuid`: every
      component's CUID, since there is no standalone tool to list them.

## 4. Verification

- [x] 4.1 Radeon PRO W6800: the variable is created on first load, reused on
      reload, a `cuid_seed` write persists, and removal yields a fresh key.
- [x] 4.2 MI350X: SPX/DPX/QPX/CPX partition CUIDs pairwise distinct.
- [x] 4.3 efivarfs mode after remount, with and without the patch (OVMF).
- [x] 4.4 Radeon PRO W6800 with `cuid_derived`: library suite, amd-smi
      driver-published tests, `cuid_sysfs_test.sh` with re-keying.
- [x] 4.5 MI350X: re-qualify the backport series.
