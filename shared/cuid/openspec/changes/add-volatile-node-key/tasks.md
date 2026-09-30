## 1. Kernel (amdgpu)

- [x] 1.1 In-memory node key; `cuid_seed` (0600, `CAP_SYS_ADMIN`, device only,
      32-octet write, read or `ENODATA`); `cuid_derived` on devices and
      partitions with `ENODATA` without a key; no EFI code, no
      `cuid_seed_state`, no generated key; key wiped at module exit.
- [x] 1.2 ABI document.
- [x] 1.3 KUnit: HMAC known-answer vectors; a key set after registration
      reaches every component.
- [x] 1.4 Every commit builds with `W=1`; `checkpatch --strict`, sparse,
      kernel-doc and KUnit clean, also under KASAN and UBSAN.

## 2. Library and amd-smi

- [x] 2.1 Key only from `cuid_seed`; `ENODATA` from `cuid_derived` is
      unpublished; remove efivarfs, key-variable parsing, immutable-flag
      handling, the write-through, `tmpfiles.d/amdcuid.conf`, the sleep hook
      and their packaging.
- [x] 2.2 `amdcuid_set_hash_key()` writes `cuid_seed`, `UNSUPPORTED` without
      amdgpu; `amdcuid_get_key_info()` as specified.
- [x] 2.3 amd-smi `set --cuid-seed`, `effective_seed` `provisioned` or
      `temporary`, help and docs.
- [x] 2.4 Unit and lifecycle tests for the above.

## 3. Documentation

- [x] 3.1 `shared/cuid` and amd-smi docs, the openspec README and
      `split-identity-from-key-store` describe the in-memory key and point to
      `adopt-uefi-key-store` for persistence.

## 4. Verification

- [x] 4.1 Library and amd-smi suites under ASan and UBSan; linters clean.
- [x] 4.2 MI350X: set the key, every GPU and partition derives and matches an
      independent computation; mode switches; unbind/rebind keeps the key;
      reload and reboot lose it; library and amd-smi suites.
