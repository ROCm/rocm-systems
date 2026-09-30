# Design: an in-memory node key

## Context

`split-identity-from-key-store` publishes primary CUIDs, UnitIDs and temporary
CUIDs with no key. A derived CUID that matches across hosts needs a key the
administrator chooses. Keeping it across reboots needs a store that other users
cannot read. The UEFI variable proposed in `adopt-uefi-key-store` is listed
0644 by efivarfs until efivarfs changes, and patching the mode from user space
leaves windows at boot and resume. This change leaves persistence out.

## Goals

- Derived CUIDs for every caller, the same on every host given the same key.
- No key that anybody other than root can read, at any time.
- No public default key, and no key the administrator did not set.
- Nothing outside amdgpu in the kernel, and no boot or resume hooks.

## Non-goals

- Keeping the key across a reboot or module reload. Hosts that need derived
  CUIDs set the key after amdgpu loads, by whatever means they already use to
  configure the host.
- A key store in firmware, a file or the TPM.

## Decisions

### Kernel

- One key per module, in kernel memory, protected by the CUID lock. It starts
  absent. There is no random or default key: a value nobody chose would make
  derived CUIDs look stable when they are not.
- `cuid_seed`, 0600 plus `CAP_SYS_ADMIN` for both read and write, on the PCI
  device only. A write must be exactly 32 octets, otherwise `EINVAL`.
  Writing the key in use changes nothing. A read returns the key so a root
  library can derive CPU, NIC and platform CUIDs with it; without a key it
  fails with `ENODATA`.
- `cuid_derived`, 0444, on the device and every partition, present from
  registration. It fails with `ENODATA` until a key is set, then returns the
  derived CUID. A partition created later derives with the key already held.
- A key write re-derives every registered component under the lock.
- Module exit wipes the key with `memzero_explicit()`.
- No `cuid_seed_state`: with one kind of key, `cuid_derived` succeeding is the
  state.

### Library

- Root: key from the first readable amdgpu `cuid_seed`; `ENODATA` means no
  key. No efivarfs, no key file. Any other failure, or a read of other than
  32 octets, is `AMDCUID_STATUS_FILE_ERROR` from `amdcuid_get_key_info()` and
  `amdcuid_refresh()`, and that call's lookups are temporary.
- The key is read on entry to each call that derives or reports it and zeroed
  before that call returns. The library holds no copy between calls, so a
  process that drops root, or a child it forks, inherits none. During a call
  the key, and the HMAC's padded key, pads and hash contexts, live in mappings
  marked `MADV_WIPEONFORK` and `MADV_DONTDUMP`, so a child that another thread
  forks mid-call gets them zero-filled and a core dump leaves them out. Where
  either advice fails (a kernel before 4.14 for `MADV_WIPEONFORK`, or a
  seccomp policy) there is no key; temporary CUIDs, which need no secret, are
  still derived, in memory that is only zeroed after use. The one copy left
  on the stack is SHA-256's message schedule inside `rocm::sha2`, for the
  length of one block's compression.
- GPUs and partitions: `cuid_derived` first, source `DRIVER`; `ENODATA` is the
  same as absent. A whole GPU without it is keyed like a CPU when root holds
  the key, else temporary; a partition without it is refused, as in
  `split-identity-from-key-store`.
- `amdcuid_set_hash_key()` writes `cuid_seed` on one device, which keys the
  whole node; without amdgpu it returns `AMDCUID_STATUS_UNSUPPORTED`.
- `amdcuid_get_key_info()` reports whether a key is set and its fingerprint,
  to root. `provisioned` means a key is set. Root that amdgpu refuses, without
  `CAP_SYS_ADMIN`, gets `PERMISSION_DENIED`, not "no key".

### amd-smi

- `set --cuid-seed <file|->` as before; its help and docs say the key lasts
  until amdgpu unloads or the host reboots.
- `effective_seed` reads `provisioned` or `temporary`.

## Risks and trade-offs

- **Derived CUIDs disappear at every reboot** until the key is set again. A
  consumer that records them sees `N/A` or a temporary CUID in between.
- **Partitions have no CUID until the key is set** in DPX and above, as with
  the identity series.
- **The key passes through user space at every boot.** Where the administrator
  keeps it is their choice and outside this change.
- **Re-keying changes every derived CUID.** There is no mapping from old to new.

## Alternatives

- UEFI variable with an efivarfs change: `adopt-uefi-key-store`, kept as the
  next step.
- UEFI variable with user-space mode fixes (tmpfiles.d, a sleep hook, a
  write-through): rejected; the key is readable between boot or resume and the
  fix.
- `LINUX_EFI_RANDOM_SEED_TABLE_GUID`: hidden on 6.2 and later, but root cannot
  set or read the key without amdgpu and it uses a GUID reserved for the random
  seed.
- A root-only file pushed by udev: persistence through user space, deferred with
  the rest.
