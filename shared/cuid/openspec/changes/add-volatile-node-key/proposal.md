## Why

A derived CUID that is the same on every host of a fleet needs a node key that
other local users cannot read. Where the key is kept across reboots is still
open: a UEFI variable is readable by every user through efivarfs until
efivarfs itself changes, and every user-space workaround leaves windows. This
change ships the key without persistence, so derived CUIDs do not wait for
that decision.

## What Changes

**Kernel (`amdgpu`)**

- The driver holds one node key in memory for all its devices. It never
  generates, stores or reads a key anywhere else; without a key it publishes no
  derived value.
- `cuid_seed` (0600, device only, `CAP_SYS_ADMIN`): a write of exactly 32
  octets installs the key and re-keys every registered GPU and partition; a
  read returns the key in use, or fails with `ENODATA` when there is none.
- `cuid_derived` (0444) on every GPU and partition: the HMAC-SHA256 derived
  CUID under the key, or `ENODATA` when there is none.
- The key survives unbind and rebind while amdgpu is loaded, and is wiped when
  it unloads. A reboot or module reload starts with no key.

**Library (`shared/cuid`)**

- Root takes the key from any amdgpu `cuid_seed`; there is no other key
  source. Without it, and for every non-root caller, CPU, NIC and platform
  identities stay temporary CUIDs.
- `cuid_derived` answers first for GPUs and partitions; `ENODATA` means not
  published.
- `amdcuid_set_hash_key()` writes `cuid_seed` when amdgpu is loaded and
  returns `AMDCUID_STATUS_UNSUPPORTED` otherwise. It refuses all-equal and
  published keys; a C caller passes 32 readable octets, and amd-smi refuses a
  key file that is not exactly 32 octets.
- No efivarfs access, key file, tmpfiles.d entry or sleep hook.

**amd-smi**: `set --cuid-seed` sets the key until amdgpu unloads;
`static --cuid`, `list` and `node --cuid` report whether a key is set.

## Capabilities

### New Capabilities

- `cuid/volatile-node-key`: the in-memory node key, `cuid_seed` and
  `cuid_derived` without a key store, and how the library and amd-smi use them.

### Modified Capabilities

None in `openspec/specs/`.

## Impact

- Builds on `split-identity-from-key-store` (identity without a key).
- Persistence is out of scope. `adopt-uefi-key-store`, which stores the key in
  a UEFI variable and needs an efivarfs change, stays a separate proposal.
- Code: kernel `amdgpu_cuid.{c,h}`, ABI document, KUnit; `shared/cuid/lib`,
  tests, docs; `projects/amdsmi`.
