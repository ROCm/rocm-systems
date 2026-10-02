## Why

A derived CUID that is not temporary needs a node key that other local users
cannot read. How amdgpu holds that key, and whether it keeps it across a
reboot, is reviewed on its own.

Primary CUIDs, UnitIDs, partition identities and temporary CUIDs need no key.
Holding them back until the node key is accepted delays everything that does
not depend on it.

## What Changes

The work lands as two series, the second on top of the first. This change
specifies the first and records the boundary; the second is
`add-volatile-node-key`, which keeps the key only in amdgpu's memory.
Keeping it across reboots in a UEFI variable is a later change,
`adopt-uefi-key-store`.

**Identity (no key)**

- amdgpu publishes `cuid_primary` (0400) and `cuid_unit_id` (0444) for every
  whole GPU and every compute partition, and follows partition-mode switches.
  A partition's UnitID is its logical XCC range. No key, no HMAC, no EFI.
- The library takes a GPU or partition as driver-published when
  `cuid_unit_id` exists, and reads `cuid_primary` for root. Without a key there
  are no derived CUIDs that are not temporary: every caller gets a temporary
  CUID for CPUs, NICs, NPUs, the platform and whole GPUs, keyed by the machine
  ID. A partition has none.
- Only AMD GPUs are listed. A CPU's auxiliary Routing ID is its physical
  package ID. The platform's auxiliary structure is the CPU Format with
  Component Type 0 and every device field zero.
- amd-smi reports primary and temporary CUIDs with their source, and
  `amd-smi node --cuid` lists every component. There is no key setter and no
  key state.
- **BREAKING** The daemon, key file, record store, `amdcuid_tool` and public
  default seed are removed, since nothing in this series uses a key.
  `amdcuid_set_hash_key()` and `amdcuid_get_key_info()` return
  `AMDCUID_STATUS_UNSUPPORTED`.

**Node key (on top, `add-volatile-node-key`)**

- amdgpu holds one node key in memory, set by root through `cuid_seed` and
  wiped when the module unloads, and publishes `cuid_derived`. Nothing
  outside amdgpu changes.
- The library reads the key from `cuid_seed`, `amdcuid_set_hash_key()` and
  `amd-smi set --cuid-seed` set it, and derived CUIDs replace temporary ones for
  root and, through `cuid_derived`, for everyone.

## Impact

- A library from either series works on a kernel from either series. With
  this series' library a node-key kernel's `cuid_derived` is not read.
- Until the node key lands no caller gets a partition CUID from the library:
  a partition has no temporary CUID because every partition of a GPU shares its
  parent's routing information. Root can read a partition's `cuid_primary` and
  `cuid_unit_id` from sysfs. An amd-smi handle for a GPU in SPX reports the
  whole GPU's CUID; one for a partition in DPX and above reports none.
- A NIC's UnitID is its PCI function number, so NIC functions that report one
  serial number have distinct CUIDs; amd-smi lists one component per PCI
  function.
- `amdsmi-seed-provisioning` and the `set --cuid-seed` requirement of
  `amdsmi-cli` in `integrate-cuid-into-amdsmi` wait for the node key.
- Supersedes, in other open changes: `key-constants` "Canonical fallback seed"
  and "Temporary and auxiliary fixed key"; `component-sources` "UnitID
  identifies a sub-unit" (index + 1); `library-consumer` staged lookup through
  `STORE` and key handling; the daemon, which no change specified.
- Code: kernel `amdgpu_cuid.{c,h}`, `amdgpu_xcp.c`, ABI document, KUnit;
  `shared/cuid/{lib,cli,daemon,scripts,tests}`; `projects/amdsmi`.
