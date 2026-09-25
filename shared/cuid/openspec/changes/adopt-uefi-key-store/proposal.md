## Why

The CUID key model is replaced. There is no daemon, no `/etc/amdcuid` key file
and no public default seed; the host's UEFI variable store is the single
authoritative key store, owned by the driver. Separately, partition 0 of every
compute-partition mode got the same CUID, because the UnitID was
`partition index + 1` over the first XCD's serial.

Every implemented layer contradicted that: the kernel fell back to
`AMD-CUID-DEFAULT-SEED-v1`, the library owned the key file and the record store,
the daemon reconciled driver seeds, a TPM-sealed variable was offered that the
driver cannot read, and the partition UnitID collided across modes.

## What Changes

**Kernel (`amdgpu`)**

- **BREAKING** The node key comes from the UEFI variable `AmdCuidKey`
  (GUID `e41c1f7f-63cb-46b9-bf27-36a55f92a06d`). The driver reads it at the first
  CUID registration and, if absent, generates 32 random octets and writes it.
- **BREAKING** Without a readable or creatable variable the driver publishes no
  CUID attribute at all. `CUID_DEFAULT_SEED` and the padded-default restore
  are removed.
- A `cuid_seed` write persists the key to the variable before re-keying; a
  failed write changes nothing.
- **BREAKING** `cuid_seed_state` reports `unprovisioned` or `provisioned`.
- **BREAKING** A partition's UnitID is its logical XCC range,
  `(xcc_count << 6) | first_xcc`, and its serial is the device serial.
- New `cuid_unit_id` (0444) beside every `cuid_derived`.
- efivarfs patch (drafted, not sent): entries under the CUID GUID are 0600.

**Library and packaging (`shared/cuid`)**

- **BREAKING** Remove the daemon, its service, udev and maintainer-script hooks,
  IPC, the key file, the record store, driver-seed reconciliation,
  `kDefaultSeed` and `scripts/amdcuid_uefi.py`.
- Root obtains the key from `cuid_seed`, else the efivarfs variable; without a
  key, and for every non-root caller, CPU/NIC/platform identities are temporary CUIDs.
- **BREAKING** `amdcuid_tool` is removed. amd-smi (`set --cuid-seed`) and
  `amdcuid_set_hash_key()` set the key, through `cuid_seed` or, without amdgpu,
  efivarfs; a key must be exactly 32 octets, and trivial and public keys are
  refused. `amd-smi node --cuid` lists every component's CUID.
- **BREAKING** Temporary CUIDs use a machine-id-keyed application key
  (provisional until the temporary-CUID specification is final).
- GPU discovery lists AMD GPUs (Vendor ID `0x1002`) only.
- Packaging ships a tmpfiles.d entry that makes the variable 0600 on kernels
  without the efivarfs patch.

**amd-smi**: seed-state strings and sources (`DRIVER`, `LIBRARY`) follow.

## Impact

- Supersedes, in other open changes: `key-constants` "Canonical fallback seed";
  `sysfs-interface` "Default seed" and the `default|custom` states;
  `component-sources` "UnitID identifies a sub-unit" (index + 1);
  `library-consumer` staged lookup through `STORE`; the daemon and the
  library's key authority, which no change specified;
  `amdsmi-seed-provisioning` "A drifted driver seed is repaired".
- Supersedes the daemon and record-store key sources in
  `integrate-cuid-into-amdsmi` (`amdsmi-identity-api`,
  `amdsmi-seed-provisioning`): amd-smi reports only `UNKNOWN`, `DRIVER` and
  `LIBRARY`.
- Code: kernel `amdgpu_cuid.{c,h}`, `amdgpu_xcp.c`, ABI doc, selftests,
  `fs/efivarfs`; compat shim; `shared/cuid/{lib,cli,daemon,scripts,tests}`;
  `projects/amdsmi` CUID strings.
- Design and every decision: `design.md`.
