# Design: a UEFI-backed node key and range-encoded partition UnitIDs

## Context

The derived CUID of every component on a host is `HMAC-SHA256(key, primary)`.
Until now the key had two owners: amdgpu held an in-memory copy that fell back
to a public default seed, and the library kept a key file, a record store and a
daemon that reconciled the two. Partitions used `index + 1` as their UnitID, so
partition 0 of every compute mode had the same CUID.

This change gives the key one owner, the host's UEFI variable store, removes
everything that existed only to keep two copies in step, and gives every
partition of every mode a distinct UnitID.

## Goals

- One authoritative key per host, readable by the driver before user space runs.
- No public default key, and no CUID published under a key nobody chose or generated.
- Every partition of every compute mode gets a distinct, deterministic identity,
  and the whole device keeps UnitID 0.
- An unprivileged caller, or a host without a key, still gets a usable
  temporary CUID that is marked as temporary.

## Non-goals

- An out-of-band implementation. The variable format is the contract it will use.
- Per-VM serials and a virtualized key store. These belong to the hypervisor.
- Non-AMD devices and an OS-level CUID service.

## Decisions

### The key store

One variable per host:

| Field | Value |
|---|---|
| Name | `AmdCuidKey` |
| Vendor GUID | `e41c1f7f-63cb-46b9-bf27-36a55f92a06d` |
| Attributes | `NON_VOLATILE \| BOOTSERVICE_ACCESS \| RUNTIME_ACCESS` (0x7), never authenticated |
| Payload | 36 octets: `version` (1 octet, = 1), `flags` (1 octet, bit 0 = provisioned by an administrator, other bits zero), 2 reserved octets (zero), 32-octet key |

A payload of any other size, version or reserved value is malformed. Nobody
overwrites a malformed variable automatically: the driver publishes no CUID and
logs once.

**Driver flow**, once per module lifetime, at the first CUID registration and
outside `cuid_seed_lock`:

1. `!IS_ENABLED(CONFIG_EFI)` or `!efivar_is_available()` → no key.
2. `efivar_lock()`; `efivar_get_variable()`.
3. `EFI_NOT_FOUND` → `get_random_bytes(32)`, flags = 0,
   `efivar_set_variable_locked(..., nonblocking=false)`. Any write error → no key.
4. Any other read error, or a malformed payload → no key.
5. `efivar_unlock()`. The result, key or no key, is cached until module unload.

`efi=noruntime`, `CONFIG_EFI_DISABLE_RUNTIME`, hypervisors without a runtime
variable store and non-UEFI boots all stop at step 1.

**Without a key the driver publishes nothing**: no `cuid_primary`, no
`cuid_derived`, no seed attributes, on the device or its partitions. Consumers
have one rule: no driver CUID means a library temporary CUID. The probe never
fails and never defers because of CUID.

**`cuid_seed` store** (`CAP_SYS_ADMIN`, exactly 32 octets): the driver writes
the variable first, with flags = provisioned, and only on success installs the
key and re-keys every component. A failed write returns `-EIO` and changes
nothing, because the firmware copy is authoritative. There is no way back to
`unprovisioned` other than deleting the variable while amdgpu is unloaded.

**`cuid_seed_state`** reports `unprovisioned` for a key the driver generated
and `provisioned` once flags bit 0 is set.

**Hosts without amdgpu.** Nothing creates the key implicitly: a lookup never
writes efivarfs. Without the variable such a host has temporary CUIDs only.

**Setting the key and listing CUIDs.** There is no standalone tool. amd-smi is
the administrator's interface: `amd-smi set --cuid-seed <file|->` sets the key,
and `amd-smi node --cuid` lists every component's CUID through
`amdsmi_get_cuid_components()`, including the CPUs, NICs and platform amd-smi
does not otherwise manage. A program calls `amdcuid_set_hash_key()`, which
amd-smi uses. The call:

- accepts exactly 32 octets;
- refuses a key whose octets are all equal, or that equals a published constant
  zero-padded (`AMD-CUID-DEFAULT-SEED-v1`, `AMD-CUID-TEMP-KEY-v1`, the test keys
  in `cuid_vectors.txt`); the kernel's `cuid_seed` does not refuse them, so
  the check is in user space only;
- writes `cuid_seed` on one device when amdgpu is loaded, so the driver
  persists it; otherwise writes the efivarfs variable directly (4-octet
  attribute header and the 36-octet payload, flags = provisioned);
- clears the immutable flag and makes the variable 0600 where efivarfs lists
  it, warning rather than failing if it cannot.

Every derived CUID on the host changes. The library is a static archive,
`libamdcuid_static.a`, with one header and a CMake package, so any program can
link it without a runtime dependency; amd-smi absorbs it.

### Confidentiality of the variable

efivarfs shows a runtime-access variable as 0644, readable by every local user.

1. A kernel patch makes efivarfs create entries under the CUID vendor GUID with
   mode 0600, next to the existing special case for the RNG seed GUID. The
   entry is not hidden, because `amdcuid_set_hash_key()` must be able to write it
   without amdgpu. The GUID is defined once, in `include/linux/efi.h`.
2. For kernels without that patch, the package ships `tmpfiles.d/amdcuid.conf`:
   ```
   h /sys/firmware/efi/efivars/AmdCuidKey-e41c1f7f-63cb-46b9-bf27-36a55f92a06d - - - - -i
   z /sys/firmware/efi/efivars/AmdCuidKey-e41c1f7f-63cb-46b9-bf27-36a55f92a06d 0600 root root -
   ```
   efivarfs marks every entry outside a short validated list `S_IMMUTABLE`, so
   a bare `chmod` fails with `EPERM`; the `h … -i` line clears the flag first.
   The variable then stays mutable, so root could remove it, but root can
   already rewrite it. The rule takes effect from the boot after the variable
   first appears, because efivarfs enumerates variables only at mount. The
   key-setting call applies the same clear-and-chmod right after it writes.
3. What remains exposed is the first boot after creation, and kernels whose
   efivarfs cannot chmod. A per-host random key read locally reveals nothing
   that `cuid_derived` (0444) does not; a fleet key read on one host is the
   real risk, so the key-setting call warns when it cannot make the variable 0600.

Rejected: keeping the fleet key somewhere else (two stores again), accepting
the exposure, and a boot-services-only variable handed over before boot.

### Library

- **Key for CPU, NIC and platform derived CUIDs, root only**: `cuid_seed` of any
  amdgpu device, which is the driver's key by construction; else the efivarfs
  variable with its header stripped and its payload validated; else none.
- **No key, or not root** → temporary CUID: bit 117 set, source `LIBRARY`. The
  same CPU therefore has a permanent CUID for root and a temporary one for other
  users; bit 117 and the reported source tell them apart.
- **GPUs and partitions** come from the driver only. `cuid_derived` is what
  says the driver publishes a CUID; a `cuid_primary` without it counts as
  nothing published. A GPU without a driver CUID gets a temporary one; a
  partition without one is refused, because a partition has no temporary
  identity. Only AMD GPUs (vendor 0x1002) are
  listed, so a BMC's display adapter gets no CUID.
- A lookup reads the key once and uses it for every property query on the
  handles it returns; a key changed later takes effect at the next lookup or
  `amdcuid_refresh()`.
- Removed: the default seed, the `/etc/amdcuid` key file and its lock, the
  `/var/lib/amdcuid` record store, the daemon with its service, udev and
  maintainer-script hooks, IPC, driver-seed reconciliation, and the TPM-sealed
  variable script, whose variable the driver cannot read.
- Reported sources are `DRIVER` and `LIBRARY`. amd-smi's seed-state strings
  follow the kernel.

### Partition identity

**UnitID is the partition's logical XCC range.** For a partition whose logical
XCC mask `m` is contiguous, as `XCP_INST_MASK` guarantees:

```
UnitID = (hweight(m) << 6) | __ffs(m)
bits 0:5   first logical XCC   (0..63)
bits 6:11  XCC count           (1..63)
bit  12    SLC                 (zero in every other mode)
```

The whole device keeps UnitID 0, and a partition's UnitID is never 0 because
its count is at least 1. On 8 XCCs: SPX 0x200; DPX 0x100, 0x104; QPX 0x080,
0x082, 0x084, 0x086; CPX 0x040 to 0x047. Two partitions with the same first
XCC and count are the same partition, so the values are distinct across every
mode. 6-XCC and harvested parts work unchanged because the mask is logical. A
non-contiguous mask is refused: that partition gets no CUID and the driver
warns.

SLC spans the whole device, like SPX, but stripes memory differently, so it is
the SPX range with bit 12 set: 0x1200 on 8 XCCs. The driver does not yet know
when SLC is active, so it never sets the bit today.

A tool that knows the key can enumerate every allowed partition's CUID without
the driver. For `N` logical XCCs split into `k` equal partitions, partition `i`
has UnitID `((N / k) << 6) | (i * N / k)`.

**Limits.** The encoding holds a first XCC of 0..63 and a count of 1..63, so up
to 63 XCCs per device and one partition per XCC in any mode. The driver's XCC
mask is 32 bits wide, so today it can name at most 32 XCCs, and amdgpu caps
partitions at `MAX_XCP` = 8. UnitID is interpreted per component type, and the
XCC range is the GPU's encoding. A CPU is identified per package, with UnitID 0
and its physical package ID as the Routing ID; cores are not components, so a
core count never reaches the UnitID. A sub-unit of another component type would
get its own encoding in the same 13 bits.

**Alternatives considered:**

| Criterion | XCC bitmask | `mode << 8 \| index + 1` | Buddy table (2N − 1 entries, published by KFD) | XCC range |
|---|---|---|---|---|
| Distinct per mode and partition | Yes | Yes | Yes | Yes |
| Device 0, SPX distinct from it | Yes | Yes | No: SPX is 0 | Yes |
| Fits 13 bits | Only up to 13 XCCs | Yes | Yes | Yes |
| SLC | No | Needs a mode number | No | Bit 12 |
| 6-XCC and harvested parts | Yes | Yes | No: `N = 2^x` only | Yes |
| No table maintained in the kernel | Yes | No | No | Yes |
| Computable without the driver | Yes | Only from the table | Only from the table | Yes |

Every buddy partition is a contiguous run of XCCs, so each maps one-to-one onto
an XCC range; the range carries the same information without the table and
also covers the cases the table cannot.

- **Serial:** every partition uses the device serial (`adev->unique_id`, then
  the PCIe DSN). With a distinct UnitID a per-XCD serial adds nothing.
- **Memory partition mode (NPS)** does not enter the identity.
- **`cuid_unit_id`** (0444, decimal) is published beside every `cuid_derived`,
  0 on the device. It is how an unprivileged reader tells which partition an
  ID names.
- The library does not compute partition UnitIDs itself: without the driver
  there is no partition CUID to produce.
- The key is exposed only as `cuid_seed` (0600 and `CAP_SYS_ADMIN`); there is
  no procfs copy.

### Temporary CUIDs (provisional)

The construction follows machine-id(5), which asks applications to key their
own identifiers from the machine ID rather than expose it:

```
K_app   = HMAC-SHA256(key = machine-id (16 octets), msg = "AMD-CUID-TEMP-v2")
serial  = HMAC-SHA256(K_app, S)[0:8]     S = the 32-octet auxiliary structure,
                                           machine-id field zero-filled
derived = HMAC-SHA256(K_app, raw_primary)
```

The result is UUIDv8 with bit 117 set, and the library refuses when there is no
machine ID. The construction is provisional until the temporary-CUID
specification is final; a change would be confined to the key derivation and
the A-* vectors.

**Containers.** The library reads `/etc/machine-id`, then
`/var/lib/dbus/machine-id`, and refuses on a missing, empty or non-hex file. A
container without a machine ID gets no temporary CUID, and one whose image bakes
in a machine ID gets the same temporary CUIDs everywhere that image runs.
Temporary CUIDs are therefore node-local and unreliable in containers; a
container should use the driver CUID through sysfs.

### Virtualization

- SR-IOV VFs publish nothing. The library gives a VF its one-based VF index
  as UnitID where it can determine the index; a VF whose index it cannot
  determine, as in a guest, gets a temporary CUID rather than UnitID 0.
- Under full passthrough, the guest's amdgpu sees the physical serial and the
  guest's own variable store, so it creates its own key. Primary CUIDs match
  the host's; derived CUIDs differ unless the same key is set in the guest.
  The key belongs to a variable store, so this is intended.
- A guest without a runtime variable store publishes nothing and uses
  temporary CUIDs.

### Vectors

D-1 and AD-2 remain, as test-key vectors. New vectors cover the UnitID of every
mode on 8 XCCs and the 36-octet payload parse cases, and the A-* set is
regenerated for `K_app`.

## Risks and trade-offs

- **Local exposure before the variable is 0600.** See "Confidentiality".
- **A malformed variable blocks every CUID on the host** until an administrator
  deletes or rewrites it. Overwriting it automatically would silently change
  every derived CUID.
- **Re-keying changes every derived CUID on the host.** There is no mapping
  from old values to new ones.
- **Root and other users see different CPU, NIC and platform CUIDs.** Bit 117
  and the source say which one a caller got.

## Open questions

- The final temporary-CUID construction.
- The source from which the driver learns that SLC is active.
- Upstream acceptance of the efivarfs 0600 patch.
- Per-VM serials and a virtualized `AmdCuidKey` in hypervisors.
