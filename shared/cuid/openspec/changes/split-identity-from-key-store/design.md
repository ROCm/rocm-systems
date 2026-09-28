## Context

The only secret in CUID is the node key. Everything the driver can compute
from hardware alone (serial, PCI IDs, XCC range) is either privileged
(`cuid_primary`) or public (`cuid_unit_id`) and needs no storage.

## Decisions

### Split at the key, not at the layer

Both the kernel and the library split at the same line: everything keyed goes
into the second series. Each series builds, tests and reviews on its own.

### Driver-published marker is `cuid_unit_id`

Before the split the library treated a device as driver-published when
`cuid_derived` existed. This series has no `cuid_derived`, so the marker
becomes `cuid_unit_id`, which both series publish at 0444 on every whole GPU
and partition. One library then serves either kernel. amd-smi looks a
partition up through its node by the same marker, so a partition never falls
back to the whole card's CUID on a kernel that names it.

### A GPU without a key is temporary, like every other component

A GPU's derived CUID needs the node key exactly as a CPU's does. Without one,
the library gives a whole GPU the temporary CUID it already built for a GPU
the driver does not name, for root and every other caller, so the value does
not depend on who asks. Root still reads the driver's `cuid_primary`.

### A GPU handle in SPX reports the whole GPU

On a GPU that publishes a partition node in every compute mode, such as an
MI300-series GPU, each amd-smi GPU handle is a compute partition, SPX included.
Without a node key a partition has no CUID, so every such handle would report
none. In SPX the one partition covers every XCC and is the same hardware as
the whole GPU, so the handle reports the whole GPU's CUID, source and primary
CUID. In DPX and above it reports none, because the whole GPU's CUID would name
every partition alike. amd-smi reads the mode from
`<render device>/current_compute_partition`.

### A NIC's UnitID is its PCI function number

The functions of a multi-port NIC can report one PCIe Device Serial Number,
and with UnitID 0 on every function they shared a primary and derived CUID.
The specification lets a driver define UnitID values for subdivided functions
rooted in the parent device, as GenPCIe takes the VF, so a NIC's UnitID is its
PCI function number, in the primary and the temporary primary alike. Function
0 keeps UnitID 0, so a single-function NIC is unchanged. Every NIC function is
then its own component, whatever the caller's privilege.

A partition's node has no BDF of its own. amd-smi's component list gives it the
address amd-smi uses for that partition's handle, as `amd-smi list` shows it,
with the partition index in the function number.

### The node key changes nothing outside amdgpu

`add-volatile-node-key` keeps the key only in amdgpu's memory and needs no
change outside amdgpu. Keeping it across reboots needs a store other users
cannot read; a UEFI variable is one, once efivarfs lists it 0600, and is
proposed separately as `adopt-uefi-key-store`.

### Library

- A GPU or partition is driver-published when `cuid_unit_id` exists in its
  directory: `/sys/bus/pci/devices/<bdf>` for a whole GPU, the partition's
  `xcp` directory for a partition. Its primary CUID is the driver's
  `cuid_primary`; a `cuid_primary` without `cuid_unit_id` counts as nothing
  published.
- Without a key every component is temporary: bit 117 set, source `LIBRARY`.
  A partition is refused, because a partition has no temporary identity. Only
  AMD GPUs (vendor 0x1002) are listed, so a BMC's display adapter gets no CUID.
- Removed: the default seed, the `/etc/amdcuid` key file and its lock, the
  `/var/lib/amdcuid` record store, the daemon with its service, udev and
  maintainer-script hooks, IPC, driver-seed reconciliation, `amdcuid_tool`, and
  the TPM-sealed variable script. `amdcuid_set_hash_key()` and
  `amdcuid_get_key_info()` stay in the ABI and return
  `AMDCUID_STATUS_UNSUPPORTED`.
- Reported sources are `DRIVER` and `LIBRARY`.

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
- **`cuid_unit_id`** (0444, decimal) is published beside every `cuid_primary`,
  0 on the device. It is how an unprivileged reader tells which partition an
  ID names.
- The library does not compute partition UnitIDs itself: without the driver
  there is no partition CUID to produce.

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

**Containers.** The library reads `/etc/machine-id`, and
`/var/lib/dbus/machine-id` only when `/etc/machine-id` does not exist. It
refuses a missing file and any content but 32 hex digits that are not all zero,
which machine-id(5) forbids; an `/etc/machine-id` that exists but is empty,
`uninitialized` or all zero belongs to an image whose identity is not set yet,
so it is not replaced by a D-Bus ID that came with the same image. A
container without a machine ID gets no temporary CUID, and one whose image bakes
in a machine ID gets the same temporary CUIDs everywhere that image runs.
Temporary CUIDs are therefore node-local and unreliable in containers.

### Virtualization

- SR-IOV VFs publish nothing. The library gives a VF its one-based VF index
  as UnitID where it can determine the index; a VF whose index it cannot
  determine, as in a guest, gets a temporary CUID rather than UnitID 0.
- Under full passthrough, the guest's amdgpu sees the physical serial, so
  primary CUIDs match the host's.

### Vectors

D-1 and AD-2 remain, as test-key vectors. New vectors cover the UnitID of every
mode on 8 XCCs, and the A-* set is regenerated for `K_app`.

## Risks

- Until the node key lands there is no stable identity across hosts for any
  caller; temporary CUIDs are node-local.
- Until the node key lands no caller gets a partition CUID from the library.

## Open questions

- The final temporary-CUID construction.
- The source from which the driver learns that SLC is active.
