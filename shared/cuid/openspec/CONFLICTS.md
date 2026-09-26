# Conflict register

Every place the published specification contradicts itself, contradicts a
standard it cites, or leaves a value an implementer must invent.

`specs/cuid/` records each one where it occurs, marked
`Recorded contradiction` / `Recorded defect` / `Recorded gap`. Every one is
resolved in `changes/amend-published-cuid-spec/`.

The baseline retains the published rule at each marked site. The corrected rule
lives in `changes/amend-published-cuid-spec/`.

`openspec/check_conflict_register.py` checks the *bookkeeping* of that mapping:
that every labelled marker has a row here, that every row names a
non-placeholder resolution, and that a short list of named constants is stated
consistently across the change dirs. It does not read the specs for meaning and
does not check that a resolution was implemented. A clean run is not a claim
that the corpus is consistent.

Source: "Persistent platform component identification for SW tools", version 84.

| ID | Conflict | Recorded in | Resolution |
|---|---|---|---|
| C1 | A firmware-supplied UUID is "used directly" as the PrimaryID, but every CUID must be a UUIDv8. An SMBIOS system UUID is version 1, 3 or 4, never 8. | `primary-identifier` | Two constructions, distinguished by the version nibble: **constructed** (payload, always v8, decodable) and **adopted** (firmware UUID verbatim, opaque). Consumers must not reject on the nibble. |
| C2 | The HMAC message is "the primary ID 122bit wide value". 122 bits is not a whole number of octets, so this does not name a byte string. | `derived-identifier` | The message is the sixteen packed octets, LSB first, padding bits present and zero. |
| C3 | The auxiliary serial must be "64bit wide" and the input structure is 256 bits; the reduction is never stated. | `auxiliary-fallback` | First 8 octets of unkeyed SHA-256 over the structure, little-endian. |
| C4 | The auxiliary input is described twice and incompatibly: an unordered field list with an `amd.com` namespace, and a fixed-width 256-bit structure. | `auxiliary-fallback` | The structure is the sole input. No namespace, no field list. |
| C5 | A DerivedID must be "constant for the lifetime of the lesser-privileged SW context", and is also a function of a salt that may change. | `identifier-model` | Constant **while the salt is unchanged**; re-keying is an administrative invalidation, and the association is recorded. |
| C6 | Scalability requires no collision "anywhere"; the fallback section says auxiliary inputs are "not guaranteed to be unique across cluster nodes". | `identifier-model` | The guarantee is scoped to canonical CUIDs. Auxiliary values are marked by bit 117 so a consumer can tell before relying on one. |
| C7 | The collision bound of 1/(2^(122-1)) assumes 122 varying bits; the derived payload carries at most 109. | `derived-identifier` | Bound restated over the 109 hash bits. No value changes. |
| C8 | The auxiliary Component Type field admits only GPU, NIC and NPU, while the primary layout defines sixteen types. | `auxiliary-fallback` | Full on-wire enumeration; the fallback is not a property of component type. |
| C9 | The CPU auxiliary table renames bits `200:215` from VendorID to FamilyID and splits Family/Model, disagreeing with both the PCIe structure and the primary layout. | `auxiliary-fallback` | One structure, one meaning per field. Device holds Family+Model combined, Vendor holds the vendor. |
| C10 | Bit 117 is UnitID part 2 in the primary table and the Auxiliary Value Identifier in the derived table. | `primary-identifier`, `derived-identifier` | UnitID narrows to `112:116`, 13 bits total. Bit 117 is the auxiliary marker in both layouts. |
| C11 | The derived hash slot is 45 bits wide, labelled `hash[64:109]` (46), inside prose saying 110. | `derived-identifier` | `hash[64:108]`, 45 bits, 109 hash bits total. The slot width was already right. |
| C12 | Auxiliary CUIDs are specified as UUIDv5 with an `amd.com` namespace, using HMAC-SHA-256 in place of SHA-1, which is not a conforming UUIDv5. | `auxiliary-fallback` | Withdrawn. Uniform UUIDv8 distinguished solely by bit 117. |
| C13 | The auxiliary input structure gives Format 17 bits and Machine ID 127; neither is a whole number of octets and the Machine ID cannot hold `/etc/machine-id`. | `auxiliary-fallback` | Format `0:15`, Machine ID `16:143`. |
| C14 | The fallback prose calls the Linux Machine ID a "32bit MachineID". `/etc/machine-id` is 128 bits. | `auxiliary-fallback` | Corrected to 128 bits. |
| C15 | The CPU auxiliary structure retains a PCIe Routing ID field for a component with no Bus/Device/Function. | `auxiliary-fallback` | The socket's physical package ID; zero on socket 0. |
| C16 | A constant seed key is permitted for auxiliary derivation but never given. | `auxiliary-fallback` | `AMD-CUID-TEMP-KEY-v1`, 20 ASCII octets, unpadded, key not message. Superseded by `adopt-uefi-key-store`: temporary CUIDs use a machine-id-keyed application key. |
| C17 | No canonical fallback seed is named, so an unprovisioned machine's derived CUID is undefined. | n/a | `AMD-CUID-DEFAULT-SEED-v1`, 24 ASCII octets, unpadded. Superseded by `adopt-uefi-key-store`: there is no fallback seed; an unprovisioned machine has temporary CUIDs. |
| C18 | The PCIe Device Serial Number's byte order is never stated. | `component-discovery` | Configuration-space order, little-endian, unswapped, from `dsn_cap_offset + 4`. |
| C19 | The NIC MAC fallback is permitted but its orientation is never stated. | `component-discovery` | Octet 0 at payload bits 0:7; an all-zero address is absent. |

C10, C12, C13 and C18 each produced a wrong value in shipped code. C1 produced
three: a Platform reported as an NPU, a firmware identity reported as
synthesised, and two platforms differing only in version bits deriving the same
derived CUID.

## Decided here, or still open: not registered above

Cross-layer decisions and implementation gaps are separate from baseline `Cn`
defects. `check_conflict_register.py` does not check this section.

### O1: UnitID for an SR-IOV Virtual Function

Historical positions:

| Position | Where | What it says |
|---|---|---|
| Baseline blesses a non-zero UnitID for a VF | `specs/cuid/primary-identifier/spec.md:76-80`, scenario "A subdivided function" | "**WHEN** a driver names a spatial partition or a Virtual Function of a physical device / **THEN** it assigns a non-zero UnitID rooted in the parent device definition" |
| The delta forbids it | `changes/pin-cuid-cross-layer-contract/specs/cuid/component-sources/spec.md:158-163`, "UnitID identifies a sub-unit, not a location" | UnitID "SHALL NOT carry a bus address, an **enumeration index**, or any other property of where the component is" |
| The shipped library does it | `lib/src/cuid_gpu.cc`, `CuidUtilities::get_gpu_vf_identity()` | A VF's UnitID is its 1-based VF index |

**Decided: the baseline wins, and nothing changes value.** See `changes/adopt-uefi-key-store/design.md`, "Virtualization".

A VF is a sub-unit of the card, like a spatial partition, and UnitID says which
sub-unit. The delta's prohibition is aimed at *location* (bus address, slot,
APIC ID); a VF index says which share of the parent this is, not where it is.
The wording was too broad, not the code: a VF carries its one-based VF index
as UnitID where the producer can determine it, and UnitID 0 names the
component as a whole. The library keeps its 1-based VF index, so no recorded
value moves.

The kernel publishes no CUID attributes on a VF.

In a guest, where there is no physfn link, the VF index cannot be determined.
Such a VF does not report UnitID 0 and does not use the card's serial, which is
the host's; it gets a temporary CUID.

### O2: an unprivileged `amd-smi` reports no CUID, with the value in sysfs — resolved

Resolved in the library, so the amd-smi sysfs fallback originally proposed is
unnecessary. With driver-published attributes, cold handle lookup enumerates before
attempting privileged single-device discovery, `get_derived_cuid()` reads the
driver's derived value first, and `AMDCUID_QUERY_SOURCE` reports the answering
stage. Without them, an unprivileged caller gets temporary CUIDs, and the
auxiliary flag comes from the derived value rather than the privileged primary.

### O3: restoring the persisted seed after module load — superseded

Superseded by `changes/adopt-uefi-key-store/`: the key persists in the
`AmdCuidKey` UEFI variable, which the driver reads at load, so nothing in user
space restores it.

### O4: what the library is supposed to enumerate — resolved for GPUs

`libamdcuid` enumerated every DRM card node as a GPU regardless of vendor, so a
node with a BMC display controller (for example ASPEED `1a03:2000`, driver
`ast`) got a GPU CUID for it, and `amdcuid_get_all_handles()` and
`amd-smi static --cuid` could report different device sets.

Resolved in `changes/adopt-uefi-key-store/` (`library-consumer`): GPU
discovery lists Vendor ID `0x1002` only.
