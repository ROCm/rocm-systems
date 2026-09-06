## Purpose

The last-resort path: what user-mode software does when no architectural serial
is reachable at all. The published specification calls the result an Auxiliary
CUID and defines it differently from every other CUID.

_Source: "Persistent platform component identification for SW tools", version 84._

## Requirements

### Requirement: When the fallback applies

Where the system provides no Device Serial Number, no Vendor-Specific Extension
Capability carrying a unique component ID, and the OS environment does not
permit access to the architectural hardware properties that would define a
serial, or where kernel services to provide a Derived CUID are absent,
**user-mode software** MAY construct a component identification value from other
accessible, potentially ambiguous data, to stand in for the device serial
number.

That permission is conditional on those sources being unreachable. Where an
architectural serial is reachable, software SHALL use it and SHALL NOT construct
an auxiliary value in its place: the fallback is explicitly "a scenario of last
resort to at least have the same support level as before the CUID has been
introduced, until system upgrades have been put in place".

#### Scenario: Kernel support is absent

- **WHEN** no kernel service publishes a CUID and no architectural serial is
  reachable
- **THEN** user-mode software may construct an auxiliary value

### Requirement: Auxiliary serial inputs

The auxiliary device serial number SHALL be a 64-bit value built by combining
Bus/Device/Function information (the Routing ID) with a system identifier value
(an SMBIOS system serial number, or the OS-provided Machine ID on Linux), and
SHALL be placed in the Device Serial Number field of the Primary CUID.

The resulting identifier is limited to local, system-level device
identification, because the inputs are not guaranteed unique across cluster
nodes or data centres.

> **Recorded defect (C14).** The prose describes the Linux Machine ID as a "32bit
> MachineID". `/etc/machine-id` is 128 bits.
>
> **Recorded gap (C3).** The auxiliary serial is required to be "64bit wide",
> and the input structure below is 256 bits, but the page never states the
> reduction from one to the other: which hash, over what, truncated how, in
> which byte order.
>
> Both resolved in `openspec/changes/amend-published-cuid-spec/`.

#### Scenario: The auxiliary value is node-local

- **WHEN** an auxiliary CUID is generated
- **THEN** it is not relied on for identification across cluster nodes

### Requirement: Auxiliary CUIDs use UUIDv5

An Auxiliary CUID SHALL use a variant of UUID version 5, generating the payload
with an HMAC-SHA-256 hash in place of the standard's SHA-1, over a namespace
string of `amd.com` together with the Vendor ID, Device ID, Revision ID, PCIe
Bus/Device/Function ID and the OS Machine ID.

> **Recorded contradiction (C4).** The auxiliary input is described twice and
> incompatibly: here as an unordered list of fields combined with an `amd.com`
> namespace string, and below as a fixed-width 256-bit structure. The page does
> not say which is the hash input, nor how the namespace and the structure
> relate.
>
> **Recorded contradiction (C12).** Every other CUID in this specification is a
> UUIDv8, and the Derived payload layout already carries an Auxiliary Value
> Identifier at bit 117 for exactly this purpose. A UUIDv5-typed value carrying
> an HMAC-SHA-256 payload is also not a conforming UUIDv5, which RFC 9562
> defines as SHA-1 over `namespace ‖ name`. Resolved in
> `openspec/changes/amend-published-cuid-spec/`.

#### Scenario: The version nibble differs from every other CUID

- **WHEN** an auxiliary CUID is rendered as published
- **THEN** its version nibble is `5`, not `8`

### Requirement: Auxiliary input structure, PCIe device

The input structure for a PCIe device SHALL be, little-endian unless otherwise
specified:

| Bit position | Name | Comments |
|---|---|---|
| 0–16 | Format | `1` = PCIe device, `2` = CPU, all other values reserved |
| 17–143 | Machine ID | From `/etc/machine-id` or equivalent OS ID |
| 144–175 | PCIe RoutingID | Segment/Bus/Device/Function from PCIe config space |
| 176–183 | RevisionID | From PCIe config space |
| 184–199 | DeviceID | From PCIe config space |
| 200–215 | VendorID | From PCIe config space |
| 216–219 | Component type | `2` = GPU, `3` = NIC, `4` = NPU |
| 220–255 | Reserved | Must be zero |

> **Recorded defect (C13).** The ranges total 256 bits, but Format is allocated 17
> bits (0–16) and Machine ID 127 bits (17–143). A Format field of 17 bits is not
> a whole number of octets, and a 127-bit Machine ID cannot hold the 128-bit
> `/etc/machine-id` the same row names. Both are off by one.
>
> **Recorded gap (C8).** The Component type row admits only `2` = GPU, `3` =
> NIC, `4` = NPU. The primary layout defines sixteen types. A Platform, a
> Storage device or a GenPCIe device has no auxiliary encoding at all.
>
> Both resolved in `openspec/changes/amend-published-cuid-spec/`.

#### Scenario: The reserved tail is zero

- **WHEN** the input structure is inspected
- **THEN** bits 220–255 are zero

### Requirement: Auxiliary input structure, CPU

The input structure for a CPU SHALL be:

| Bit position | Name | Comments |
|---|---|---|
| 0–16 | Format | `2` = CPU |
| 17–143 | Machine ID | From `/etc/machine-id` or equivalent OS ID |
| 144–175 | PCIe RoutingID | Segment/Bus/Device/Function from PCIe config space |
| 176–183 | RevisionID | From CPUID on x86, or other CPU detection mechanisms |
| 184–199 | ModelID | From CPUID on x86, or other CPU detection mechanisms |
| 200–215 | FamilyID | From CPUID on x86, or other CPU detection mechanisms |
| 216–219 | Component type | `1` for CPU |
| 220–255 | Reserved | Must be zero |

> **Recorded gap (C15).** This structure retains a PCIe Routing ID field for a CPU,
> which has no PCIe Bus/Device/Function of its own, and does not say what to put
> there.
>
> **Recorded contradiction (C9).** This table places FamilyID at `200-215`,
> which is the VendorID field in the PCIe structure and in the primary layout,
> and splits Family and Model across two 16-bit fields. The primary layout
> instead packs a CPU's "Family # and Model # values combined" into the single
> 16-bit DeviceID field. The two disagree about how CPU identity is laid out,
> and the CPU auxiliary structure carries no vendor at all.
>
> Both resolved in `openspec/changes/amend-published-cuid-spec/`.

#### Scenario: A CPU auxiliary value

- **WHEN** a CPU's auxiliary input structure is built
- **THEN** its Format is `2` and its Component type is `1`

### Requirement: Key handling for auxiliary derivation

Depending on OS security policy, a **constant seed key value** MAY be used,
where the use scenario allows the same Derived CUID to be shared across multiple
software contexts. That permission is conditional: where the use scenario does
not allow one Derived CUID to be shared across multiple software contexts, a
constant seed key value SHALL NOT be used.

Because the Auxiliary CUID and its Derived CUID are generated at the same
privilege level in the application context, sharing the Derived CUID needs no
further privilege differentiation.

The Auxiliary CUID SHOULD NOT be used in communication with remote application
contexts.

> **Recorded gap (C16).** The constant seed key value is permitted but never given.
> Resolved in `openspec/changes/amend-published-cuid-spec/`.

#### Scenario: A constant key is permitted

- **WHEN** policy allows sharing one derived value across contexts
- **THEN** a constant seed key may be used for the auxiliary derivation
