## MODIFIED Requirements

### Requirement: Auxiliary CUIDs use UUIDv5

**Withdrawn and replaced.** An Auxiliary CUID SHALL use the same 122-bit payload
layout, the same component type numbering, the same UUIDv8 framing and the same
derivation as any other CUID, and SHALL be distinguished **solely by bit 117**,
the Auxiliary Value Identifier, being set.

A producer SHALL NOT emit an auxiliary CUID with a version nibble other than
`8`, and SHALL NOT use the version nibble to signal that a value is auxiliary.

The `amd.com` namespace string, the namespace form, and the question of HMAC
operand order for the auxiliary path are all **void**: under uniform UUIDv8
there is no namespace.

Three consequences are normative:

- The format does not use UUIDv5. RFC 9562 defines version 5 as SHA-1 over
  `namespace ‖ name`; the CUID payload is HMAC-SHA-256 over a defined structure,
  so a producer SHALL NOT emit a constructed CUID claiming version 5. A consumer
  still SHALL NOT reject a value on its version nibble: as `identifier-model`
  states, an adopted firmware-supplied UUID legitimately carries version 1, 3, 4
  or 5 and is used verbatim.
- A consumer parses every constructed CUID payload with one code path and one
  mask. It MAY read the version nibble to tell an adopted UUID from a
  constructed payload, but SHALL NOT treat an unrecognised version as a reason
  to discard the value.
- The same device presents the same UUID version regardless of the privilege of
  the caller or the host it is enumerated on. Whether a serial was reachable is a
  property of the environment, not of the device.

*The Derived payload table already carries an Auxiliary Value Identifier at bit
117 for exactly this purpose, so the UUIDv5 path is redundant as well as
non-conforming.*

#### Scenario: A consumer identifies an auxiliary value

- **WHEN** a consumer reads any CUID
- **THEN** it tests payload bit 117
- **AND** the version nibble is `8` in both cases

#### Scenario: The version nibble differs from every other CUID

*Carried from the baseline with its outcome reversed. The heading is kept so the
loss check can pair the two.*

- **WHEN** an auxiliary CUID is rendered under this amendment
- **THEN** its version nibble is `8`, not `5`, and so does **not** differ from
  every other CUID
- **AND** the value that distinguishes it is payload bit 117, not the nibble

### Requirement: Auxiliary input structure, PCIe device

The input structure SHALL be 256 bits, packed LSB-first into 32 octets:

| Bit | Width | Field |
|---|---|---|
| **0:15** | **16** | **Format**: `1` = PCIe device, `2` = CPU, others reserved |
| **16:143** | **128** | **Machine ID** |
| 144:175 | 32 | PCIe Routing ID |
| 176:183 | 8 | RevisionID |
| 184:199 | 16 | DeviceID |
| 200:215 | 16 | VendorID |
| 216:219 | 4 | Component Type, on-wire numbering |
| 220:255 | 36 | Reserved, SHALL be zero |

The Machine ID SHALL be 128 bits: on Linux, the 32 hexadecimal characters of
`/etc/machine-id` decoded to 16 octets, octet `k` occupying structure bits
`16 + 8k` through `23 + 8k`. A producer that cannot obtain a machine identity
SHALL NOT emit an auxiliary identifier at all. Packing zero there instead would
give two identically configured hosts the same auxiliary identifier for their
corresponding components.

The PCIe Routing ID SHALL be `(segment << 16) | (bus << 8) | (device << 3) |
function`.

Component Type SHALL use the on-wire numbering, the same values as the Primary
payload's Component Type field, rather than the restricted set the published
table lists.

A producer SHALL NOT derive the structure from a formatted string, and SHALL NOT
filter, normalise or otherwise transform the field values before packing them.

*The published ranges total 256, so the error is not arithmetic: Format is
allocated 17 bits and Machine ID 127, and neither is a whole number of octets. A
127-bit Machine ID also cannot hold the 128-bit `/etc/machine-id` the same row
names. Moving the boundary by one bit fixes both. See
`pin-cuid-cross-layer-contract/specs/cuid/auxiliary-identifier` for the
string-derived-input prohibition.*

#### Scenario: Both fields are whole octets

- **WHEN** the Format and Machine ID field widths are measured
- **THEN** they are 16 and 128 bits

#### Scenario: Two functions of one device differ

- **WHEN** the auxiliary serial is computed for `0000:65:00.0` and
  `0000:65:00.1` on the same machine
- **THEN** the two Routing ID fields differ, and so do the two serials

#### Scenario: The reserved tail is zero

- **WHEN** the input structure is inspected
- **THEN** bits 220–255 are zero

### Requirement: Auxiliary input structure, CPU

There SHALL be **one** input structure, not two. The CPU case SHALL use the same
field positions and the same per-field meanings as the PCIe case, with Format
`2` and Component Type `1`, and:

- **RevisionID** (`176:183`) SHALL hold the CPUID stepping;
- **DeviceID** (`184:199`) SHALL hold the CPU's Family and Model combined, the
  same way the primary payload defines a CPU's DeviceID;
- **VendorID** (`200:215`) SHALL hold the CPU vendor ID;
- the **PCIe Routing ID field** (`144:175`) SHALL be **zero**.

*A CPU has no Bus/Device/Function of its own, and zero is the only Routing ID
two producers can agree on. The published CPU table also renames bits `200:215`
from VendorID to FamilyID and splits Family and Model (C9), leaving a CPU
auxiliary value with no vendor at all.*

#### Scenario: A CPU carries no routing ID

- **WHEN** a CPU's auxiliary input structure is built
- **THEN** bits 144:175 are zero

#### Scenario: A CPU auxiliary value

- **WHEN** a CPU's auxiliary input structure is built
- **THEN** its Format is `2` and its Component type is `1`

### Requirement: Key handling for auxiliary derivation

The constant seed key the published text permits SHALL be exactly the 20-octet
ASCII string `AMD-CUID-TEMP-KEY-v1`, with no terminating NUL and no padding.

The derivation SHALL use the same operand order as every other CUID derivation:
`HMAC-SHA-256(key = the temporary fixed key, message = the 16 auxiliary primary
octets)`. A producer SHALL NOT exchange the operands and SHALL NOT substitute a
fixed constant for the message.

The key is public by construction: an auxiliary CUID is built from
non-privileged information precisely so that unprivileged and out-of-band
consumers can reproduce it, which they cannot do with a secret.

*One producer swapped the operands and passed a fixed application UUID as the
message. The derivation reads bit 117 out of whatever it is handed as the
primary, so with a constant there the derived value was never marked auxiliary.*

#### Scenario: A constant key is permitted

*Carried from the baseline with the key pinned: the permission survives, the
freedom to choose which constant does not.*

- **WHEN** policy allows sharing one derived value across contexts
- **THEN** the constant seed key `AMD-CUID-TEMP-KEY-v1` may be used for the
  auxiliary derivation
- **AND** no other constant may be substituted for it

#### Scenario: The auxiliary marker survives derivation

- **WHEN** a derived CUID is computed from an auxiliary primary
- **THEN** payload bit 117 of the derived value is set

#### Scenario: Independently verifiable

- **WHEN** `openssl dgst -sha256 -mac HMAC -macopt key:AMD-CUID-TEMP-KEY-v1` is
  run over the 16 auxiliary primary octets
- **THEN** it prints the derivation's digest

### Requirement: The reduction from the input structure to the serial

The auxiliary serial SHALL be the **first 8 octets of the unkeyed SHA-256 digest
of the 32-octet input structure, interpreted as a little-endian 64-bit value**,
and SHALL be placed in payload bits 0:63 of an otherwise normal primary payload
with bit 117 set.

*The published text requires a "64bit wide device serial number" and defines a
256-bit input structure, but never states how one becomes the other. Which hash,
truncated from which end, in which byte order are all choices, and a producer
that chose differently at any of the three emits a value sharing no bits.*

#### Scenario: The serial is a truncated digest

- **WHEN** the auxiliary serial is computed
- **THEN** it is the low 8 octets of SHA-256 over the input structure, read
  little-endian

### Requirement: The input structure is the sole hash input

The 32-octet input structure SHALL be the only input to the auxiliary serial.
There SHALL be no namespace string, and no separate field list.

*The published text describes the input twice: as an unordered field list with an
`amd.com` namespace, and as the fixed-width structure. The structure wins because
it is fully specified; the list gives neither an order nor an encoding.*

#### Scenario: One input, fully positioned

- **WHEN** two producers build the auxiliary serial for the same device
- **THEN** they hash byte-identical 32-octet structures

### Requirement: Auxiliary Component Type uses the full enumeration

The Component Type field of the input structure SHALL use the same on-wire
numbering as the primary payload, all sixteen values.

*The published table admits only `2` = GPU, `3` = NIC and `4` = NPU, so a
Platform, Storage, Memory or GenPCIe device has no auxiliary encoding at all.*

#### Scenario: Every component type can take the fallback

- **WHEN** a Storage device has no reachable serial
- **THEN** its auxiliary input structure carries Component Type `5`
