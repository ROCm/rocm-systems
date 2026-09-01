## Purpose

The Component Unified ID wire format: how a component's hardware identity is
packed into a 122-bit payload, how that payload is rendered as an RFC 9562
UUIDv8, and how a secondary identifier is derived from it. Every producer must
emit byte-identical values for the same inputs, so this is a frozen ABI shared
between the kernel driver and the userspace library.

## ADDED Requirements

### Requirement: Primary payload layout

The primary CUID payload SHALL be exactly 122 bits, packed LSB-first into 16
octets, with the following field positions:

| Bits | Width | Field |
|---|---|---|
| 0:63 | 64 | Serial number (Device Serial Number) |
| 64:71 | 8 | UnitID part 1 (low bits) |
| 72:79 | 8 | Revision ID |
| 80:95 | 16 | Device ID |
| 96:111 | 16 | Vendor ID |
| 112:116 | 5 | UnitID part 2 (high bits) |
| 117 | 1 | Auxiliary Value Identifier |
| 118:121 | 4 | Component Type |

Bits 122:127 of the 16-octet buffer SHALL be zero padding.

UnitID is therefore 13 bits total, split across two ranges. Bit 117 is the
Auxiliary Value Identifier in both the primary and the derived layout; it is
**not** part of UnitID, which is why UnitID part 2 ends at bit 116.

#### Scenario: A GPU with a genuine serial number

- **WHEN** a component is packed with serial `0x0123456789ABCDEF`, vendor ID
  `0x1002`, device ID `0x73A3`, revision `0x00`, UnitID `0` and component type
  GPU
- **THEN** octets 0:7 carry the serial in little-endian order, octet 9 carries
  the revision, octets 10:11 the device ID and octets 12:13 the vendor ID, all
  little-endian
- **AND** the component type occupies bits 118:121, spanning the octet boundary:
  its low two bits are the top two bits of octet 14 and its high two bits are
  the low two bits of octet 15

#### Scenario: Component type values above 3 are not truncated

- **WHEN** a component is packed with component type NPU (`0x4`)
- **THEN** payload bits 120:121 carry the value's high two bits
- **AND** the emitted type reads back as `0x4`, not `0x0`

*Two independent implementations dropped bits 120:121 and rendered the component
type modulo 4, so an NPU collided with a Platform.*

### Requirement: Component type numbering

Component Type SHALL use the specification's on-wire numbering:

| Value | Type |
|---|---|
| 0x0 | Platform |
| 0x1 | CPU |
| 0x2 | GPU |
| 0x3 | NIC |
| 0x4 | NPU |
| 0x5 | Storage |
| 0x6 | Memory |
| 0x7 | Generic PCIe device |
| 0x8 | Generic component |
| 0x9 | Rack tray |
| 0xA | Rack |
| 0xB–0xE | Reserved |
| 0xF | Other |

Any enumeration exposed by a producer to its own callers SHALL use these same
values, so that the enumeration can be written to the wire directly without a
translation step.

#### Scenario: A GPU is emitted as type 2

- **WHEN** the CUID for a GPU is generated
- **THEN** the Component Type field contains `0x2`

*A producer whose enumeration is offset by one emits `0x3` here, which a
conforming reader interprets as a NIC. This has shipped.*

### Requirement: UUIDv8 framing and octet order

A 122-bit payload SHALL be rendered as a 128-bit RFC 9562 UUIDv8. The six
framing bits SHALL be **inserted**, not overwritten: the version nibble at
rendered bits 48:51 and the two variant bits at 64:65 displace the payload after
them rather than replacing payload content.

Numbering the rendered value MSB-first over its octets, the result SHALL be:

| Bits | Content |
|---|---|
| 0:47 | Payload part 1, the payload's least significant bits |
| 48:51 | Version, always `8` |
| 52:63 | Payload part 2 |
| 64:65 | Variant, always `10b` |
| 66:127 | Payload part 3, the payload's most significant bits |

The least significant payload octet therefore appears at the **front** of the
printed UUID string. The six bits displaced off the end are payload bits
122:127, which are always zero, so the framing is lossless and the payload can
be recovered exactly.

#### Scenario: Payload orientation is not mirrored

- **WHEN** the payload `d4abaad39b34c5060000a37302108000` (little-endian octet
  order) is framed
- **THEN** the rendered UUID is `d4abaad3-9b34-8c50-9800-028dcc084200`

*The opposite direction yields `2004009c-e8c0-8000-86c5-349bd3aaabd4`, a
different identifier for the same component. Both have shipped; only the first
is conforming.*

#### Scenario: Framing bits are correct per RFC 9562

- **WHEN** any CUID is rendered
- **THEN** the high nibble of octet 6 is `0x8`
- **AND** the top two bits of octet 8 are `10b`

#### Scenario: Round-trip

- **WHEN** a rendered UUIDv8 is parsed back by removing the version and variant
  bits
- **THEN** the recovered 122-bit payload is bit-for-bit identical to the input
  payload

### Requirement: Secondary identifier derivation

The secondary (derived) CUID SHALL be computed as
`HMAC-SHA256(key = seed, message = the 16 packed primary payload octets)`, where
the primary payload is the message and the shared secret is the key.

The 256-bit digest SHALL be folded into a derived payload as:

| Payload bits | Content |
|---|---|
| 0:63 | digest bits 0:63 |
| 64:71 | Reserved, zero |
| 72:116 | digest bits 64:108 (45 bits) |
| 117 | Auxiliary Value Identifier, copied from the primary |
| 118:121 | Reserved, zero |

The derived payload SHALL then be framed as a UUIDv8 by the same rule as the
primary.

The derived slot is 45 bits, not 46: bit 117 is reserved for the Auxiliary Value
Identifier in both layouts.

#### Scenario: Two producers agree

- **WHEN** the kernel driver and the userspace library are given the same
  component identity and the same seed
- **THEN** both emit byte-identical primary CUIDs
- **AND** both emit byte-identical secondary CUIDs

#### Scenario: The secondary discloses neither the serial nor the seed

- **WHEN** a consumer holds a secondary CUID and knows the component's vendor,
  device, revision and type
- **THEN** recovering the serial number or the seed requires breaking
  HMAC-SHA256

#### Scenario: The same component on two nodes with the same seed

- **WHEN** the same physical component is enumerated on two nodes provisioned
  with the same seed
- **THEN** both nodes report the same secondary CUID

### Requirement: Auxiliary Value Identifier

Bit 117 SHALL indicate that the identifier was constructed from a synthesised
auxiliary identity rather than a genuine hardware serial number. It SHALL be set
in the primary payload and carried unchanged into the derived payload.

Auxiliary CUIDs SHALL be generated by user-mode software only, as a last resort
when kernel infrastructure is absent. A kernel producer SHALL NOT set bit 117:
where it has a serial it emits a genuine identifier, and where it does not it
emits nothing.

All CUIDs, auxiliary or not, SHALL be rendered as UUIDv8. Producers SHALL NOT
use a different UUID version to distinguish auxiliary values.

#### Scenario: A consumer distinguishes an auxiliary value

- **WHEN** a consumer reads any CUID
- **THEN** it determines whether the value is auxiliary by masking bit 117
- **AND** it does not need to branch on the UUID version nibble, which is always
  `8`

#### Scenario: A kernel-produced CUID is never auxiliary

- **WHEN** a CUID is read from a kernel interface
- **THEN** bit 117 is clear

### Requirement: Payload stability

For a given component, the primary CUID SHALL be identical across reboots,
driver reloads, and topology changes such as reslotting or renumbering, provided
the component's serial number and identity registers are unchanged. The
secondary CUID SHALL be identical under the same conditions provided the seed is
also unchanged.

The identifier SHALL NOT depend on PCI bus/device/function, DRM render node
number, enumeration order, or any other value that is a property of where the
component is rather than which component it is.

#### Scenario: Component moved to a different slot

- **WHEN** a component is moved to a different PCI slot and the machine is
  rebooted
- **THEN** its primary and secondary CUIDs are unchanged
