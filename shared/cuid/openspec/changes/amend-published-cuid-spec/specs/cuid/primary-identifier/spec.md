## MODIFIED Requirements

### Requirement: Primary payload layout for PCIe devices

The PrimaryID payload for a PCIe device SHALL be 122 bits, laid out as:

| Bit | Name | Comments |
|---|---|---|
| 0:63 | Device / Platform / Component / CPU serial number | As published |
| 64:71 | UnitID (part 1) | Low bits |
| 72:79 | RevisionID | |
| 80:95 | DeviceID | |
| 96:111 | VendorID | |
| **112:116** | **UnitID (part 2)** | **High bits: five, not six** |
| **117** | **Auxiliary Value Identifier** | **Named in this layout too, not only the derived one** |
| 118:121 | Component Type | |

UnitID SHALL therefore be **13 bits** in total, split across bits `64:71` and
`112:116`.

Bit 117 SHALL be the Auxiliary Value Identifier in the Primary layout and in the
Derived layout alike, and SHALL NOT be part of UnitID.

*The published table gives UnitID part 2 six bits, `112:117`, while the Derived
table gives bit 117 to the Auxiliary Value Identifier. Both cannot hold; see D-E1
in `design.md`. A producer that kept the bit for UnitID would set the auxiliary
marker on any device whose UnitID exceeded 8191.*

#### Scenario: UnitID is 13 bits

- **WHEN** a UnitID of `0x1FFF` is packed
- **THEN** it round-trips exactly
- **AND** bit 117 is unaffected

#### Scenario: A large UnitID does not set the auxiliary marker

- **WHEN** a component with a UnitID above 8191, such as `0x2000`, is presented
  to a producer
- **THEN** bit 117 remains clear, the packed UnitID being confined to its 13 bits
- **AND** the value does not silently become an auxiliary-marked identifier

*8191 is the largest UnitID the corrected 13-bit field holds, so `0x2000` is the
first value that reached bit 117 under the published `112:117` layout.*

#### Scenario: A serial shorter than 64 bits

- **WHEN** the architectural serial is narrower than 64 bits
- **THEN** it is MSB zero-extended into bits 0:63

#### Scenario: A serial longer than 64 bits

- **WHEN** the architectural serial is wider than 64 bits
- **THEN** it is truncated

#### Scenario: A component-provided UUID pre-empts the layout

- **WHEN** an ACPI device object or an SMBIOS entry already provides a UUID
  specific to the individual component
- **THEN** that UUID value is used as the PrimaryID directly

### Requirement: A component-provided UUID is adopted, not constructed

Where an ACPI device object or an SMBIOS entry already provides a UUID specific
to the individual component, that UUID SHALL be the PrimaryID, used verbatim.

Such an identifier is **adopted**. It SHALL retain the version and variant bits
its source wrote, which in practice are version 1, 3 or 4 and never 8. Its
payload fields (Component Type, Vendor, Device, Revision, UnitID and the
Auxiliary Value Identifier) SHALL NOT be decoded from it, because it has none:
the bits in those positions are whatever firmware wrote.

Where an adopted identifier is derived from, the HMAC message SHALL be the
sixteen octets of the UUID itself. A producer SHALL NOT de-frame it as though it
were a 122-bit payload first.

*The published text says "used directly" and separately requires every CUID to be
a UUIDv8. "Used directly" is the one worth keeping: the firmware value is already
unique, and an adopted identifier has no CUID payload at all, so the version
nibble does real work here by telling a consumer whether there are fields to
read.*

*Measured consequence of not stating this. De-framing real SMBIOS UUIDs as CUID
payloads yielded a Component Type of `0xB` (Reserved), `0x4` (NPU) and `0xF`
(Other) for three platforms and set the Auxiliary Value Identifier on one of
them, reporting a genuine firmware identity as synthesised. De-framing also drops
six of the UUID's bits, so two platforms whose system UUIDs differ only in
version and variant bits derive the same secondary CUID.*

#### Scenario: The firmware UUID is preserved exactly

- **WHEN** SMBIOS reports a system UUID
- **THEN** the Platform CUID is those sixteen octets, including the firmware's
  own version and variant bits

#### Scenario: No field is decoded from an adopted identifier

- **WHEN** a consumer holds a Platform CUID sourced from firmware
- **THEN** it reports the Component Type and the auxiliary marker as not
  applicable rather than decoding them

#### Scenario: Derivation hashes the whole UUID

- **WHEN** a secondary CUID is derived from an adopted primary
- **THEN** the HMAC message is the sixteen octets of the UUID
- **AND** two platforms differing only in version or variant bits derive
  different secondary CUIDs
