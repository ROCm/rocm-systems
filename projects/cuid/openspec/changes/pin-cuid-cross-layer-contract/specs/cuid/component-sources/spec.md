## Purpose

States, per component type, what goes into payload bits 0:63 and in what
precedence, so that two producers looking at the same machine reach the same
serial rather than each picking a defensible but different source.

## ADDED Requirements

### Requirement: Serial precedence for PCIe components

For a GPU, NIC or NPU, the value in payload bits 0:63 SHALL be selected in this
order, stopping at the first source that yields a non-zero value:

1. A serial number supplied by the component's driver, where one exists.
2. The PCIe Device Serial Number extended capability.
3. A vendor-specific capability defined for the component, where one exists.

A source that yields zero SHALL be treated as absent and SHALL NOT be used. Zero
is what an unimplemented capability and an unpopulated register both read as; a
producer that accepts it gives every affected component on the machine the same
CUID.

If no source yields a non-zero value, the component has no canonical primary. A
kernel producer SHALL publish nothing for it; a user-mode producer MAY fall back
to an auxiliary CUID.

#### Scenario: An all-zero Device Serial Number is rejected

- **WHEN** the PCIe Device Serial Number capability reads as all zeroes
- **THEN** the producer continues to the next source rather than packing zero

#### Scenario: Two serial-less components do not collide

- **WHEN** two components on one machine have no reachable serial
- **THEN** neither is published with a canonical primary

### Requirement: The Device Serial Number is used in configuration-space order

The PCIe Device Serial Number capability holds two 32-bit dwords, the first of
which is the low half. The eight octets are therefore a little-endian 64-bit
value, and payload bits 0:63 SHALL carry that value as read. A producer SHALL
NOT byte-swap it, and SHALL NOT decode it by copying the octets over a
host-order integer.

The capability's serial dwords begin four octets after the extended-capability
header, so a producer SHALL read from `dsn_cap_offset + 4`.

*The library byte-swapped and the kernel did not. Both read the same eight octets
off the same card and packed two byte-reversed serials, so the two layers named
it with two entirely unrelated CUIDs, neither detectably wrong on its own.*

#### Scenario: Kernel and library agree on a DSN-sourced serial

- **WHEN** a component's serial comes from the PCIe Device Serial Number
  capability rather than from a driver
- **THEN** the kernel and the library pack the same 64-bit value into payload
  bits 0:63
- **AND** the two primary CUIDs are byte-identical

#### Scenario: The low dword is the low half

- **WHEN** the DSN capability's first serial dword is `0xD3AAABD4` and its
  second is `0x06C5349B`
- **THEN** payload bits 0:63 read `0x06C5349BD3AAABD4`

### Requirement: A NIC may fall back to its permanent MAC address

Where a NIC exposes neither a driver serial, nor a Device Serial Number, nor a
vendor-specific capability, a user-mode producer MAY use the interface's
permanent MAC address as the value in payload bits 0:63, with octet 0 of the
address at payload bits 0:7 and the remaining two octets zero.

An all-zero address SHALL be treated as absent. A producer SHALL NOT use an
administratively assigned address in preference to the permanent one where both
are available.

This source is user-mode only. It is listed here rather than left to each
producer because a MAC address is a burned-in per-device value that would
otherwise be reached for independently, in two different orientations, by anyone
who needed a NIC identity. That is how the Device Serial Number came to be
byte-swapped in one layer and not the other.

#### Scenario: The address orientation is fixed

- **WHEN** the permanent MAC address is `02:00:00:aa:bb:cc`
- **THEN** payload bits 0:7 read `0x02`

#### Scenario: An unconfigured interface has no serial

- **WHEN** the permanent MAC address reads as all zeroes
- **THEN** the producer treats the NIC as having no reachable serial

### Requirement: Identity register widths

Vendor ID and Device ID SHALL be read as 16-bit values from PCI configuration
space at their defined offsets, in configuration-space byte order, with no
byte-swap applied.

Revision ID SHALL be read as the single octet at offset `0x08`. The octet at
`0x09` is the programming interface and SHALL NOT be folded into it.

*Both mistakes have shipped: a producer that byte-swapped a correctly-read
little-endian value emitted vendor `0x0210` for AMD's `0x1002`, and a producer
that read two octets at `0x08` into an 8-bit field kept the programming
interface and discarded the revision.*

#### Scenario: AMD's vendor ID is emitted as 0x1002

- **WHEN** the primary payload of an AMD component is decoded
- **THEN** payload bits 96:111 read `0x1002`

#### Scenario: Revision ID excludes the programming interface

- **WHEN** the primary payload of a component whose programming interface is
  non-zero is decoded
- **THEN** payload bits 72:79 read the revision alone

### Requirement: Platform CUID uses the SMBIOS UUID verbatim

Where the platform supplies a system UUID through SMBIOS or ACPI, the Platform
CUID SHALL be those 16 octets used directly, in the order the firmware presents
them, with no reframing, no component type, no vendor or device fields and no
derivation applied to the primary.

A producer SHALL NOT collapse the system UUID to a narrower value and pack it
through the normal layout. Doing so discards half of an identifier that the
firmware has already made unique, and makes the Platform CUID depend on the
producer's fold rather than on the platform.

Where no system UUID is defined but a system serial number is, the Platform CUID
SHALL be constructed through the normal 122-bit layout with Component Type
`0x0`, UnitID, Revision, Device and Vendor zero, and payload bits 0:63 set to
the first 8 octets of the unkeyed SHA-256 digest of the serial number string,
interpreted little-endian.

Where neither is available the platform has no CUID.

#### Scenario: The system UUID is passed through unchanged

- **WHEN** SMBIOS reports a system UUID
- **THEN** the Platform CUID string is that UUID
- **AND** all 16 octets are preserved, including the firmware's own version and
  variant bits

#### Scenario: A serial-only platform is constructed through the normal layout

- **WHEN** SMBIOS reports no system UUID but does report a system serial number
- **THEN** the Platform CUID is a UUIDv8 with Component Type `0x0`

#### Scenario: A sentinel UUID is not accepted

- **WHEN** SMBIOS reports a system UUID whose octets are all `0x00` or all `0xFF`
- **THEN** the producer treats it as absent and takes the serial-number branch

### Requirement: UnitID identifies a sub-unit, not a location

UnitID SHALL carry a component's partition or sub-unit index, and SHALL be zero
for a component that is not partitioned. It SHALL NOT carry a bus address, an
enumeration index, or any other property of where the component is.

UnitID is 13 bits, split across payload bits 64:71 and 112:116.

#### Scenario: An unpartitioned component reports UnitID zero

- **WHEN** the primary payload of a NIC or NPU is decoded
- **THEN** UnitID reads zero

#### Scenario: Partitions of one device are distinguishable

- **WHEN** a partitioned GPU exposes eight partitions
- **THEN** the eight primary CUIDs differ only in the UnitID field
