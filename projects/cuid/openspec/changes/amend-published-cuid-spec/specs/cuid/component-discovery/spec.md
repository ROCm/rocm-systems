## MODIFIED Requirements

### Requirement: PCIe and CXL device serial number

Unchanged in source and location. The **orientation** SHALL additionally be
specified.

The Device Serial Number capability holds two 32-bit dwords, the first of which
is the low half, beginning four octets after the extended-capability header. The
eight octets are therefore a little-endian 64-bit value, and payload bits 0:63
SHALL carry that value **as read**.

A producer SHALL NOT byte-swap it, and SHALL NOT decode it by copying the octets
over a host-order integer.

*The published text gives the capability ID and the width but never the byte
order, so each implementation chose. One byte-swapped and one did not; both read
the same eight octets off the same card and packed two byte-reversed serials, and
neither value was detectably wrong on its own.*

#### Scenario: The low dword is the low half

- **WHEN** the first serial dword is `0xD3AAABD4` and the second is `0x06C5349B`
- **THEN** payload bits 0:63 read `0x06C5349BD3AAABD4`

#### Scenario: The serial is read from the DSN capability

- **WHEN** a PCIe endpoint exposes Extended Capability ID `0003h`
- **THEN** its 64-bit Device Serial Number is used in payload bits 0:63

### Requirement: Permitted substitutes for an absent Device Serial Number

Unchanged in precedence. The **MAC address orientation** SHALL additionally be
specified.

Where a NIC's MAC address is used in place of a Device Serial Number, octet 0 of
the address SHALL occupy payload bits 0:7, with the remaining two octets of the
64-bit field zero.

An all-zero address SHALL be treated as absent. A producer SHALL NOT use an
administratively assigned address in preference to the permanent one where both
are available.

A source that yields zero SHALL be treated as absent generally: zero is what an
unimplemented capability and an unprogrammed register both read as, and a
producer that accepts it gives every affected component on the machine the same
CUID.

#### Scenario: A NIC without a DSN

- **WHEN** a network adapter exposes no Device Serial Number
- **THEN** its MAC address may be used in place of it

#### Scenario: The address orientation is fixed

- **WHEN** the permanent MAC address is `02:00:00:aa:bb:cc`
- **THEN** payload bits 0:7 read `0x02`

#### Scenario: An all-zero Device Serial Number is rejected

- **WHEN** the DSN capability reads as all zeroes
- **THEN** the producer continues to the next source rather than packing zero
