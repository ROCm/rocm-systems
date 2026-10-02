## Purpose

The normative worked examples for the CUID format, and the obligation on every
producer to reproduce them. Every defect found so far would have been caught by a
single vector.

## ADDED Requirements

### Requirement: Producers reproduce the normative vectors

Every producer of a CUID SHALL reproduce each vector below bit-for-bit from the
stated inputs, and SHALL verify that it does so as part of its automated tests.

A vector is normative in all three of its columns: the 16-octet payload, the full
HMAC digest where one is involved, and the rendered UUID string. A producer that
matches the UUID but not the payload has an offsetting pair of errors and will
diverge on the next field that changes.

All payloads are written as the 16 octets in buffer order, octet 0 first. All
UUID strings are the standard 8-4-4-4-12 rendering of the 16 framed octets in
order. Every HMAC digest is reproducible with `openssl dgst -sha256 -mac HMAC`
over the payload octets, but the key option differs by vector. A vector keyed
with an ASCII constant, which is every vector except D-2, uses
`-macopt key:<ascii-key>`. D-2 is keyed with 32 binary octets, `00` through
`1f`, which `key:` cannot express, so it uses
`-macopt hexkey:000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f`.

#### Scenario: The vector suite passes in every producer

- **WHEN** a producer's test suite runs
- **THEN** every vector in this specification is checked
- **AND** a mismatch fails the build

### Requirement: Primary payload vectors

A producer SHALL reproduce each primary vector below exactly, in both its payload
octets and its rendered UUID string.

**P-1**: a Radeon Pro W6800, hardware-verified on `0000:63:00.0`.

| Input | Value |
|---|---|
| Serial | `0x06C5349BD3AAABD4` |
| UnitID | `0` |
| Revision ID | `0x00` |
| Device ID | `0x73A3` |
| Vendor ID | `0x1002` |
| Component Type | `0x2` (GPU) |
| Auxiliary bit | `0` |

```
payload : d4abaad39b34c5060000a37302108000
uuid    : d4abaad3-9b34-8c50-9800-028dcc084200
```

**P-2**: the second W6800 in the same machine, `0000:03:00.0`, all other inputs
as P-1.

| Input | Value |
|---|---|
| Serial | `0x8E8C71777252EBFF` |

```
payload : ffeb527277718c8e0000a37302108000
uuid    : ffeb5272-7771-88c8-b800-028dcc084200
```

**U-1**: UnitID `0x0123`, exercising the split across payload bits 64:71 and
112:116. All other inputs as P-1.

```
payload : d4abaad39b34c5062300a37302108100
uuid    : d4abaad3-9b34-8c50-988c-028dcc084204
```

Octet 8 holds `0x23`, the low eight bits; octet 14 holds `0x01` in its low five
bits, the high five.

#### Scenario: P-1 reproduces

- **WHEN** a producer packs the P-1 inputs
- **THEN** the 16 payload octets are `d4abaad39b34c5060000a37302108000`
- **AND** the rendered UUID is `d4abaad3-9b34-8c50-9800-028dcc084200`

#### Scenario: The payload is not mirrored

- **WHEN** the P-1 payload is framed
- **THEN** the rendered UUID begins `d4abaad3`, the least significant payload
  octets

*The opposite direction yields `2004009c-e8c0-8000-86c5-349bd3aaabd4`, a
different identifier for the same component. It has shipped; only the orientation
above is conforming.*

#### Scenario: UnitID splits correctly

- **WHEN** a producer packs the U-1 inputs
- **THEN** payload octet 8 is `0x23` and payload octet 14 is `0x81`

### Requirement: Component type vectors

A producer SHALL reproduce each component type vector below exactly, and SHALL
render all six as distinct UUID strings.

All inputs as P-1, varying only Component Type. These pin payload bits 120:121,
the two bits that two independent producers dropped.

| Type | Value | payload octet 14 | payload octet 15 | uuid |
|---|---|---|---|---|
| Platform | `0x0` | `0x00` | `0x00` | `d4abaad3-9b34-8c50-9800-028dcc084000` |
| CPU | `0x1` | `0x40` | `0x00` | `d4abaad3-9b34-8c50-9800-028dcc084100` |
| GPU | `0x2` | `0x80` | `0x00` | `d4abaad3-9b34-8c50-9800-028dcc084200` |
| NIC | `0x3` | `0xC0` | `0x00` | `d4abaad3-9b34-8c50-9800-028dcc084300` |
| NPU | `0x4` | `0x00` | `0x01` | `d4abaad3-9b34-8c50-9800-028dcc084001` |
| Other | `0xF` | `0xC0` | `0x03` | `d4abaad3-9b34-8c50-9800-028dcc084303` |

#### Scenario: An NPU does not collide with a Platform

- **WHEN** the NPU and Platform vectors are rendered
- **THEN** the two UUID strings differ

*They differ only in the last octet, and only because payload bits 120:121
survive the framing. A producer that renders payload bits 126:127 in that
position instead emits `d4abaad3-9b34-8c50-9800-028dcc084000` for both.*

#### Scenario: The type round-trips through the framing

- **WHEN** each of the six rendered UUIDs is parsed back to a payload
- **THEN** the recovered Component Type equals the input value

### Requirement: Framing is lossless over payload bits 0:121

The UUIDv8 framing SHALL preserve every one of payload bits 0 through 121 and
SHALL discard exactly payload bits 122 through 127, which are always zero.

#### Scenario: Exactly the padding is discarded

- **WHEN** each of the 128 single-bit payloads is framed and compared against
  the framing of the all-zero payload
- **THEN** the payloads that render identically to it are exactly bits 122, 123,
  124, 125, 126 and 127

#### Scenario: Round-trip over random payloads

- **WHEN** an arbitrary 122-bit payload is framed and then parsed back
- **THEN** the recovered payload is bit-for-bit identical to the input

### Requirement: Derived payload vectors

A producer SHALL reproduce each derived vector below exactly, in its HMAC digest,
its payload octets and its rendered UUID string.

**D-1**: the P-1 primary derived under the canonical fallback seed
`AMD-CUID-DEFAULT-SEED-v1` (24 octets).

```
message : d4abaad39b34c5060000a37302108000
hmac    : 61ffe99ab3e0e16a92c75457554ef0f67308b739df5e3cf87a2309a2c29df4ef
payload : 61ffe99ab3e0e16a0092c75457550e00
uuid    : 61ffe99a-b3e0-8e16-a802-4b1d515d5438
```

**D-2**: the P-1 primary derived under the 32-octet test key `000102 ... 1e1f`.
This vector exists because the key is trivially reproducible and contains no
placeholder that might later change.

```
message : d4abaad39b34c5060000a37302108000
hmac    : 73488f9eea526ce14989fe906c7a6e3cbd5f77eabee9b2d4a659703e05954617
payload : 73488f9eea526ce1004989fe906c1a00
uuid    : 73488f9e-ea52-86ce-8401-2627fa41b068
```

The fold is visible in both: digest octets 0:7 verbatim, a zero octet, digest
octets 8:12 verbatim, then digest octet 13 masked to its low five bits with the
auxiliary bit above it, then a zero octet.

#### Scenario: D-1 reproduces

- **WHEN** a producer derives the P-1 primary under the canonical fallback seed
- **THEN** the rendered UUID is `61ffe99a-b3e0-8e16-a802-4b1d515d5438`

#### Scenario: The derived slot is 45 bits, not 46

- **WHEN** the D-2 payload is compared against its digest
- **THEN** payload octet 14 is `0x1a`, being digest octet 13 (`0x7a`) masked to
  its low five bits
- **AND** the sixth bit of that octet, payload bit 117, is clear because the
  primary was not auxiliary

#### Scenario: Independently verifiable

- **WHEN** `openssl dgst -sha256 -mac HMAC -macopt key:AMD-CUID-DEFAULT-SEED-v1`
  is run over the 16 P-1 payload octets
- **THEN** it prints the D-1 digest

### Requirement: Auxiliary vectors

A producer of auxiliary CUIDs SHALL reproduce each auxiliary vector below exactly,
including the 32-octet input structure and its SHA-256 digest.

**A-1**: an auxiliary primary for a GPU at segment `0`, bus `0x63`, device `0`,
function `0`, on a machine whose `/etc/machine-id` decodes to
`0123456789abcdef0123456789abcdef`. Format `1`, Revision `0x00`, Device
`0x73A3`, Vendor `0x1002`, Component Type `0x2`.

```
input structure : 01000123456789abcdef0123456789abcdef0063000000a37302100200000000
sha-256         : 3395667ef8fa40ef5350d555d46a60e7998cd88db8c05c0858a6db284da0438c
auxiliary serial: 0xEF40FAF87E669533
payload         : 3395667ef8fa40ef0000a3730210a000
uuid            : 3395667e-f8fa-840e-bc00-028dcc084280
```

Payload octet 14 is `0xA0`: the auxiliary bit at position 5, the GPU type's low
two bits above it.

**A-2**: the A-1 primary derived under the temporary fixed key
`AMD-CUID-TEMP-KEY-v1` (20 octets).

```
message : 3395667ef8fa40ef0000a3730210a000
hmac    : 808fa59f69490ef9325ffa69c1bc4dd8cc7e3eb651ad557339b6362f4c8e5775
payload : 808fa59f69490ef900325ffa69c13c00
uuid    : 808fa59f-6949-80ef-a400-c97fe9a704f0
```

#### Scenario: A-1 sets the auxiliary bit

- **WHEN** the A-1 payload is decoded
- **THEN** bit 117 is set

#### Scenario: A-2 carries the auxiliary bit

- **WHEN** the A-2 payload is decoded
- **THEN** bit 117 is set
- **AND** payload octet 14 is `0x3c`, being digest octet 13 masked to five bits
  with the auxiliary bit above it

#### Scenario: The input structure is fixed-width

- **WHEN** the A-1 input structure is inspected
- **THEN** it is 32 octets
- **AND** its final four octets are zero, being the reserved field

### Requirement: The vector suite is shared between producers

The vectors SHALL be maintained as a single artifact consumed by every producer,
not transcribed independently into each. A producer SHALL fail its build if its
copy has drifted from the shared artifact.

A vector that has decayed in the same direction as the bug it was meant to catch
confirms the bug against itself. That has already happened once: a reverse-lookup
test decoded the component type from the same wrong bit positions the packer
wrote it to, and passed.

#### Scenario: Drift is detected

- **WHEN** a producer's local copy of the vectors differs from the shared
  artifact
- **THEN** the build fails

#### Scenario: A format change cannot land in one producer alone

- **WHEN** a change to the payload layout, the framing or the derivation is made
  in one producer
- **THEN** the shared vectors no longer match
- **AND** every other producer's build fails until it is updated too
