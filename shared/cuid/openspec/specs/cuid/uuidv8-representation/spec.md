## Purpose

How the 122-bit CUID payload is carried inside a 128-bit RFC 9562 UUID version 8
value. The same framing serves both the Primary and the Derived identifier.

_Source: "Persistent platform component identification for SW tools", version 84._

## Requirements

### Requirement: UUIDv8 payload framing

A CUID SHALL be represented as a 128-bit RFC 9562 UUID version 8 value, with a
122-bit ID value payload distributed as follows:

| Bit position | Value | Comments |
|---|---|---|
| 0:47 | ID value (part 1) | LSB of ID value |
| 48:51 | must be 8 (fixed, RFC 9562 UUIDv8) | |
| 52:63 | ID value (part 2) | |
| 64:65 | `10b` (RFC 9562 variant) | |
| 66:127 | ID value (part 3) | MSB of ID value |

The payload SHALL be little-endian ordered and LSB aligned.

The same format SHALL be used for both Primary and Derived identifier
generation.

#### Scenario: The version nibble is 8

- **WHEN** any CUID is rendered
- **THEN** rendered bits 48:51 are `8`

#### Scenario: The variant is RFC 9562 compliant

- **WHEN** any CUID is rendered
- **THEN** rendered bits 64:65 are `10b`

#### Scenario: The payload is 122 bits

- **WHEN** the ID value carried by the three payload ranges is measured
- **THEN** it is 48 + 12 + 62 = 122 bits
