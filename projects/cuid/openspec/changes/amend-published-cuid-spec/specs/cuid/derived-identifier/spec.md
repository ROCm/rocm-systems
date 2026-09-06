## MODIFIED Requirements

### Requirement: Derived payload layout

The 122 bits of the DerivedID value SHALL be laid out as:

| Bit | Name | Comments |
|---|---|---|
| 0:63 | `hash[0:63]` | LSB of the digest |
| 64:71 | Reserved | Must be 0 |
| **72:116** | **`hash[64:108]`** | **45 bits, matching the slot's width** |
| 117 | Auxiliary Value Identifier | Copied from the primary |
| 118:121 | Reserved2 | Must be 0 |

The DerivedID SHALL therefore carry **109** hash bits, not 110.

*The published table labels a 45-bit slot `bits [64:109]`, 46 bits, inside prose
saying 110. The slot width is what leaves room for bit 117, so the label and the
prose move. No conforming value changes.*

#### Scenario: The derived slot is 45 bits

- **WHEN** a derived payload is compared against its digest
- **THEN** payload octet 14 is digest octet 13 masked to its low five bits
- **AND** the sixth bit of that octet is the Auxiliary Value Identifier

#### Scenario: The reserved fields are zero

- **WHEN** a DerivedID payload is decoded
- **THEN** bits 64:71 and 118:121 are zero

#### Scenario: Collision resistance

*Carried from the baseline with the figure corrected over the 109 hash bits.*

- **WHEN** the chance of an accidental collision is computed
- **THEN** the bound is 1/(2^(109-1)), taken over the 109 hash bits, not
  1/(2^(122-1))

### Requirement: Derivation by HMAC-SHA-256

Unchanged in operand order and algorithm. The key SHALL additionally be
specified as follows.

A provisioned shared salt SHALL be **exactly 32 octets**. A producer SHALL
reject any other length and SHALL continue to use the previously effective key
rather than adopting a truncated or padded one.

Until an administrator provisions a secret, a producer SHALL key the derivation
with the **canonical fallback seed**, which is exactly the 24-octet ASCII string
`AMD-CUID-DEFAULT-SEED-v1`, with no terminating NUL and no padding to any other
length.

The fallback seed is public. A derived CUID produced under it is stable and
reproducible but SHALL NOT be treated as fleet-unique, and SHALL be documented
as a placeholder.

*HMAC-SHA-256 pads a short key to its own 64-octet block internally, so padding
to 32 first is a different key. Unstated, two implementations each invent one.*

#### Scenario: The operands

- **WHEN** a DerivedID is generated
- **THEN** the PrimaryID payload is the message and the shared salt is the key

#### Scenario: The salt is 256 bits

- **WHEN** the shared secret is established
- **THEN** it is 256 bits

#### Scenario: A shared salt gives a shared answer

- **WHEN** two nodes carrying the same salt derive from the same PrimaryID
- **THEN** they produce the same DerivedID

#### Scenario: An unprovisioned kernel and library agree

- **WHEN** both are asked for the derived CUID of the same component on a
  machine with no provisioned seed
- **THEN** both key with the same 24 octets and emit the same value

#### Scenario: A wrong-sized salt is refused

- **WHEN** a 16-octet salt is provisioned
- **THEN** the producer reports an error and the derived CUID is unchanged

### Requirement: The HMAC message is sixteen octets

The message SHALL be the **sixteen octets** into which the 122-bit primary
payload is packed, LSB first, with payload bits 122:127 present and zero.

*HMAC-SHA-256 consumes octets, and "the primary ID 122bit wide value" is not a
whole number of octets, so it does not name a byte string. A producer that fed
the bits MSB-first or right-aligned emits values sharing no bits, with nothing in
either value to show which was which.*

#### Scenario: The message is the packed payload

- **WHEN** a derived CUID is computed
- **THEN** the HMAC message is the same sixteen octets the primary CUID is
  framed from

#### Scenario: The padding bits are part of the message

- **WHEN** the message is inspected
- **THEN** its final six bits are the payload's zero padding

### Requirement: The collision bound follows the varying bits

The collision bound for a DerivedID SHALL be stated over the **109 hash bits**
it carries, not over 122.

*Thirteen of the published figure's 122 bits are reserved or the auxiliary flag
and fixed across every value, so they contribute nothing to collision resistance.
This changes no value and no code; an overstated bound gets quoted into a
security review, and 2^109 is ample as it stands.*

#### Scenario: The bound is over what varies

- **WHEN** the collision bound is computed
- **THEN** it is taken over the 109 hash bits the derived payload carries
