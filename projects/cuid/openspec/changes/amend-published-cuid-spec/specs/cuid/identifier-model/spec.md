## MODIFIED Requirements

### Requirement: Identifier properties

Unchanged, except that **Scalability** and **Persistence** are qualified so that
each says what it actually guarantees.

**Scalability (C6).** A **canonical** CUID, one whose serial comes from an
architectural hardware source, SHALL NOT collide with any other component in
any system, network or cloud environment.

An **auxiliary** CUID SHALL NOT carry that guarantee. It is built from the OS
machine identity and the device's routing ID, neither of which is unique across
nodes, and it changes when the OS is reinstalled. It SHALL be marked by payload
bit 117 so a consumer can tell the two apart **before** relying on either.

*The published text asserts the guarantee globally and then withdraws it for the
auxiliary case. Both are correct about different things.*

**Persistence (C5).** A DerivedID SHALL be constant for the lifetime of the
lesser-privileged software context **for as long as the shared salt is
unchanged**.

Re-keying SHALL be understood as an administrative invalidation of every
DerivedID previously handed out, not as a routine operation. A producer SHALL
record the primary-to-derived association whenever it changes, so a value handed
out before a re-key remains traceable.

*The published text requires the DerivedID to be constant for the context's
lifetime and separately makes it a function of a salt, without saying what
happens if the salt changes. Something has to give, and it cannot be the
derivation.*

#### Scenario: The identifier is 128 bits

- **WHEN** a CUID is represented
- **THEN** it is 128 bits, matching the size of commonly used GUID and UUID
  values

#### Scenario: Collision is not a practical concern

*Carried from the baseline with the scope narrowed to canonical CUIDs.*

- **WHEN** two distinct components anywhere in a cluster are identified by
  **canonical** CUIDs
- **THEN** their CUIDs differ, because the inputs are unique by manufacturing
  requirement and the transforms lose no precision below the data size
- **AND** the same is not claimed of an auxiliary CUID, which payload bit 117
  marks so a consumer can tell before relying on it

#### Scenario: An auxiliary value is not claimed to be globally unique

- **WHEN** a consumer reads a CUID with payload bit 117 set
- **THEN** it does not rely on it being unique beyond the node

#### Scenario: Re-keying invalidates rather than corrupts

- **WHEN** the shared salt is replaced
- **THEN** every derived CUID on the node changes
- **AND** every primary CUID is unchanged
- **AND** the producer records the new association

### Requirement: Two identifier levels

Unchanged, with one addition: the specification defines two **constructions**,
and a consumer SHALL be able to tell them apart.

A **constructed** CUID SHALL be built from the 122-bit payload layout and framed
as a UUIDv8. Its version nibble SHALL be `8` and its payload fields SHALL be
decodable.

An **adopted** CUID SHALL be a UUID that firmware or another standard has
already made unique, such as an SMBIOS system UUID or an ACPI device object UUID,
used verbatim. It SHALL retain the version and variant bits its source wrote,
and its payload fields SHALL NOT be decodable.

A consumer SHALL NOT reject a CUID on the basis of its version nibble. It SHALL
treat the nibble as the discriminator between the two constructions: `8` means
constructed and decodable, anything else means adopted and opaque.

*See `primary-identifier` for why this is a correction rather than an
elaboration.*

#### Scenario: A tool consumes either level identically

- **WHEN** a tool is handed a CUID
- **THEN** it can use it as an identifier without knowing which level it is

#### Scenario: The derived value does not disclose the primary

- **WHEN** a consumer holds a DerivedID
- **THEN** regenerating the PrimaryID from it is prevented by the hash

#### Scenario: A consumer knows whether it may decode

- **WHEN** a consumer holds a CUID whose version nibble is not `8`
- **THEN** it treats the value as an opaque adopted identifier
- **AND** it does not read a Component Type or an auxiliary marker out of it

#### Scenario: A validating consumer does not reject a platform

- **WHEN** a consumer that validates UUIDs is handed a Platform CUID sourced
  from SMBIOS
- **THEN** it accepts it
