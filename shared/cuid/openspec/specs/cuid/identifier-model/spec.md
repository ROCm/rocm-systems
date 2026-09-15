## Purpose

The two-level identifier model the CUID specification defines: a **PrimaryID**
built from architectural hardware properties and available only to privileged
software, and a **DerivedID** produced from it by a non-reversible keyed hash
and handed to lower-privilege software. Both are called a Component Unified ID
and are interchangeable as identifiers; they differ only in the privilege level
at which each is available.

_Source: "Persistent platform component identification for SW tools", version 84._

## Requirements

### Requirement: Two identifier levels

The specification SHALL define exactly two identifier levels.

The **PrimaryID** SHALL be generated from the component's actual physical
hardware properties as presented in PCIe configuration space or an ACPI device
object, and SHALL be available only at privileged software levels.

The **DerivedID** SHALL be generated deterministically and algorithmically from
the PrimaryID, and SHALL cloak the hardware identifier from lower-privilege
software by means of a non-reversible hash.

Both values SHALL be referred to as a Component Unified ID (CUID), and SHALL be
equivalent in use as identifiers as far as applications and tool chains are
concerned.

#### Scenario: A tool consumes either level identically

- **WHEN** a tool is handed a CUID
- **THEN** it can use it as an identifier without knowing which level it is

#### Scenario: The derived value does not disclose the primary

- **WHEN** a consumer holds a DerivedID
- **THEN** regenerating the PrimaryID from it is prevented by the hash

### Requirement: The primary is withheld from unprivileged contexts

The PrimaryID SHALL NOT be available to non-privileged software contexts unless
explicitly permitted by system software security policy.

#### Scenario: An unprivileged context is refused the primary

- **WHEN** a non-privileged context requests the PrimaryID and policy does not
  permit it
- **THEN** it is not provided

### Requirement: The generating layer records the association

The higher-privilege software level that generates a DerivedID SHALL
persistently track its association with the PrimaryID, and SHALL permit
privileged tools to associate the two values directly for physical component
identification.

The association SHALL be tracked with a timestamp in a log, so that privileged
tools can identify the period during which a software context was bound to a
particular physical component.

It is recommended that the higher-privileged software expose a consistent API
for privileged tools to retrieve the associations.

#### Scenario: A recorded derived value is traced back

- **WHEN** a privileged tool holds a DerivedID recorded at a known time
- **THEN** it can determine which physical component that value named at that
  time

### Requirement: Identifier properties

A CUID SHALL satisfy all of the following.

| Property | Obligation |
|---|---|
| Compatibility | 128 bits, represented as an RFC 9562 UUID version 8 value |
| Generality | Generatable for any physical or hypervisor-projected component identifiable from presented architectural hardware properties alone |
| Consistency | Generated from architectural, individual, hardware-inherent properties through a deterministic algorithm; the same algorithm serves bare metal and guest VM |
| Portability | Implementable in any OS or system environment |
| Non-reversibility | The DerivedID is uniquely associated with the PrimaryID, and the algorithm prevents regenerating the PrimaryID from it |
| Scalability | Efficiently generatable, and non-colliding with any other component in any system, network or cloud environment |
| Persistence | Once created, a DerivedID is constant for the lifetime of the lesser-privileged software context that uses it |

Quantum-safe attestation of the device is explicitly a **non-goal**; where that
level of assurance is required it SHALL be provided by other mechanisms.

> **Recorded contradiction (C6, Scalability).** Scalability here requires that a
> CUID "does not collide with any other component anywhere in a system, network
> or cloud environment anywhere". `auxiliary-fallback` states that the auxiliary
> inputs "are not guaranteed to be unique across cluster nodes or whole data
> centers". Both cannot hold for an auxiliary CUID.
>
> **Recorded contradiction (C5, Persistence).** Persistence here requires a
> DerivedID to be "kept constant for the lifetime of the lesser-privileged SW
> context". `derived-identifier` makes the DerivedID a function of the shared
> salt. Re-keying therefore breaks this requirement, and the page never says
> whether re-keying is permitted or what it means when it happens.
>
> Both resolved in `openspec/changes/amend-published-cuid-spec/`.

#### Scenario: The identifier is 128 bits

- **WHEN** a CUID is represented
- **THEN** it is 128 bits, matching the size of commonly used GUID and UUID
  values

#### Scenario: Collision is not a practical concern

- **WHEN** two distinct components anywhere in a cluster are identified
- **THEN** their CUIDs differ, because the inputs are unique by manufacturing
  requirement and the transforms lose no precision below the data size

### Requirement: Persistence across live migration

Where a software context is live-migrated across physical device or platform
boundaries, its DerivedID SHALL remain constant for the lifetime of that
context, and the higher-privileged level SHALL re-associate it with the
PrimaryID of the physical component it is bound to afterwards.

#### Scenario: A migrated context keeps its identifier

- **WHEN** a VM is live-migrated to another physical platform
- **THEN** the DerivedID it sees is unchanged
- **AND** the privileged layer records the new primary association
