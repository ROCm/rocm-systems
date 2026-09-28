## ADDED Requirements

### Requirement: Identity without a key

A driver without a node key SHALL publish `cuid_primary` and `cuid_unit_id` for
every whole GPU and compute partition, and SHALL NOT publish `cuid_derived`,
`cuid_seed` or `cuid_seed_state`.

#### Scenario: Kernel without the key store

- **WHEN** amdgpu has no node key
- **THEN** `cuid_primary` (0400) and `cuid_unit_id` (0444) exist for the GPU
  and each partition
- **AND** no `cuid_derived` exists

### Requirement: cuid_unit_id attribute

Beside every `cuid_primary` the driver SHALL publish `cuid_unit_id`, mode
0444, the component's UnitID in decimal followed by a newline: `0` on the
device, the partition's UnitID on a partition.

#### Scenario: Readable without privilege

- **WHEN** an unprivileged user reads a partition's `cuid_unit_id`
- **THEN** it matches the UnitID packed into that partition's primary

### Requirement: Driver-published marker

The library and amd-smi SHALL treat a GPU or partition as driver-published
when `cuid_unit_id` exists in its sysfs directory. A `cuid_primary` without
`cuid_unit_id` SHALL count as nothing published.

#### Scenario: Identity kernel, root caller

- **WHEN** `cuid_unit_id` exists for a whole GPU and `cuid_derived` does not
- **THEN** the library reports the driver's `cuid_primary` as the primary CUID
- **AND** a temporary CUID with source `LIBRARY` as the derived CUID

#### Scenario: A published partition is never the whole card

- **WHEN** `cuid_unit_id` exists in a partition's `xcp` directory
- **THEN** amd-smi looks the partition up there
- **AND** does not report the whole card's CUID for it
