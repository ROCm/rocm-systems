## ADDED Requirements

### Requirement: Identity without a key

A driver that implements no node key SHALL publish `cuid_primary` and
`cuid_unit_id` for every whole GPU and compute partition, and SHALL NOT publish
`cuid_derived` or `cuid_seed`.

#### Scenario: Kernel without the node key

- **WHEN** amdgpu implements no node key
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

- **WHEN** `cuid_unit_id` exists for a whole GPU and `cuid_derived` does not,
  including a GPU in SPX whose amd-smi handle names its one partition
- **THEN** the library and amd-smi report the GPU's `cuid_primary` as the
  primary CUID
- **AND** a temporary CUID with source `LIBRARY` as the derived CUID

### Requirement: A GPU handle names a compute partition

Where the driver publishes `cuid_unit_id` in a GPU handle's partition node
(`<render device>/xcp`), amd-smi SHALL look the handle up there, and report
that partition's CUID when it has one. When it has none, amd-smi SHALL report
the whole GPU's CUID, with the GPU's source and primary CUID, if the GPU is in
SPX, whose one partition covers every XCC, and SHALL report no CUID in any mode
with more than one partition. A handle of a GPU in a mode with more than one
partition SHALL NOT report the whole GPU's CUID, whether or not the driver
publishes a partition node.

#### Scenario: SPX partition without a CUID

- **WHEN** a GPU in SPX publishes `cuid_unit_id` 512 in its partition node and
  no derived CUID for it
- **THEN** its amd-smi handle reports the whole GPU's CUID and source

#### Scenario: DPX partition without a CUID

- **WHEN** a GPU in DPX publishes partition nodes without a derived CUID
- **THEN** each of its amd-smi handles reports `AMDSMI_STATUS_NOT_SUPPORTED`
- **AND** none reports the whole GPU's CUID

#### Scenario: DPX GPU on a driver without CUID support

- **WHEN** a GPU in DPX publishes no CUID attribute at all
- **THEN** the handles of partition 0 and partition 1 both report
  `AMDSMI_STATUS_NOT_SUPPORTED`
- **AND** the same GPU in SPX reports its whole-GPU CUID
