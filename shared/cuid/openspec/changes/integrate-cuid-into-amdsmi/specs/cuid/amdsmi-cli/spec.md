## Purpose

The operator-facing surface: what `amd-smi` prints, what it accepts, and what it
refuses to print. This is the layer at which a CUID either reaches a fleet
inventory or does not.

## ADDED Requirements

### Requirement: Static output reports the derived CUID

`amd-smi`'s static device output SHALL include the derived CUID, the component
type, whether the value is auxiliary, and which stage answered, for every GPU,
whenever CUID support is built in and the library answers. It SHALL also report
the node seed's state alongside them, because an unprovisioned node's derived
CUIDs are keyed with a public placeholder and an operator reading a derived CUID
without that context cannot tell a fleet-unique value from a reproducible one.

These fields SHALL be requested with `--cuid`, and `--cuid` SHALL NOT be one of
`amd-smi static`'s default arguments: the CUID block is opt-in.

This reverses an earlier position in this proposal, which required the block in
the default output. The common build is one without `libamdcuid`, where including
the block adds a column of `N/A` to every GPU in every default invocation, which
trains operators and scripts to ignore the fields rather than look for them.

`--cuid-primary` SHALL imply `--cuid`. It selects a *field of* the CUID block, so
requesting it without also selecting the block is never what a caller meant:
before it implied `--cuid`, `amd-smi static --asic --cuid-primary` printed no CUID
at all.

The node seed's state SHALL be reported once for the invocation, not once per
GPU. Repeating a node-wide property inside every GPU's dictionary invites a
reader to believe the seed is per-device.

Where CUID support is absent or the library has no value for a device, the output
SHALL say so explicitly rather than omitting the fields, using the tool's
existing `N/A` marker. A missing field is indistinguishable from a parsing
failure to a script.

#### Scenario: A GPU's CUID appears when it is asked for

- **WHEN** `amd-smi static --cuid` is run on a machine with CUID support
- **THEN** each GPU's block contains its derived CUID, its component type, its
  auxiliary state and its source
- **AND** the node seed's provisioned state and fingerprint are reported once
  for the invocation, outside the per-GPU blocks

#### Scenario: The CUID block is absent from the default output

- **WHEN** `amd-smi static` is run with no arguments selecting any block
- **THEN** no CUID fields appear

#### Scenario: --cuid-primary implies --cuid

- **WHEN** `amd-smi static --asic --cuid-primary` is run as root
- **THEN** the CUID block is printed, with the primary CUID in it

#### Scenario: Absence is explicit

- **WHEN** the CUID block is requested and CUID support is not built in
- **THEN** the CUID fields are present and read `N/A`

### Requirement: The primary CUID is not printed by default

`amd-smi` SHALL NOT include the primary CUID in its default static output. It
SHALL be shown only when the caller explicitly asks for it, through a separate
selector (`--cuid-primary`), and has the privilege to read it.

The primary payload embeds the component's raw serial number. `amd-smi static`
output is routinely attached to public bug reports.

#### Scenario: Default output omits the primary

- **WHEN** `amd-smi static` is run as root
- **THEN** the primary CUID does not appear

#### Scenario: An explicit request as root shows it

- **WHEN** the primary CUID is explicitly requested by a privileged caller
- **THEN** it is printed

#### Scenario: An explicit request without privilege does not

- **WHEN** the primary CUID is explicitly requested by an unprivileged caller
- **THEN** it is reported as unavailable rather than printed

### Requirement: The seed is provisioned through a documented command

`amd-smi` SHALL provide a command that provisions the node seed from 32 octets
supplied by the caller: `amd-smi set --cuid-seed FILE`.

The command SHALL take the seed from a file, or from standard input where the
argument is `-`, and SHALL NOT accept it as a command-line argument value. It
SHALL reject any length other than 32 octets before calling the API, so that the
error names the file rather than reporting a bare invalid-argument status.

A secret passed as an argument is visible in `/proc/<pid>/cmdline` to every user
on the machine for the lifetime of the process, and is written to the invoking
user's shell history.

Provisioning state is reportable without a command of its own: the provisioning
command SHALL echo the resulting state and fingerprint, and the static output's
CUID block SHALL carry the same two fields. A separate state command is not
required and SHALL NOT be the only way to learn the state, because the state is
only meaningful next to the derived CUID it qualifies.

#### Scenario: A seed is provisioned from a file

- **WHEN** an operator runs the provisioning command against a 32-octet file
- **THEN** the seed is provisioned and the command reports success

#### Scenario: A wrong-sized seed file is named in the error

- **WHEN** the file holds 16 octets
- **THEN** the command fails with an error naming the file and its length
- **AND** no provisioning call is made

#### Scenario: A seed cannot be passed on the command line

- **WHEN** the command-line surface is inspected
- **THEN** no option accepts seed material as an argument value

#### Scenario: Provisioning reports the fingerprint, not the seed

- **WHEN** the provisioning command completes on a node
- **THEN** it reports that the node is provisioned and prints the 8-octet
  fingerprint
- **AND** it prints no part of the seed

#### Scenario: The state is also readable without provisioning anything

- **WHEN** `amd-smi static` is run on a provisioned node
- **THEN** the CUID block reports the provisioned state and the fingerprint

### Requirement: Machine-readable output carries the same fields

Every field this specification requires in human-readable output SHALL also
appear in `amd-smi`'s JSON and CSV output, under stable names.

The names SHALL be `derived_cuid`, `primary_cuid`, `component_type`,
`auxiliary`, `source`, `seed_provisioned` and `seed_fingerprint`. Naming them
here is the point of the requirement: a field whose name is whatever the
implementation happened to choose is a field a fleet inventory has to rediscover
after every release.

#### Scenario: JSON output carries the CUID fields

- **WHEN** static output is requested as JSON
- **THEN** the derived CUID, the component type, the auxiliary flag, the source
  and the seed state appear under those names

#### Scenario: CSV output carries them too

- **WHEN** static output is requested as CSV
- **THEN** the same names appear as columns, flattened out of the CUID block

#### Scenario: The Python interface exposes the same calls

- **WHEN** the Python interface is used
- **THEN** the identity snapshot and the seed commands are available with the
  same semantics as the C API
