## Purpose

What `amd-smi` reports for a component's Component Unified ID, and what a caller
can determine about a value it has been handed. A CUID recorded without its
provenance is a string.

## ADDED Requirements

### Requirement: A single call returns a consistent identity snapshot

`amd-smi` SHALL expose one call that returns, for a processor handle, all of:

| Field | Content |
|---|---|
| Primary CUID | the canonical UUIDv8 string, or empty where the caller is not privileged enough to have it |
| Derived CUID | the secondary UUIDv8 string |
| Component type | the on-wire Component Type value |
| Auxiliary | whether payload bit 117 is set |
| Source | which stage of the staged lookup answered |

The five values SHALL describe one snapshot of one component. A producer SHALL
NOT satisfy this requirement with separate per-field calls that a caller must
combine, because a re-key or a device rescan between two such calls yields a
value recorded with the wrong provenance.

The call SHALL NOT fail merely because the primary is unavailable to the caller.
An unprivileged caller SHALL receive the derived CUID, the component type, the
auxiliary flag and the source, with the primary returned as an empty string.

Privilege here is whatever the layer below enforces, and the two layers differ:
the driver gates `cuid_primary` on `CAP_SYS_ADMIN`, while `libamdcuid` gates its
primary query on an effective UID of zero. A caller holding `CAP_SYS_ADMIN`
without being root therefore still receives an empty primary through this call.

#### Scenario: An unprivileged caller gets the derived CUID

- **WHEN** a caller whose effective UID is not zero asks for a GPU's CUID
  information
- **THEN** the call succeeds
- **AND** the derived CUID is populated
- **AND** the primary CUID is the empty string

#### Scenario: A privileged caller gets both

- **WHEN** a caller whose effective UID is zero asks for the same GPU
- **THEN** both the primary and the derived CUID are populated

#### Scenario: Both callers see the same driver-published derived CUID

- **WHEN** the driver publishes `cuid_secondary` for a GPU, or a record store
  holds its derived value, and both a privileged and an unprivileged caller ask
- **THEN** both receive the same derived CUID

*The equality is scoped to those two stages on purpose. Where neither answers,
the library computes the value itself, and an unprivileged process cannot read
the component's serial, so it computes an auxiliary identity where a root process
computes the canonical one and the two derived values differ. `cuid_secondary` is
`0444` precisely so that the value an ordinary user records is the value root
records.*

#### Scenario: The fields agree with each other

- **WHEN** the returned auxiliary flag is set
- **THEN** payload bit 117 of the returned derived CUID is set

### Requirement: The auxiliary marker is reported, not merely encoded

`amd-smi` SHALL report whether a CUID is auxiliary as an explicit field. A caller
SHALL NOT have to parse the UUID string and mask payload bit 117 to learn it.

An auxiliary CUID is built from `/etc/machine-id` and the device's routing ID. It
changes when the OS is reinstalled and it is not unique across nodes. A fleet
inventory that records it in the same column as a canonical CUID will report two
machines as one, or one machine as two after a reimage, and nothing in the value
distinguishes them.

#### Scenario: A synthesised identity is flagged

- **WHEN** a GPU has no reachable serial and the library synthesises an
  auxiliary CUID for it
- **THEN** `amd-smi` reports the auxiliary flag as set

#### Scenario: A driver-published identity is not flagged

- **WHEN** the CUID came from the driver's sysfs interface
- **THEN** the auxiliary flag is clear

### Requirement: The answering stage is reported

`amd-smi` SHALL report which stage of the staged lookup produced the value, as
one of: the device driver's published interface, a local daemon or configuration
store, the library's own computation, or `unknown` where the answering stage
cannot be determined.

The auxiliary flag says the *identity* was synthesised. The source says whether
the *value* is authoritative. A canonical primary the library reconstructed from
PCI configuration space is a real serial and carries a clear auxiliary flag, but
it is still a second computation of something the kernel owns. Separating the
store from the library's own computation requires `libamdcuid` to report the
answering stage, which it does not, so those two SHALL be reported as unknown
rather than guessed apart.

#### Scenario: A driver-published value is identified as such

- **WHEN** the driver publishes `cuid_secondary` for a GPU
- **THEN** the reported source is the driver interface

#### Scenario: A source that cannot be determined is reported as unknown

- **WHEN** a GPU's driver publishes nothing, so the value was produced by the
  store or by the library's own computation
- **THEN** the auxiliary flag still reflects the payload, and the reported
  source is unknown rather than a guess, because `libamdcuid` writes a computed
  value back into the same store it consults, so the presence of a record proves
  only that something looked the device up before
- **AND** a caller SHALL read unknown as "not known to be driver-published"
  rather than as "locally computed"

### Requirement: The legacy device UUID is retained and superseded

`amd-smi` SHALL continue to expose its existing device UUID call, returning
exactly the value it returns today. That call SHALL NOT be changed to return a
CUID.

Its documentation SHALL state that it is not a CUID, that it changes when a GPU
is repartitioned, and that the CUID calls are the identifiers to record.

The two values answer different questions. The legacy UUID incorporates the KFD
partition index, so it names an addressable device; a CUID names a physical
component and moves its partition index into the UnitID field without changing
the rest. Both callers exist. Redefining the legacy call would change the meaning
of a published ABI under consumers who have already recorded its output.

#### Scenario: The legacy value is unchanged

- **WHEN** an existing consumer calls the device UUID entry point
- **THEN** it receives the same value it received before this change

#### Scenario: Repartitioning distinguishes the two

- **WHEN** a GPU is repartitioned
- **THEN** the legacy device UUID changes
- **AND** the CUID's non-UnitID fields do not

### Requirement: The single-string CUID call remains and shares one path

The existing single-string CUID entry point SHALL remain, SHALL return the
derived CUID, and SHALL be implemented in terms of the snapshot call rather than
performing its own lookup.

Two lookup paths to one value is how the two producers came to disagree. There
SHALL be one.

#### Scenario: The two calls agree

- **WHEN** both the single-string call and the snapshot call are made for one
  GPU
- **THEN** the string returned by the first equals the derived CUID in the
  second

### Requirement: An absent CUID library degrades, it does not fail the build

Where `libamdcuid` is not available at build time, the CUID entry points SHALL
still be present in the ABI and SHALL return a "not supported" status. Their
presence SHALL NOT be conditional on the build option: the exported symbol set
is the same either way, so a consumer can call them unconditionally and read the
status rather than feature-detecting the symbol.

Where the library is available, it SHALL be enabled by default. A build option
MAY force it off. The detection SHALL be `find_package(amdcuid CONFIG)` against
the package `libamdcuid` exports, not a `find_library()` paired with a separately
located header, because the two halves found independently can come from
different installs and nothing in the build says so.

A null-argument call SHALL return an invalid-argument status whether or not the
library was linked, so that a caller cannot distinguish the two builds by
passing garbage.

#### Scenario: A build without the library still links

- **WHEN** `amd-smi` is built on a tree with no `libamdcuid`
- **THEN** it builds and links
- **AND** the CUID entry points are exported
- **AND** they return a not-supported status

#### Scenario: A build with the library enables it without being asked

- **WHEN** `libamdcuid` is present and no build option is given
- **THEN** the CUID entry points are functional

#### Scenario: A null argument is a caller error in both builds

- **WHEN** a null pointer is passed to any of the CUID entry points
- **THEN** the status is invalid-argument, not not-supported
