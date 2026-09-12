## Purpose

How the node-wide derivation seed is provisioned through `amd-smi`, and what may
be read back. Without provisioning, every derived CUID in a fleet is keyed with a
public placeholder and is reproducible by anyone; with it, the seed is the most
sensitive value the tool can touch.

## ADDED Requirements

### Requirement: Provisioning takes exactly 32 octets

`amd-smi` SHALL expose a call that provisions the node-wide seed from exactly 32
octets, and SHALL reject any other length before any state changes.

The C entry point SHALL take a fixed 32-octet argument and no length parameter,
because there is no other accepted length: a seed of another size is corruption,
not a shorter secret, and a length parameter invites the caller to think
otherwise. The bindings above it, the Python interface and the CLI, SHALL each
check the length themselves before calling in, so that the caller who supplied a
wrong-sized buffer gets an error naming what it supplied.

Provisioning SHALL require privilege. On success, every derived CUID on the node
SHALL be keyed with the new seed; primary CUIDs SHALL be unchanged. Any cached
or recorded derived value produced under the previous seed SHALL be invalidated
by the same operation, or the node continues to report pre-re-key values from a
store that outlives the key.

The call SHALL propagate a failure to persist the seed to its caller. It SHALL
NOT report success when the seed was applied in memory but could not be stored,
because the next process to start would then derive different values from the
same nominally provisioned node.

#### Scenario: A short seed is refused

- **WHEN** a caller provisions 16 octets through the Python interface or the CLI
- **THEN** the call fails before the API is entered
- **AND** the derived CUIDs on the node are unchanged

#### Scenario: An unprivileged caller is refused

- **WHEN** an unprivileged caller attempts to provision a seed
- **THEN** the call fails with a permission error
- **AND** the seed in use is unchanged

#### Scenario: A store failure is reported

- **WHEN** the seed cannot be written to its store
- **THEN** the call reports failure rather than success

#### Scenario: Provisioning changes every seed-derived value

- **WHEN** a 32-octet seed is provisioned
- **THEN** every derived CUID on the node that is keyed with the seed changes
- **AND** every primary CUID is unchanged
- **AND** an auxiliary component's derived CUID is unchanged, because it is keyed
  with the fixed public temporary key `AMD-CUID-TEMP-KEY-v1` and not with the
  seed

#### Scenario: A record written under the old seed does not survive

- **WHEN** a derived CUID has been recorded in the producer's own store and the
  seed is then replaced
- **THEN** a subsequent query returns the value derived from the new seed, not
  the recorded one

### Requirement: The seed is never readable through `amd-smi`

`amd-smi` SHALL NOT expose the provisioned seed, in whole or in part, through any
API, any command-line output, any log, or any error message.

`amd-smi` is run casually under `sudo` and its output is routinely pasted into
bug reports and support bundles. The kernel's `cuid_seed` attribute is readable
because a provisioning daemon needs to verify its own write against one device,
which is a different caller with a different threat model.

#### Scenario: No call returns the seed

- **WHEN** the API surface is enumerated
- **THEN** no entry point returns seed material

#### Scenario: Output carries no seed material

- **WHEN** a seed is provisioned and the tool's output and logs are inspected
- **THEN** no octet of the seed appears in either

### Requirement: Provisioning state is reportable without the seed

`amd-smi` SHALL report whether the node is provisioned or is using the canonical
public fallback seed, together with a non-reversible fingerprint of the seed in
use.

The fingerprint SHALL be the first 8 octets of the unkeyed SHA-256 of the seed.

The operational question is "do these two nodes carry the same seed", not "what
is the seed". A truncated digest answers the first and not the second.

The reported state describes the node's seed store. Where a producer holds a
seed of its own, a driver publishing a CUID from a per-device seed for example,
the reported fingerprint SHALL NOT be read as a statement about the key that
produced any particular identifier. The two agree at provisioning and can drift
afterwards, so a consumer that needs to know which key produced a value must ask
the producer that answered.

A seed store that exists but cannot be read by the caller SHALL be reported as a
permission failure, and SHALL NOT be reported as an unprovisioned node. No
provisioning state and no fingerprint SHALL be reported in that case.

"Unprovisioned" and "not readable from here" are different answers to different
questions. The store is root-owned and root-readable, so an ordinary caller on a
provisioned node is in the second state. Reporting it as the first returns the
public fallback seed's fingerprint for a node that carries a secret: a wrong
answer rather than an absent one, and the one an operator auditing a fleet reads
as "this node was never provisioned". A caller told it lacks privilege can re-run
under `sudo`; a caller told the node is unprovisioned has no reason to.

#### Scenario: An unprovisioned node says so

- **WHEN** no seed has been provisioned
- **THEN** the reported state is the canonical fallback seed
- **AND** the fingerprint is that of the 24-octet canonical seed

#### Scenario: A seed store the caller cannot read is not unprovisioned

- **WHEN** an unprivileged caller reports provisioning state on a node whose
  seed store exists but is not readable by that caller
- **THEN** the call fails with a permission error
- **AND** no provisioning state and no fingerprint are reported
- **AND** the node is not reported as unprovisioned

#### Scenario: Two nodes with the same seed match

- **WHEN** the same 32-octet seed is provisioned on two nodes
- **THEN** both report the same fingerprint

#### Scenario: The fingerprint does not disclose the seed

- **WHEN** the fingerprint is inspected
- **THEN** it is 8 octets of a SHA-256 digest
- **AND** recovering the seed from it requires breaking SHA-256 preimage
  resistance
