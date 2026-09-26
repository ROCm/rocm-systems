## Purpose

The kernel-side sysfs contract through which a driver publishes a component's
CUID: which attributes exist, what they contain, who may read them, how the
derivation secret is provisioned, and what is recorded when it changes. The
userspace library consumes this in preference to recomputing identifiers itself.
Once upstream it is a stable kernel ABI.

## ADDED Requirements

### Requirement: CUID attribute group

A driver that can supply a component's identity SHALL expose a CUID attribute
group on the component's device node. For a PCI component the attributes appear
under `/sys/bus/pci/devices/<dev>/`.

The group SHALL consist of exactly three attributes: `cuid_primary`,
`cuid_secondary` and `cuid_seed`. It SHALL be created as a group so that all
three appear atomically, and removed in full when the device is unbound.

A driver SHALL NOT expose a temporary or auxiliary CUID attribute. Auxiliary
identifiers are user-mode only.

#### Scenario: Attributes present after bind

- **WHEN** the driver binds to a supported device
- **THEN** `cuid_primary`, `cuid_secondary` and `cuid_seed` exist under that
  device's sysfs directory

#### Scenario: Attributes removed on unbind

- **WHEN** the driver is unbound or the module unloaded
- **THEN** all three attributes are removed
- **AND** no stale reference to the device remains

#### Scenario: Group creation fails

- **WHEN** creating the attribute group fails
- **THEN** device probe reports the failure
- **AND** teardown does not attempt to remove attributes that were never created

### Requirement: cuid_primary attribute

`cuid_primary` SHALL be read-only and SHALL emit the component's primary CUID as
a lower-case, hyphenated UUID string followed by a newline.

Because the payload embeds the component's raw serial number, the attribute
SHALL have mode `0400` and SHALL additionally verify `CAP_SYS_ADMIN` on the
reading process at read time, returning `-EPERM` when the check fails.

The value SHALL be fixed for the lifetime of the device binding.

#### Scenario: Privileged read

- **WHEN** a process with `CAP_SYS_ADMIN` reads `cuid_primary`
- **THEN** it receives the primary CUID as a UUIDv8 string with a trailing
  newline

#### Scenario: Unprivileged read is refused

- **WHEN** a process without `CAP_SYS_ADMIN` reads `cuid_primary`
- **THEN** the read fails with `EPERM`

*The file mode alone is not sufficient: a process may hold an open descriptor
obtained under different credentials, or the file may be reachable through a
relaxed-permission path.*

#### Scenario: Value is stable

- **WHEN** `cuid_primary` is read repeatedly, including after `cuid_seed` is
  written
- **THEN** the value is identical every time

### Requirement: cuid_secondary attribute

`cuid_secondary` SHALL be read-only with mode `0444` and SHALL emit the derived
CUID as a lower-case, hyphenated UUID string followed by a newline.

The value discloses neither the serial number nor the seed, so it is the value
unprivileged tooling is expected to consume.

Reads SHALL be consistent: a read SHALL return either the value derived from the
previous seed or the value derived from the new one, never a partially updated
value.

#### Scenario: Unprivileged read succeeds

- **WHEN** any process reads `cuid_secondary`
- **THEN** it receives the derived CUID with no privilege check

#### Scenario: Value changes when the seed changes

- **WHEN** a new seed is written to `cuid_seed`
- **THEN** a subsequent read of `cuid_secondary` returns the value derived from
  the new seed

#### Scenario: Concurrent read during a seed write

- **WHEN** `cuid_secondary` is read concurrently with a write to `cuid_seed`
- **THEN** the read returns a complete, well-formed UUID derived from one seed
  or the other

### Requirement: cuid_seed attribute

`cuid_seed` SHALL be readable and writable, with mode `0600`, and SHALL
additionally verify `CAP_SYS_ADMIN` on both the read and the write path,
returning `-EPERM` when the check fails.

The seed SHALL be handled as raw bytes, and a write SHALL be accepted only when
it is exactly 32 bytes long. A write of any other length, shorter or longer,
SHALL be rejected with `-EINVAL`. A provisioned seed is the full 256 bits or it
is not a seed: a short write is a truncated secret, not a weaker one, so it is
refused rather than padded. A read SHALL return the seed currently in use, the
provisioned secret or the built-in default when none has been provisioned, in
the same encoding a write accepts, so that a provisioned value round-trips. The
built-in default is a fixed public string and is shorter than 32 bytes; a read
before any provisioning therefore returns fewer than 32 bytes, and that value is
not required to be writable back.

A successful write SHALL take effect immediately: the derived CUID is recomputed
before the write returns.

#### Scenario: Provisioning a seed

- **WHEN** a privileged administrator writes a 32-byte secret to `cuid_seed`
- **THEN** the write succeeds and reports the full byte count consumed
- **AND** `cuid_secondary` immediately reflects the new seed

#### Scenario: Round-trip

- **WHEN** a seed is written and then read back
- **THEN** the bytes read are identical to the bytes written

#### Scenario: Write of any length other than 32 rejected

- **WHEN** a write of more than 32 bytes is attempted
- **THEN** the write fails with `EINVAL`
- **AND** the seed in use is unchanged
- **WHEN** a write of fewer than 32 bytes is attempted, including a 16-byte
  secret and a 32-byte secret with a trailing newline stripped to 31
- **THEN** the write fails with `EINVAL`
- **AND** the seed in use is unchanged

#### Scenario: Unprivileged access refused

- **WHEN** a process without `CAP_SYS_ADMIN` reads or writes `cuid_seed`
- **THEN** the operation fails with `EPERM`

#### Scenario: Seed does not survive a reload

- **WHEN** the driver is unloaded and reloaded after a seed was provisioned
- **THEN** the default seed is in use again and the seed must be re-provisioned

*The seed is deliberately not persisted by this interface. Persisting a secret
requires a store that is not world-readable, which is a separate decision.*

### Requirement: Default seed

Until a secret is provisioned, the derived CUID SHALL be computed from a
documented, public default seed, so that `cuid_secondary` is always readable and
well-formed.

Because the default is public, the resulting value is stable but not
fleet-unique. The interface documentation SHALL state that it is a placeholder
and that real deployments must provision a secret.

#### Scenario: Reading before provisioning

- **WHEN** `cuid_secondary` is read on a freshly loaded driver
- **THEN** it returns a valid UUIDv8 derived from the default seed

#### Scenario: Two nodes without provisioning

- **WHEN** two nodes both use the default seed
- **THEN** the same physical component yields the same secondary CUID on both,
  but the value provides no fleet-level secrecy

### Requirement: Association record

Whenever the derived CUID is computed or recomputed, the driver SHALL record the
association between the primary and the derived identifier in the kernel log, at
device-info level, including both values.

Re-keying invalidates every secondary CUID handed out so far, so the record
exists to let an operator later determine which physical component a previously
issued derived CUID named.

#### Scenario: Association logged at probe

- **WHEN** the driver binds and computes the initial derived CUID
- **THEN** the kernel log contains one entry for that device naming both the
  primary and the secondary CUID

#### Scenario: Association logged on re-key

- **WHEN** a new seed is written
- **THEN** a further log entry records the primary and the new secondary CUID

### Requirement: Secret handling

The seed SHALL be zeroed from kernel memory when the device is torn down and
when a group creation failure aborts setup. Intermediate HMAC digests SHALL be
zeroed after use.

The seed SHALL NOT appear in any log message, error string, or attribute other
than `cuid_seed` itself.

#### Scenario: Teardown wipes the seed

- **WHEN** the device is unbound after a seed was provisioned
- **THEN** the stored seed bytes are explicitly zeroed

#### Scenario: Seed is never logged

- **WHEN** any CUID operation emits a log message
- **THEN** the message contains no seed material

### Requirement: Registration conditions

A driver SHALL register the CUID group only for components whose identity it can
supply, and SHALL supply the component type appropriate to the device.

Where a genuine serial number is unavailable, from neither a device-specific
serial nor the PCIe Device Serial Number capability, the driver SHALL NOT publish a
synthesised identifier. It either does not register the group, or the absence is
handled at a higher layer; in no case does it emit an identifier with the
Auxiliary Value Identifier set.

Failure to register the CUID group SHALL NOT prevent the device from
functioning.

#### Scenario: Device with no serial number

- **WHEN** the driver binds to a device exposing no serial number of any kind
- **THEN** no CUID attributes claiming a genuine identity are published
- **AND** the device otherwise initialises normally

#### Scenario: GPU registration

- **WHEN** the GPU driver registers the group
- **THEN** the component type in the payload is GPU
