## MODIFIED Requirements

### Requirement: cuid_seed_state attribute

Beside every `cuid_derived` the driver SHALL publish `cuid_seed_state`, mode
0444, reading `unprovisioned\n` while the key is the one the driver generated
and `provisioned\n` once an administrator has set it (flags bit 0 of the UEFI
payload). It describes the key's origin only.

#### Scenario: Driver-generated key

- **WHEN** the driver created the variable on this or an earlier boot
- **THEN** `cuid_seed_state` reads `unprovisioned`

#### Scenario: Administrator key

- **WHEN** 32 octets were written to `cuid_seed`
- **THEN** `cuid_seed_state` reads `provisioned` on every device and partition,
  including after a reboot

### Requirement: cuid_seed attribute

`cuid_seed` (0600, `CAP_SYS_ADMIN`) SHALL accept exactly 32 octets. The driver
SHALL first write them to the UEFI variable with the provisioned flag set; only
if that write succeeds SHALL it install the key and re-key every registered
component. A failed write SHALL return `-EIO` and change nothing. A read SHALL
return the 32-octet key in use. There is no default to restore.

#### Scenario: Key persists across reboot

- **WHEN** a key is written to `cuid_seed` and the host reboots
- **THEN** every `cuid_derived` equals its value before the reboot

#### Scenario: Firmware refuses the write

- **WHEN** the variable write fails
- **THEN** the write returns `-EIO`
- **AND** every `cuid_derived` is unchanged

## ADDED Requirements

### Requirement: cuid_unit_id attribute

Beside every `cuid_derived` the driver SHALL publish `cuid_unit_id`, mode
0444, the component's UnitID in decimal followed by a newline: `0` on the
device, the partition's UnitID on a partition.

#### Scenario: Readable without privilege

- **WHEN** an unprivileged user reads a partition's `cuid_unit_id`
- **THEN** it matches the UnitID packed into that partition's primary

## REMOVED Requirements

### Requirement: Default seed

**Reason**: There is no fixed default key.
**Migration**: The driver generates a random per-host key in UEFI; without UEFI
it publishes nothing.
