## ADDED Requirements

### Requirement: The node key lives only in driver memory

amdgpu SHALL hold at most one node key for all its devices, in kernel memory. It
SHALL NOT generate a key, and SHALL NOT read or write a key in firmware, a file
or any other store. The key SHALL survive unbind and rebind while the module is
loaded, and SHALL be wiped when the module unloads.

#### Scenario: Fresh load

- **WHEN** amdgpu loads
- **THEN** no key is set
- **AND** every `cuid_derived` read fails with `ENODATA`

#### Scenario: Module reload

- **WHEN** a key was set and amdgpu is unloaded and loaded again
- **THEN** no key is set

### Requirement: cuid_seed sets and returns the key

`cuid_seed` SHALL exist on every amdgpu PCI device and on no partition, mode
0600, and SHALL require `CAP_SYS_ADMIN` for read and write. A write of exactly
32 octets SHALL install the key for every device and partition; a write of any
other length SHALL fail with `EINVAL` and change nothing. A read SHALL return
the 32-octet key, or fail with `ENODATA` when none is set.

#### Scenario: Setting the key

- **WHEN** root writes 32 octets to one GPU's `cuid_seed`
- **THEN** every GPU's and partition's `cuid_derived` is readable
- **AND** a read of any GPU's `cuid_seed` returns those octets

#### Scenario: Wrong length

- **WHEN** root writes 31 octets
- **THEN** the write fails with `EINVAL` and nothing changes

### Requirement: cuid_derived under the key

`cuid_derived` SHALL exist, mode 0444, on every amdgpu device and partition
that publishes `cuid_unit_id`. With a key it SHALL return the derived CUID
HMAC-SHA256(key, primary payload) in the derived layout; without one the read
SHALL fail with `ENODATA`. A partition created after the key was set SHALL
derive with it.

#### Scenario: Mode switch after setting the key

- **WHEN** a key is set and a GPU switches from SPX to DPX
- **THEN** both new partitions return a derived CUID

### Requirement: The library uses only cuid_seed

A root library SHALL take the key from an amdgpu `cuid_seed` and from nowhere
else, and SHALL treat `ENODATA` from `cuid_derived` as not published.
`amdcuid_set_hash_key()` SHALL write `cuid_seed` when amdgpu is loaded and
SHALL return `AMDCUID_STATUS_UNSUPPORTED` otherwise. The library SHALL NOT read
or write efivarfs or a key file.

#### Scenario: No key set

- **WHEN** root looks up a CPU with amdgpu loaded and no key set
- **THEN** it receives a temporary CUID

#### Scenario: Setting the key without amdgpu

- **WHEN** root calls `amdcuid_set_hash_key()` with no amdgpu device
- **THEN** it returns `AMDCUID_STATUS_UNSUPPORTED`
