## MODIFIED Requirements

### Requirement: Staged lookup, kernel first

The library SHALL answer a lookup from the driver's attributes when the driver
publishes them (source `DRIVER`), and otherwise SHALL compute it (source
`LIBRARY`). There is no record store and no daemon. A permanent identity SHALL
be computed only by a root caller holding the node key; every other caller, and
every caller on a host with no key, SHALL receive a temporary CUID. A GPU
partition with no driver identity SHALL be refused, not given a temporary CUID.

#### Scenario: Non-root CPU lookup

- **WHEN** an unprivileged caller asks for a CPU's CUID
- **THEN** it receives a temporary CUID with bit 117 set and source `LIBRARY`

### Requirement: Key handling in the library

A root library SHALL obtain the node key from any amdgpu `cuid_seed`, otherwise
from the UEFI variable through efivarfs, validating the 36-octet payload. It
SHALL NOT read a key file, SHALL NOT write the variable outside
`amdcuid_set_hash_key()`, and SHALL NOT hold a fallback key.

#### Scenario: Variable not yet visible

- **WHEN** the driver created the variable after efivarfs was mounted
- **THEN** the library takes the key from `cuid_seed`

## ADDED Requirements

### Requirement: Only AMD GPUs are GPU components

The library SHALL enumerate as GPUs only PCI functions with Vendor ID `0x1002`.
Another vendor's display device, such as a BMC's, SHALL NOT be listed and SHALL
NOT receive a temporary CUID.

#### Scenario: BMC display controller

- **WHEN** a host has an ASPEED BMC display device (`1a03:2000`) beside AMD GPUs
- **THEN** discovery lists the AMD GPUs and not the BMC device

### Requirement: amd-smi lists every component

amd-smi SHALL expose the library's whole inventory, not only the GPUs it
manages: `amdsmi_get_cuid_components()` SHALL return every component with a
CUID, with its type, source, temporary flag, BDF and sysfs path, ordered by
type, and `amd-smi node --cuid` SHALL print it with the node key's state. The
primary CUIDs SHALL be included only with `--cuid-primary`, and only for root.

#### Scenario: Root lists the node

- **WHEN** root runs `amd-smi node --cuid` on a host with a node key
- **THEN** the platform, each CPU package, each AMD GPU and each NIC is listed
- **AND** none of them is temporary

#### Scenario: Unprivileged caller

- **WHEN** a non-root user runs `amd-smi node --cuid`
- **THEN** CPU, NIC and platform CUIDs are temporary and the key state is
  reported as needing root
