## MODIFIED Requirements

### Requirement: Staged lookup, kernel first

The library SHALL take a primary CUID from the driver's attributes when the
driver publishes them, and otherwise SHALL compute it. There is no record
store and no daemon. Without a node key, every derived CUID SHALL be a
temporary CUID computed by the library (source `LIBRARY`), for every caller. A
GPU partition with no derived identity SHALL be refused, not given a temporary
CUID.

#### Scenario: Non-root CPU lookup

- **WHEN** an unprivileged caller asks for a CPU's CUID
- **THEN** it receives a temporary CUID with bit 117 set and source `LIBRARY`

#### Scenario: Root and other users agree on a GPU

- **WHEN** root and an unprivileged caller ask for the same whole GPU's derived
  CUID on a host without a node key
- **THEN** both receive the same temporary CUID

### Requirement: Key handling in the library

The library SHALL NOT read, store or write a node key, and SHALL NOT hold a
fallback key. `amdcuid_set_hash_key()` and `amdcuid_get_key_info()` SHALL
return `AMDCUID_STATUS_UNSUPPORTED`.

#### Scenario: Setting a key

- **WHEN** root calls `amdcuid_set_hash_key()`
- **THEN** it returns `AMDCUID_STATUS_UNSUPPORTED` and nothing is written

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
type, and `amd-smi node --cuid` SHALL print it. Every PCI function with a CUID
SHALL be listed, whatever the caller's privilege, and functions that share a
serial number SHALL each be listed with their own CUID. Two components whose
derived CUIDs coincide anyway SHALL both be left out, since a CUID that names
two components identifies neither, and the rest of the inventory SHALL still be
returned. A GPU partition's BDF SHALL be the address amd-smi uses for its handle.
The primary CUIDs SHALL be
included only with `--cuid-primary`, and only for root.

#### Scenario: Any caller lists the node

- **WHEN** a user runs `amd-smi node --cuid` on a host without a node key
- **THEN** the platform, each CPU package, each AMD GPU, each NPU and each NIC
  function is listed
- **AND** each of them is temporary

#### Scenario: Two NIC functions that share a serial number

- **WHEN** both functions of a two-port NIC read the same PCIe Device Serial
  Number
- **THEN** root and every other caller list both, each with its own BDF, sysfs
  path and CUIDs

#### Scenario: Two components whose CUIDs coincide

- **WHEN** two components derive the same CUID
- **THEN** neither is listed, a lookup of either is refused, and every other
  component is listed
