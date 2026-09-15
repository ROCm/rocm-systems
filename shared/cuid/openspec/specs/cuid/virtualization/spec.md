## Purpose

How the model behaves under SR-IOV and hardware virtualisation: what the
hypervisor generates, what the guest sees, and what has to be tracked across
live migration.

_Source: "Persistent platform component identification for SW tools", version 84._

## Requirements

### Requirement: A projected identifier is primary inside the guest

Where platform security policy does not permit direct hardware identification,
or where a device presents SR-IOV Virtual Functions, the hypervisor SHALL
generate a VM-specific hardware fingerprint (a PCIe device serial number or
platform/component UUID) and project it into the guest.

That value is a Derived CUID from the hypervisor's point of view, but SHALL
operate as a **primary** ID for the VM context's guest OS.

The guest OS kernel SHALL generate the VM-specific primary CUID from the VF
device information projected into the VM, and subsequently an application-level
non-privileged VM-specific Derived CUID.

The approach in the guest SHALL be identical to what the guest would do on bare
metal, keeping one approach for system software and tools.

#### Scenario: The guest does not special-case itself

- **WHEN** a guest OS generates a CUID from a projected VF
- **THEN** it uses the same algorithm it would use on bare metal

#### Scenario: The hypervisor holds the mapping

- **WHEN** a hypervisor projects a VM-specific fingerprint
- **THEN** it maintains a mapping from that value to the physical primary CUID
  it manages

### Requirement: The mapping survives live migration

Where a VM is live-migrated across a provider's physical infrastructure, the
provider SHALL maintain an updated mapping so that the migration can be tracked,
and SHALL keep projecting the same VM-specific hardware fingerprint into the VM
so that it remains accessible by the same methods as on bare metal.

#### Scenario: A migrated VM sees no change

- **WHEN** a VM is migrated between physical hosts
- **THEN** the fingerprint projected into it is unchanged
- **AND** the provider's mapping records the new physical component

### Requirement: Network adapter virtualisation

Device virtualisation for a network adapter SHALL follow the same principles as
for any other guest VM identification.

#### Scenario: A virtualised NIC

- **WHEN** a NIC is presented to a guest through SR-IOV
- **THEN** the guest identifies it the same way it would on bare metal
