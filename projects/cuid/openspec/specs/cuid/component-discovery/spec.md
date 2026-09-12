## Purpose

Where the architectural values that feed a PrimaryID actually come from, per
component class and per operating system.

_Source: "Persistent platform component identification for SW tools", version 84._

## Requirements

### Requirement: PCIe and CXL device serial number

The PCI Express standard provides an optional 64-bit Device Serial Number
capability in extended configuration space, Extended Capability ID `0003h`.

The capability SHALL be defined in the physical function 0 configuration space
of the device, and SHOULD be mirrored across all subfunctions of a multifunction
device.

Software SHALL generate the PrimaryID from the values retrieved from the
device's PCIe configuration space.

It is highly recommended that this extended capability be added to any AMD
device's configuration space that does not already have it.

> **Recorded gap (C18).** The capability holds two 32-bit dwords and the page
> never states their byte order, nor where in the capability the serial dwords
> begin. Two producers reading the same eight octets off the same card can pack
> two byte-reversed serials, and neither is detectably wrong on its own.
> Resolved in `openspec/changes/amend-published-cuid-spec/`.

#### Scenario: The serial is read from the DSN capability

- **WHEN** a PCIe endpoint exposes Extended Capability ID `0003h`
- **THEN** its 64-bit Device Serial Number is used in payload bits 0:63

### Requirement: Permitted substitutes for an absent Device Serial Number

Where a PCIe device exposes no Device Serial Number Extended Capability, it is
**permitted but not recommended** to substitute:

1. a value retrieved from the Vendor-Specific Capability entry, where defined;
2. for a network device, NVMe, or another device with a device-class-specific ID
   mechanism, its MAC address or other individual ID.

The Device Serial Number Extended Capability SHALL take precedence over both
substitutes: a substitute SHALL be used only where that capability is absent.

Retrieving either alternate value requires vendor-specific, proprietary code and
therefore SHOULD be avoided where possible. It is permitted as a fallback.

> **Recorded gap (C19).** A MAC address is six octets and the page never states
> its orientation in the 64-bit serial field, nor what to do with an all-zero
> address. Resolved in `openspec/changes/amend-published-cuid-spec/`.

#### Scenario: A NIC without a DSN

- **WHEN** a network adapter exposes no Device Serial Number
- **THEN** its MAC address may be used in place of it

### Requirement: Operating system access paths

Both Linux and Windows SHALL be able to retrieve PCIe configuration space
information in kernel mode and in application user space: on Linux through
`/sys`, on Windows through SetupAPI and WMI.

Either OS MAY require super-user or administrator privilege to reach some of
that information from a guest OS or from application level.

#### Scenario: Unprivileged access may be refused

- **WHEN** an unprivileged application reads configuration space
- **THEN** the OS may refuse, depending on configuration

### Requirement: CPU identification

The processor serial number is typically not enabled by default for software
readout and requires a firmware option.

On AMD CPUs the feature is called PPIN. Its presence SHALL be identified by
`CPUID Fn8000_0008.EBX[23]`; a `1` indicates the Processor Inventory /
Identification Number is present. The OEM processor serial number is exposed
architecturally through a Machine Specific Register named PPIN. Intel provides a
similar serial number readable by privileged code.

Where the CPU exposes no individual number, software MAY use:

- the SMBIOS platform ID as a proxy value; or
- CPUID Family, Model, Revision ID and Microcode version, together with the APIC
  ID and the ACPI SRAT table identification, to determine an individual
  core/processor ID and the NUMA topology.

The second is explicitly noted as **ambiguous**, because multiple processors in
different systems may present similar values.

#### Scenario: PPIN is unavailable

- **WHEN** PPIN is not enabled in firmware
- **THEN** the platform serial number is recommended as the association instead

### Requirement: Platform identification

The platform SHALL be identified from SMBIOS tables.

On UEFI systems the SMBIOS base address is found in the EFI configuration table
under SMBIOS GUID `{EB9D2D31-2D88-11D3-9A16-0090273FC14D}`.

- The **System Information (Type 1)** structure provides the platform serial
  number at offset `07h` and the platform UUID at offset `08h`.
- The **Baseboard (Type 2)** structure provides a baseboard/motherboard serial
  number at offset `07h`.

Together these identify the individual platform and motherboard, and by
inference most soldered components on the motherboard.

On Linux the table is retrievable through `/sys/firmware/dmi`. On Windows,
through WMI.

The **Platform Component UID SHALL be the SMBIOS UUID value used directly.**

#### Scenario: The platform CUID is the SMBIOS UUID

- **WHEN** SMBIOS reports a system UUID
- **THEN** that value is the Platform CUID, used directly

### Requirement: AMD proprietary device identification

All newer AMD GPU and CPU SOC products contain an embedded security processor
(PSP/ASP) which can identify a particular SOC through a proprietary register
interface, exposing a 64-bit individual chip identifier.

An internal AMD-proprietary library (swPSP) provides access to PSP services from
untrusted kernel-mode components. The interface is under NDA; support for
accessing the PSP has been upstreamed into Linux and other operating systems.

This information is available on all current and past AMD components but MAY NOT
be exposable in all scenarios. Software SHALL therefore treat the chip
identifier as an optional source: it SHALL NOT require PSP access in order to
produce a PrimaryID, and SHALL use another architectural source where the
identifier is not reachable.

#### Scenario: The chip identifier is not always reachable

- **WHEN** the deployment does not permit PSP access
- **THEN** the 64-bit chip identifier is unavailable and another source is used
