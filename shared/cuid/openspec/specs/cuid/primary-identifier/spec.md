## Purpose

The PrimaryID: the 122-bit payload built from a component's architectural
hardware values, and the component-type enumeration that says which values those
are.

_Source: "Persistent platform component identification for SW tools", version 84._

## Requirements

### Requirement: Primary payload layout for PCIe devices

The PrimaryID payload for a PCIe device SHALL be 122 bits, laid out as:

| Bit | Name | Comments |
|---|---|---|
| 0:63 | Device / Platform / Component / CPU serial number | From the PCIe DSN capability, SMBIOS platform serial number, ACPI, or another architectural component identifier. Where none is available, a software-generated fingerprint from a proprietary individual marker. An integer, MSB zero-extended if smaller than 64 bits and truncated if larger. |
| 64:71 | UnitID (part 1) | Zero if the component does not support UnitID |
| 72:79 | RevisionID | Zero if the component does not support a revision ID |
| 80:95 | DeviceID | From PCIe configuration space or SMBIOS ID. For a CPU, Family and Model combined. |
| 96:111 | VendorID | From PCIe configuration space or SMBIOS vendor ID, following the PCIe 16-bit vendor ID definition (e.g. `0x1022` or `0x1002` for AMD) |
| 112:117 | UnitID (part 2) | Used if the UnitID value is >= 256, zero otherwise |
| 118:121 | Component Type | See the component type table |

> **Recorded contradiction (C10).** This table gives UnitID part 2 six bits, 112:117.
> The Derived payload table in `derived-identifier` assigns bit 117 to the
> Auxiliary Value Identifier. Bit 117 therefore has two owners across the two
> tables, and a producer cannot satisfy both. Resolved in
> `openspec/changes/amend-published-cuid-spec/`.

#### Scenario: A serial shorter than 64 bits

- **WHEN** the architectural serial is narrower than 64 bits
- **THEN** it is MSB zero-extended into bits 0:63

#### Scenario: A serial longer than 64 bits

- **WHEN** the architectural serial is wider than 64 bits
- **THEN** it is truncated

> **Recorded contradiction (C1).** This rule, and the Platform Component UID
> rule in `component-discovery`, hand back a firmware-authored UUID unchanged.
> A firmware system UUID is in practice version 1, 3 or 4, never 8. That
> contradicts the Compatibility requirement in `identifier-model` ("represented
> as a UUID version 8 value") and the framing requirement in
> `uuidv8-representation` ("the same format SHALL be used for both Primary and
> Derived identifier generation"). A consumer that validates the version nibble
> rejects every Platform CUID. Resolved in
> `openspec/changes/amend-published-cuid-spec/`.

#### Scenario: A component-provided UUID pre-empts the layout

- **WHEN** an ACPI device object or an SMBIOS entry already provides a UUID
  specific to the individual component
- **THEN** that UUID value is used as the PrimaryID directly

### Requirement: UnitID identifies a sub-unit

UnitID SHALL identify a sub-differentiation of the component, used by a driver
to identify a particular independent operating unit and so to create a virtual
PrimaryID where no other architectural differentiation is available.

A UnitID of zero SHALL mean either that UnitID is not used for that component
type, or that the value references the whole device as identified via PCIe or
ACPI.

The vendor driver at the privileged level MAY define other UnitID values for
subdivided component functions rooted in the parent device definition.

#### Scenario: The whole device

- **WHEN** a CUID names the device as enumerated by PCIe
- **THEN** its UnitID is zero

#### Scenario: A subdivided function

- **WHEN** a driver names a spatial partition or a Virtual Function of a
  physical device
- **THEN** it assigns a non-zero UnitID rooted in the parent device definition

### Requirement: Component type enumeration

Component Type SHALL take one of the following values, each of which also
determines where the other parameters come from:

| Value | Component | Sources |
|---|---|---|
| 0 | Platform | VendorID indicates OEM or SI |
| 1 | CPU | VendorID: CPU vendor. DeviceID: on x86-64, Family and Model from CPUID; otherwise architecture-specific core type from ACPI PPTT or boot tree. RevisionID: on x86-64, CPU RevID. Serial from Platform Serial # or an architectural instruction (on AMD, PPIN). UnitID from APIC ID or equivalent per core. |
| 2 | GPU | VendorID, DeviceID from PNPID (PCIe configuration space, ACPI device object). UnitID from partition ID. |
| 3 | NIC | VendorID, DeviceID from PNPID. If no Device Serial Number is defined, the MAC address value can be used for that field instead. |
| 4 | NPU | VendorID, DeviceID from PNPID |
| 5 | Storage | VendorID, DeviceID from PNPID |
| 6 | Memory | VendorID, DeviceID from UEFI/ACPI objects, typically populated from SPD ROM or CXL |
| 7 | GenPCIe | Any general PCIe or CXL device not identified by another type. UnitID from VF where applicable. |
| 8 | GenC | Any software-visible non-PCIe component, e.g. in an I/O die, not identified by another type |
| 9 | RackTray | Rack slot/tray identifier. VendorID: PNPID OEM/ODM vendor. DeviceID: vendor-defined model ID for the slot/tray. RevisionID: vendor-defined model revision. |
| 0x0A | Rack | Rack identifier, fields as RackTray but for the rack |
| 0x0B–0x0E | Reserved | For future component definition |
| 0x0F | Other | VendorID, DeviceID from PNPID, not otherwise specified |

#### Scenario: A GPU is type 2

- **WHEN** a GPU's PrimaryID is built
- **THEN** its Component Type field is `2`

#### Scenario: A NIC with no Device Serial Number

- **WHEN** a NIC exposes no PCIe Device Serial Number
- **THEN** its MAC address value may be used in bits 0:63 instead
