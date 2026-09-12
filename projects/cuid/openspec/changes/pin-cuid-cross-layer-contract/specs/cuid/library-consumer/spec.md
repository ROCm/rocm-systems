## Purpose

The userspace CUID library's obligations as a consumer of the privileged identity
rather than a second, independent producer of it: the lookup order, the
device-type enumeration it exposes, and what it may compute for itself.

## ADDED Requirements

### Requirement: Staged lookup, kernel first

The library SHALL obtain a component's CUID by trying these sources in order and
stopping at the first that answers:

1. The device driver's published interface, where the device is supported and
   the interface is present.
2. A local daemon or configuration store, where one is installed.
3. Its own computation, which SHALL follow the serial precedence stated by the
   `component-sources` requirement "Serial precedence for PCIe components": a
   driver-supplied serial, then the PCIe Device Serial Number extended
   capability, then a vendor-specific capability. Only where none of those
   yields a non-zero value may the library fall back to an auxiliary CUID.

Where this requirement and `component-sources` could be read as differing about
what stage 3 does, `component-sources` governs. It states the ladder; this
requirement only places that ladder after the driver and daemon stages.

The library SHALL NOT recompute a primary or derived CUID for a component whose
driver already publishes one, even when it has the inputs to do so.

Two producers that independently compute the same value will eventually disagree,
and nothing about the values themselves reveals it. The kernel is the only layer
that can read the privileged identity, so it is the authority.

#### Scenario: A supported GPU is read from the driver

- **WHEN** the library is asked for the CUID of a GPU whose driver publishes one
- **THEN** the value the library returns is the value the driver published
- **AND** the library does not read PCI configuration space to construct it

#### Scenario: An unsupported device falls through

- **WHEN** the library is asked for the CUID of a device whose driver publishes
  nothing and for which no daemon entry exists
- **THEN** the library works the `component-sources` serial ladder for that
  component before considering an auxiliary value

#### Scenario: A device with a reachable serial is not made auxiliary

- **WHEN** no driver and no daemon answer, but the component's PCIe Device
  Serial Number capability yields a non-zero value
- **THEN** the library computes a primary CUID from that serial

#### Scenario: A device with no reachable serial becomes auxiliary

- **WHEN** no driver and no daemon answer and no source in the
  `component-sources` ladder yields a non-zero value
- **THEN** the library computes an auxiliary CUID and marks it with bit 117

The driver stage SHALL be attempted for every component that has a PCI routing
ID, not only for the component types whose drivers publish a CUID today. It is a
property of the bus, not of one driver: a NIC or an NPU whose driver grows the
same attributes must be answered by the kernel from that moment on, with no
further change in the library. A component with no routing ID, meaning the CPU
and the platform, SHALL fall straight through to the later stages.

#### Scenario: A NIC whose driver publishes a CUID is not recomputed

- **WHEN** a NIC's driver publishes the CUID attributes and the library is asked
  for that NIC's CUID
- **THEN** the library returns the driver's value

#### Scenario: A CPU does not attempt the driver stage

- **WHEN** the library is asked for a CPU's CUID
- **THEN** it proceeds directly to the daemon and then to its own computation

#### Scenario: Divergence is impossible where the driver answers

- **WHEN** a driver-published CUID and a library-reported CUID for the same
  component are compared
- **THEN** they are identical, because they are the same value

### Requirement: Device type enumeration uses the on-wire values

The library's public device-type enumeration SHALL use the specification's
on-wire values:

| Name | Value |
|---|---|
| Platform | `0x0` |
| CPU | `0x1` |
| GPU | `0x2` |
| NIC | `0x3` |
| NPU | `0x4` |

A sentinel or "none" value SHALL NOT occupy a value that a conforming Component
Type field can hold; it SHALL be `0xFF`.

The enumeration SHALL be usable as the Component Type field directly, without a
translation step, and the interface that writes it SHALL accept the enumeration
type rather than a plain integer, so that a raw value cannot be passed by
mistake.

*This is a change to a published API. The current enumeration is offset by one:
it defines `GPU = 0x3`, the library casts it straight onto the wire, and a
conforming reader decodes the result as a NIC. Every library-emitted GPU CUID
changes as a result, which is correct, because the current ones misname the
device.*

#### Scenario: A GPU is emitted as type 2

- **WHEN** the library generates a primary CUID for a GPU
- **THEN** the Component Type field contains `0x2`

#### Scenario: The sentinel is out of range

- **WHEN** the enumeration's sentinel value is inspected
- **THEN** it is `0xFF`, which no 4-bit Component Type field can hold

#### Scenario: A raw integer is not accepted

- **WHEN** an integer that is not a member of the enumeration is passed where a
  device type is expected
- **THEN** the program does not compile

### Requirement: Temporary values are identifiable through the public interface

Where the library's documentation states that a CUID can be identified as
temporary or auxiliary, that SHALL be true through the public interface: a caller
SHALL be able to determine it without decoding the payload itself.

A producer SHALL NOT document a marker it does not implement.

#### Scenario: A caller can ask

- **WHEN** a caller holds an auxiliary CUID obtained from the library
- **THEN** the library reports it as auxiliary through a documented query

#### Scenario: Stored and recomputed values agree

- **WHEN** an auxiliary CUID is written to the library's on-disk record and read
  back
- **THEN** the value read back is bit-for-bit identical to the value written,
  including bit 117

*The round-trip currently loses this: the writer stores five hash bits in the
octet that holds bit 117 and the reader recovers six, so a stored auxiliary value
reads back with a hash bit that was never in it.*

### Requirement: Key handling in the library

The library SHALL treat a key file whose size is not exactly 32 octets as
corruption rather than absence, SHALL refuse it, and SHALL NOT silently fall back
to the canonical seed in that case.

Where no key file exists at all, the library SHALL use the canonical fallback
seed, which is the same 24 octets the kernel uses, so that an unprovisioned
machine reports the same derived CUID from both layers.

The library SHALL NOT proceed with a null or unset key. It SHALL report an error.

The library SHALL propagate the outcome of a key-provisioning call to its caller.

#### Scenario: A truncated key file is refused

- **WHEN** the key file is 16 octets
- **THEN** the library reports a key error and derives nothing

#### Scenario: An absent key file uses the canonical seed

- **WHEN** no key file exists
- **THEN** the library derives under the canonical fallback seed
- **AND** its derived CUIDs equal the driver's

#### Scenario: Provisioning failure is reported

- **WHEN** a caller provisions a key and the store cannot be written
- **THEN** the call reports failure rather than success

### Requirement: Only SHA-256 is accepted

The library SHALL accept only SHA-256 as the derivation digest and SHALL reject
any request to select another, including at the point of selection rather than at
first use.

A larger digest does not fit the derived layout, and accepting the request while
sizing buffers for SHA-256 is a memory-safety defect as well as a conformance
one.

#### Scenario: A larger digest is rejected

- **WHEN** a caller selects SHA-512
- **THEN** the selection fails
- **AND** no derivation is attempted
