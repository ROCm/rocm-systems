## MODIFIED Requirements

### Requirement: UnitID identifies a sub-unit, not a location

A whole component SHALL have UnitID 0. A GPU spatial partition SHALL have
UnitID `(n << 6) | f`, where `f` is the first and `n` the number of logical
XCCs in the partition, so bits 0:5 hold the first XCC and bits 6:11 the count.
Bit 12 SHALL be set only for the SLC mode, which is otherwise the SPX range
(0x1200 on 8 XCCs), and SHALL be zero in every other mode. A partition whose logical XCC set is not contiguous
SHALL get no CUID. A partition's serial SHALL be its device's serial. The memory
partition mode SHALL NOT enter the identity. A CPU package SHALL have UnitID 0.

#### Scenario: Partition 0 differs between modes

- **WHEN** an 8-XCC device is switched SPX, DPX, QPX, CPX
- **THEN** partition 0 has UnitID 0x200, 0x100, 0x080 and 0x040 respectively
- **AND** its CUID differs in every mode

#### Scenario: Every partition in every mode is distinct

- **WHEN** all partitions of all compute modes of one device are enumerated
- **THEN** no two share a UnitID and none has UnitID 0

#### Scenario: A partition's UnitID is computable without the driver

- **WHEN** a tool knows that a device has `N` logical XCCs and that a mode
  splits it into `k` equal partitions
- **THEN** partition `i` has UnitID `((N / k) << 6) | (i * N / k)`
- **AND** no table published by the driver is needed to compute it
