# Changelog for rocminfo

## Since last release ROCm 7.12

### Resolved issues

* Fixed BDF (Bus:Device.Function) ID truncation issue that caused incorrect display of PCI device identifiers. The `bdf_id` field was incorrectly declared as `uint16_t` instead of `uint32_t`, causing silent truncation when HSA runtime returned the full 32-bit BDF ID value. This has been corrected to properly display complete BDF information for all GPU agents.
