# Packet Generation Summary

## Overview
Generated `new_packets.json` using `packet_gen_tool` with the `--continuous` flag to create packets matching the structure of `original_packets.json`.

## Results
- **Successfully Generated**: 31 out of 36 events (86%)
- **Failed to Generate**: 5 events (14%)

### Generated Events by Block
- **GL2C**: 9/9 events (100%)
- **GRBM**: 2/2 events (100%)
- **SQ**: 20/22 events (91%)
- **TA**: 0/3 events (0%)

### Missing Events
The following events could not be generated due to validation failures:

1. **SQ Block**
   - Event 288 (0x120)
   - Event 293 (0x125)

2. **TA Block**
   - Event 15 (0xf) - Maps to TA_TA_BUSY
   - Event 45 (0x2d) - Maps to TA_BUFFER_LOAD_WAVEFRONTS
   - Event 46 (0x2e) - Maps to TA_BUFFER_STORE_WAVEFRONTS

**Reason for Failure**: Counter collection validation errors. These events may have been deprecated, require special configuration, or the validation rules have changed since original_packets.json was created.

## Key Differences Between Original and New Packets

### 1. Packet Structure
- **Original**: Each packet type (start/stop/read) contains 2 packets
  - Packet 0: PM4 packet
  - Packet 1: AQL dispatch packet (64 bytes)
- **New**: Each packet type contains only 1 packet
  - Packet 0: PM4 packet only (no AQL packet)

### 2. Packet Sizes
- **Original Start Packets**: 496 bytes (PM4) + 64 bytes (AQL)
- **New Start Packets**: 100 bytes (PM4 only) for GL2C/GRBM, 124 bytes for SQ

The size difference is due to:
- Original packets configure 16 counter instances
- New packets configure only 1 counter instance (using BLOCK:0:EVENT format)

### 3. Register Addressing
- **Original**: Uses register offset 0x1b80 for GL2C counters
- **New**: Uses register offset 0xbb80 for GL2C counters
  - Bit 15 is set in new packets (0x8000 bit difference)
  - This may indicate a different mode or configuration

### 4. Read Packet Sizes
- **Original**: 1228 bytes (includes reads for 16 instances)
- **New**: 112 bytes for GL2C/GRBM, 1240 bytes for SQ
  - SQ read packets are larger due to multiple counter reads

## Tool Usage

### Command Format
```bash
./packet_gen_tool --continuous --quiet --packet=TYPE gfx12 0x0 BLOCK:INDEX:EVENT
```

### Parameters
- `--continuous`: Output as continuous hex string (no spaces, no 0x prefix)
- `--quiet`: Suppress all output except packet data
- `--packet=TYPE`: Select packet type (start, stop, or read)
- `gfx12`: GPU architecture
- `0x0`: Base address (used 0x0 for all)
- `BLOCK:INDEX:EVENT`: Counter specification
  - BLOCK: Hardware block name (GL2C, GRBM, SQ, TA)
  - INDEX: Instance index (always 0 for single instance)
  - EVENT: Event ID in hex format (e.g., 0x29 for event 41)

### Examples
```bash
# Generate start packet for GL2C event 41
./packet_gen_tool --continuous --quiet --packet=start gfx12 0x0 GL2C:0:0x29

# Generate stop packet for SQ event 1
./packet_gen_tool --continuous --quiet --packet=stop gfx12 0x0 SQ:0:0x1

# Generate read packet for GRBM event 0
./packet_gen_tool --continuous --quiet --packet=read gfx12 0x0 GRBM:0:0x0
```

## File Locations
- **Input**: `/home/ben/rocm-systems/perf-pmu-stub/src/aql_c/tools/original_packets.json`
- **Output**: `/home/ben/rocm-systems/perf-pmu-stub/src/aql_c/tools/new_packets.json`
- **Generator Script**: `/home/ben/rocm-systems/perf-pmu-stub/src/aql_c/tools/generate_new_packets.py`

## Notes
1. The `--continuous` flag in the task description refers to continuous hex output format, not continuous/streaming sampling mode
2. The packet_gen_tool does not generate AQL dispatch packets - only PM4 packets
3. Some events in the original_packets.json may have been generated with different validation rules or may no longer be valid
4. The generated packets use a single counter instance configuration, whereas the original used 16 instances
