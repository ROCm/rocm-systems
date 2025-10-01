#!/usr/bin/env python3
"""
Parse original_aql_out.txt and extract packet information organized by block and event.
"""

import json
import re
from collections import defaultdict

def parse_event_info(line):
    """Parse EVENT_INFO line to extract block name and event ID."""
    # Format: EVENT_INFO: event_id=X_block_name=Y_block_id=Z,...
    match = re.search(r'event_id=(\d+)_block_name=(\w+)_block_id=(\d+)', line)
    if match:
        event_id = match.group(1)
        block_name = match.group(2)
        block_id = match.group(3)
        return event_id, block_name, block_id
    return None, None, None

def parse_packet_line(line):
    """Parse a packet line to extract the hex data."""
    # Format: PACKET_TYPE: size=N data=HEXDATA
    match = re.search(r'(AQL_PACKET_\w+|CMD_BUFFER_\w+):\s+size=(\d+)\s+data=([0-9a-fA-F]+)', line)
    if match:
        packet_type = match.group(1)
        size = match.group(2)
        data = match.group(3)
        return packet_type, size, data
    return None, None, None

def main():
    input_file = '/home/ben/original_aql_out.txt'
    output_file = '/home/ben/rocm-systems/perf-pmu-stub/src/aql_c/tools/original_packets.json'

    # Structure: blocks -> events -> packet_types -> packets
    result = defaultdict(lambda: defaultdict(lambda: {
        'start_packets': [],
        'stop_packets': [],
        'read_packets': []
    }))

    current_event_id = None
    current_block_name = None
    current_block_id = None

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check if this is an EVENT_INFO line
            if 'EVENT_INFO:' in line:
                event_id, block_name, block_id = parse_event_info(line)
                if event_id and block_name:
                    current_event_id = event_id
                    current_block_name = block_name
                    current_block_id = block_id
                continue

            # Check if this is a packet line
            packet_type, size, data = parse_packet_line(line)
            if packet_type and current_event_id and current_block_name:
                packet_info = {
                    'size': int(size),
                    'data': data
                }

                # Categorize the packet
                if 'START' in packet_type:
                    result[current_block_name][current_event_id]['start_packets'].append(packet_info)
                elif 'STOP' in packet_type:
                    result[current_block_name][current_event_id]['stop_packets'].append(packet_info)
                elif 'READ' in packet_type:
                    result[current_block_name][current_event_id]['read_packets'].append(packet_info)

    # Convert defaultdict to regular dict for JSON serialization
    output_data = {}
    for block_name, events in sorted(result.items()):
        output_data[block_name] = {}
        for event_id, packets in sorted(events.items(), key=lambda x: int(x[0])):
            output_data[block_name][event_id] = {
                'start_packets': packets['start_packets'],
                'stop_packets': packets['stop_packets'],
                'read_packets': packets['read_packets']
            }

    # Write to JSON file with proper formatting
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Parsed packets successfully!")
    print(f"Output written to: {output_file}")
    print(f"\nSummary:")
    for block_name in sorted(output_data.keys()):
        event_count = len(output_data[block_name])
        print(f"  Block '{block_name}': {event_count} events")

if __name__ == '__main__':
    main()
