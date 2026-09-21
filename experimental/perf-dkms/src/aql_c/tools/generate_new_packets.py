#!/usr/bin/env python3
"""
Script to generate new_packets.json using packet_gen_tool
"""

import json
import subprocess
import sys
import os

def run_packet_gen_tool(block, event_hex, packet_type):
    """
    Run packet_gen_tool and return the hex output

    Args:
        block: Block name (e.g., "GL2C")
        event_hex: Event ID in hex format (e.g., "0x29")
        packet_type: "start", "stop", or "read"

    Returns:
        Hex string output from the tool
    """
    cmd = [
        "./packet_gen_tool",
        "--continuous",
        "--quiet",
        f"--packet={packet_type}",
        "gfx12",
        "0x0",
        f"{block}:0:{event_hex}"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running packet_gen_tool for {block}:{event_hex} {packet_type}: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return None

def generate_packet_entry(block, event_hex, packet_type):
    """
    Generate a packet entry (with size and data fields)

    Returns:
        dict with "size" and "data" fields
    """
    hex_data = run_packet_gen_tool(block, event_hex, packet_type)
    if hex_data is None:
        return None

    # Calculate size in bytes (hex string length / 2)
    size = len(hex_data) // 2

    return {
        "size": size,
        "data": hex_data
    }

def main():
    # Load original packets to get the structure
    print("Loading original_packets.json...")
    with open("original_packets.json", "r") as f:
        original_data = json.load(f)

    # Create new data structure
    new_data = {}

    # Iterate through all blocks and events
    total_events = sum(len(events) for events in original_data.values())
    current = 0

    for block in sorted(original_data.keys()):
        print(f"\nProcessing block: {block}")
        new_data[block] = {}

        events = original_data[block]
        for event_id_str in sorted(events.keys(), key=int):
            current += 1
            event_id = int(event_id_str)
            event_hex = f"0x{event_id:x}"

            print(f"  [{current}/{total_events}] Processing event {event_id} ({event_hex})...", end=" ")

            # Generate start packets
            start_packet = generate_packet_entry(block, event_hex, "start")
            if start_packet is None:
                print(f"FAILED")
                continue

            # Generate stop packets
            stop_packet = generate_packet_entry(block, event_hex, "stop")
            if stop_packet is None:
                print(f"FAILED")
                continue

            # Generate read packets
            read_packet = generate_packet_entry(block, event_hex, "read")
            if read_packet is None:
                print(f"FAILED")
                continue

            # Create entry in new_data
            # Note: Each packet type has a single packet (not an array with multiple packets like original)
            new_data[block][event_id_str] = {
                "start_packets": [start_packet],
                "stop_packets": [stop_packet],
                "read_packets": [read_packet]
            }

            print(f"OK")

    # Write to new_packets.json
    print("\nWriting new_packets.json...")
    with open("new_packets.json", "w") as f:
        json.dump(new_data, f, indent=2)

    print("Done! Generated new_packets.json")
    print(f"\nSummary:")
    print(f"  Blocks: {len(new_data)}")
    for block, events in new_data.items():
        print(f"    {block}: {len(events)} events")

if __name__ == "__main__":
    main()
