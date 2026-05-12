#!/usr/bin/env python3

import os
import re
from collections import defaultdict

def extract_event_id_og(filename):
    """Extract event ID from og_pm4 filename format: SQ_XXXX_CMD_BUFFER_TYPE.txt"""
    match = re.match(r'SQ_([0-9a-fA-F]+)_CMD_BUFFER_(\w+)\.txt', filename)
    if match:
        return match.group(1).lower(), match.group(2).lower()
    return None, None

def extract_event_id_new(filename):
    """Extract event ID from new_pm4 filename format: SQ_0xXX_TYPE.txt"""
    match = re.match(r'SQ_0x([0-9a-fA-F]+)_(\w+)\.txt', filename)
    if match:
        return match.group(1).lower(), match.group(2).lower()
    return None, None

def parse_pm4_content(filepath):
    """Parse PM4 content and extract commands"""
    commands = []

    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        return [], f"Error reading file: {e}"

    # Split content by --- separators or parse line by line
    lines = content.strip().split('\n')

    current_command = None
    current_details = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip header lines in og_pm4 format
        if line.startswith('Event ID:') or line.startswith('Packet Type:') or \
           line.startswith('Line Number:') or line.startswith('Hex Data Length:') or \
           line.startswith('=') or line.startswith('PM4 DECODER OUTPUT'):
            continue

        # Check if this is a command line
        if line in ['EVENT_WRITE', 'SET_UCONFIG_REG', 'SET_SH_REG', 'SET_CONTEXT_REG']:
            # Save previous command if exists
            if current_command:
                commands.append({
                    'type': current_command,
                    'details': current_details.copy()
                })

            current_command = line
            current_details = []
        elif line == '---':
            # End of current command
            if current_command:
                commands.append({
                    'type': current_command,
                    'details': current_details.copy()
                })
                current_command = None
                current_details = []
        elif current_command and line.startswith('  '):
            # This is a detail line for the current command
            current_details.append(line)

    # Don't forget the last command if no --- at the end
    if current_command:
        commands.append({
            'type': current_command,
            'details': current_details.copy()
        })

    return commands, None

def normalize_hex(hex_str):
    """Normalize hex strings for comparison"""
    # Remove 0x prefix and convert to lowercase
    if hex_str.startswith('0x'):
        hex_str = hex_str[2:]
    return hex_str.lower()

def compare_commands(og_commands, new_commands):
    """Compare two command lists and return differences"""
    differences = []

    # Basic comparison
    if len(og_commands) != len(new_commands):
        differences.append(f"Command count mismatch: Original has {len(og_commands)}, Generated has {len(new_commands)}")

    # Compare commands one by one
    max_len = max(len(og_commands), len(new_commands))

    for i in range(max_len):
        og_cmd = og_commands[i] if i < len(og_commands) else None
        new_cmd = new_commands[i] if i < len(new_commands) else None

        if og_cmd is None:
            differences.append(f"Command {i+1}: Missing in original, Generated has {new_cmd['type']}")
            continue
        if new_cmd is None:
            differences.append(f"Command {i+1}: Missing in generated, Original has {og_cmd['type']}")
            continue

        # Compare command types
        if og_cmd['type'] != new_cmd['type']:
            differences.append(f"Command {i+1}: Type mismatch - Original: {og_cmd['type']}, Generated: {new_cmd['type']}")
            continue

        # Compare details
        if len(og_cmd['details']) != len(new_cmd['details']):
            differences.append(f"Command {i+1} ({og_cmd['type']}): Detail count mismatch - Original: {len(og_cmd['details'])}, Generated: {len(new_cmd['details'])}")

        for j, (og_detail, new_detail) in enumerate(zip(og_cmd['details'], new_cmd['details'])):
            if og_detail != new_detail:
                # Check if it's just a register address difference
                og_reg = re.search(r'Base Register: (0x[0-9a-fA-F]+)', og_detail)
                new_reg = re.search(r'Base Register: (0x[0-9a-fA-F]+)', new_detail)

                if og_reg and new_reg:
                    differences.append(f"Command {i+1} ({og_cmd['type']}): Register address mismatch - Original: {og_reg.group(1)}, Generated: {new_reg.group(1)}")
                else:
                    differences.append(f"Command {i+1} ({og_cmd['type']}): Detail mismatch - Original: '{og_detail}', Generated: '{new_detail}'")

    return differences

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    og_dir = os.path.join(script_dir, 'og_pm4')
    new_dir = os.path.join(script_dir, 'new_pm4')

    # Get all files
    og_files = [f for f in os.listdir(og_dir) if f.endswith('.txt') and not f.startswith('PROCESSING')]
    new_files = [f for f in os.listdir(new_dir) if f.endswith('.txt')]

    # Group files by event ID and packet type
    og_groups = defaultdict(dict)
    new_groups = defaultdict(dict)

    for filename in og_files:
        event_id, packet_type = extract_event_id_og(filename)
        if event_id and packet_type:
            og_groups[event_id][packet_type] = filename

    for filename in new_files:
        event_id, packet_type = extract_event_id_new(filename)
        if event_id and packet_type:
            new_groups[event_id][packet_type] = filename

    # Find matches and compare
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PM4 FILES COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("METHODOLOGY:")
    report_lines.append("This report compares PM4 decoder output between original (og_pm4) and")
    report_lines.append("generated (new_pm4) files. Files are matched by event ID and packet type.")
    report_lines.append("Comparisons focus on:")
    report_lines.append("- Command count and types")
    report_lines.append("- Register addresses and values")
    report_lines.append("- Command sequences")
    report_lines.append("")

    total_comparisons = 0
    files_with_differences = 0
    unmatched_files = []

    # Compare matched files
    all_event_ids = sorted(set(og_groups.keys()) | set(new_groups.keys()))

    for event_id in all_event_ids:
        og_packets = og_groups.get(event_id, {})
        new_packets = new_groups.get(event_id, {})

        all_packet_types = sorted(set(og_packets.keys()) | set(new_packets.keys()))

        for packet_type in all_packet_types:
            og_file = og_packets.get(packet_type)
            new_file = new_packets.get(packet_type)

            if og_file and new_file:
                report_lines.append(f"=== SQ Event 0x{event_id} - {packet_type.upper()} Packet ===")
                report_lines.append(f"Original: og_pm4/{og_file}")
                report_lines.append(f"Generated: new_pm4/{new_file}")
                report_lines.append("")

                # Parse and compare
                og_commands, og_error = parse_pm4_content(os.path.join(og_dir, og_file))
                new_commands, new_error = parse_pm4_content(os.path.join(new_dir, new_file))

                if og_error or new_error:
                    report_lines.append(f"ERROR: {og_error or new_error}")
                else:
                    differences = compare_commands(og_commands, new_commands)

                    if differences:
                        files_with_differences += 1
                        report_lines.append("Differences Found:")
                        for diff in differences:
                            report_lines.append(f"- {diff}")
                    else:
                        report_lines.append("No differences found - files are identical")

                total_comparisons += 1
                report_lines.append("")

            else:
                if og_file and not new_file:
                    unmatched_files.append(f"og_pm4/{og_file} (no corresponding new_pm4 file)")
                elif new_file and not og_file:
                    unmatched_files.append(f"new_pm4/{new_file} (no corresponding og_pm4 file)")

    # Summary
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append(f"Total file pairs compared: {total_comparisons}")
    report_lines.append(f"File pairs with differences: {files_with_differences}")
    report_lines.append(f"File pairs identical: {total_comparisons - files_with_differences}")
    report_lines.append("")

    if unmatched_files:
        report_lines.append("UNMATCHED FILES:")
        for unmatched in unmatched_files:
            report_lines.append(f"- {unmatched}")
    else:
        report_lines.append("All files were successfully matched and compared.")

    # Write report
    report_path = os.path.join(script_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Report generated: {total_comparisons} comparisons, {files_with_differences} with differences")
    print(f"Unmatched files: {len(unmatched_files)}")

if __name__ == '__main__':
    main()