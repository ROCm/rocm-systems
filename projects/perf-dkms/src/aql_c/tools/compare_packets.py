#!/usr/bin/env python3
"""
PM4 Packet Comparison Tool

This script compares PM4 packets for a specific block and event between two JSON files
(original_packets.json and new_packets.json). It decodes the packets using pm4_decoder
and displays the differences in a clear, readable format.

Usage:
    python compare_packets.py <block_name> <event_id>

Examples:
    python compare_packets.py GL2C 41
    python compare_packets.py SQ 0x42
    python compare_packets.py TA 100
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ANSI color codes for better output formatting
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def parse_event_id(event_id_str: str) -> str:
    """
    Parse event ID from command line argument.
    Handles both decimal (e.g., "41") and hexadecimal (e.g., "0x29") formats.
    Returns the decimal string representation for JSON lookup.

    Args:
        event_id_str: Event ID as string (decimal or hex)

    Returns:
        Event ID as decimal string
    """
    try:
        # Handle hexadecimal format (0x...)
        if event_id_str.lower().startswith('0x'):
            event_id = int(event_id_str, 16)
        else:
            event_id = int(event_id_str, 10)
        return str(event_id)
    except ValueError:
        raise ValueError(f"Invalid event ID format: {event_id_str}. "
                       "Use decimal (e.g., '41') or hex (e.g., '0x29')")


def load_json_file(filepath: Path) -> Dict:
    """
    Load and parse a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Parsed JSON data as dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {filepath}: {e.msg}",
                                  e.doc, e.pos)


def get_block_data(json_data: Dict, block_name: str, event_id: str) -> Optional[Dict]:
    """
    Extract packet data for a specific block and event from JSON.

    Args:
        json_data: Parsed JSON data
        block_name: Name of the block (e.g., "GL2C", "SQ")
        event_id: Event ID as decimal string

    Returns:
        Dictionary containing start_packets, stop_packets, and read_packets,
        or None if block/event not found
    """
    # Check if block exists in JSON
    if block_name not in json_data:
        return None

    # Check if event exists in block
    if event_id not in json_data[block_name]:
        return None

    return json_data[block_name][event_id]


def is_valid_pm4_packet(hex_data: str) -> bool:
    """
    Check if a hex string starts with a valid PM4 Type 3 packet header.

    Args:
        hex_data: Hex string representing packet data

    Returns:
        True if the first DWORD has PM4 Type 3 signature (0xCxxxxxxx)
    """
    if len(hex_data) < 8:
        return False

    try:
        # Get first DWORD (first 8 hex chars)
        first_dword = int(hex_data[0:8], 16)
        # Check for PM4 Type 3 signature: bits 31-30 should be 11 (0xC0000000)
        return (first_dword & 0xC0000000) == 0xC0000000
    except ValueError:
        return False


def write_packets_to_hex_file(packets: List[Dict], filepath: Path) -> None:
    """
    Write packet data to a hex file for pm4_decoder processing.
    Filters out non-PM4 packets (metadata, auxiliary data, etc.)

    Args:
        packets: List of packet dictionaries with 'data' field
        filepath: Path where hex file should be written
    """
    with open(filepath, 'w') as f:
        for packet in packets:
            # Only write valid PM4 packets (skip metadata/auxiliary packets)
            if is_valid_pm4_packet(packet['data']):
                f.write(packet['data'] + '\n')


def decode_packets(hex_filepath: Path, decoder_path: Path) -> Tuple[bool, str]:
    """
    Decode packets using pm4_decoder.

    Args:
        hex_filepath: Path to the hex file containing packet data
        decoder_path: Path to the pm4_decoder executable

    Returns:
        Tuple of (success: bool, output: str)
    """
    if not decoder_path.exists():
        return False, f"pm4_decoder not found at: {decoder_path}"

    try:
        # Run pm4_decoder with --rebase 0x0 to normalize memory addresses, reading from stdin
        with open(hex_filepath, 'r') as f:
            result = subprocess.run(
                [str(decoder_path), '--rebase', '0x0'],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout
            )

        # Combine stdout and stderr for complete output
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr

        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        return False, "pm4_decoder timed out after 10 seconds"
    except Exception as e:
        return False, f"Error running pm4_decoder: {str(e)}"


def filter_rebase_lines(output: str) -> str:
    """
    Filter out rebase-related lines that should not affect comparison.
    These lines are still printed but not used for comparison.

    Args:
        output: Decoder output string

    Returns:
        Filtered output string
    """
    lines = output.split('\n')
    filtered = []
    for line in lines:
        # Skip rebase information lines
        if line.startswith('Rebasing to new base address:'):
            continue
        if line.startswith('Original base address:'):
            continue
        if 'COPY_DATA:' in line and '->' in line and '(offset:' in line:
            # Skip lines like "COPY_DATA: 0x... -> 0x... (offset: 0x...)"
            continue
        if 'ACQUIRE_MEM:' in line and '->' in line and '(offset:' in line:
            # Skip lines like "ACQUIRE_MEM: 0x... -> 0x... (offset: 0x...)"
            continue
        filtered.append(line)
    return '\n'.join(filtered)


def compare_decoder_outputs(original_output: str, new_output: str) -> Tuple[bool, str]:
    """
    Compare two decoder outputs and generate a diff report.
    Filters out rebase information lines before comparison.

    Args:
        original_output: Output from decoding original packets
        new_output: Output from decoding new packets

    Returns:
        Tuple of (are_equal: bool, diff_report: str)
    """
    # Filter out rebase lines for comparison
    orig_filtered = filter_rebase_lines(original_output)
    new_filtered = filter_rebase_lines(new_output)

    # Simple string comparison on filtered output
    if orig_filtered.strip() == new_filtered.strip():
        return True, ""

    # Generate side-by-side comparison using filtered output
    orig_lines = orig_filtered.strip().split('\n')
    new_lines = new_filtered.strip().split('\n')

    diff_report = []
    diff_report.append(f"\n{'=' * 80}")
    diff_report.append(f"{'ORIGINAL':<40} | {'NEW':<40}")
    diff_report.append(f"{'-' * 40}-+-{'-' * 40}")

    max_lines = max(len(orig_lines), len(new_lines))
    for i in range(max_lines):
        orig_line = orig_lines[i] if i < len(orig_lines) else ""
        new_line = new_lines[i] if i < len(new_lines) else ""

        # Truncate lines if too long for side-by-side display
        orig_display = (orig_line[:38] + '..') if len(orig_line) > 40 else orig_line
        new_display = (new_line[:38] + '..') if len(new_line) > 40 else new_line

        # Highlight differences
        if orig_line != new_line:
            marker = " *"
        else:
            marker = "  "

        diff_report.append(f"{orig_display:<40} | {new_display:<40}{marker}")

    diff_report.append(f"{'=' * 80}\n")

    return False, '\n'.join(diff_report)


def compare_packet_type(packet_type: str, original_packets: List[Dict],
                       new_packets: List[Dict], decoder_path: Path,
                       temp_dir: Path) -> Dict:
    """
    Compare a specific packet type (start, stop, or read) between original and new.

    Args:
        packet_type: Type of packet ("START", "STOP", or "READ")
        original_packets: List of original packet dictionaries
        new_packets: List of new packet dictionaries
        decoder_path: Path to pm4_decoder executable
        temp_dir: Temporary directory for hex files

    Returns:
        Dictionary with comparison results
    """
    result = {
        'type': packet_type,
        'match': False,
        'error': None,
        'original_output': '',
        'new_output': '',
        'diff': ''
    }

    # Create temporary hex files
    orig_hex = temp_dir / f"original_{packet_type.lower()}.hex"
    new_hex = temp_dir / f"new_{packet_type.lower()}.hex"

    try:
        # Write packets to hex files
        write_packets_to_hex_file(original_packets, orig_hex)
        write_packets_to_hex_file(new_packets, new_hex)

        # Decode original packets
        orig_success, orig_output = decode_packets(orig_hex, decoder_path)
        if not orig_success:
            result['error'] = f"Failed to decode original packets: {orig_output}"
            return result

        # Decode new packets
        new_success, new_output = decode_packets(new_hex, decoder_path)
        if not new_success:
            result['error'] = f"Failed to decode new packets: {new_output}"
            return result

        result['original_output'] = orig_output
        result['new_output'] = new_output

        # Compare outputs
        are_equal, diff_report = compare_decoder_outputs(orig_output, new_output)
        result['match'] = are_equal
        result['diff'] = diff_report

    except Exception as e:
        result['error'] = f"Error during comparison: {str(e)}"

    return result


def print_comparison_results(block_name: str, event_id: str,
                            results: List[Dict]) -> None:
    """
    Print comparison results in a clear, readable format.

    Args:
        block_name: Name of the block being compared
        event_id: Event ID being compared
        results: List of comparison result dictionaries
    """
    print(f"\n{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}PM4 Packet Comparison{Colors.ENDC}")
    print(f"{Colors.BOLD}Block: {Colors.OKCYAN}{block_name}{Colors.ENDC}")
    print(f"{Colors.BOLD}Event: {Colors.OKCYAN}{event_id}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

    for result in results:
        packet_type = result['type']
        print(f"{Colors.BOLD}{Colors.OKBLUE}{packet_type} PACKETS:{Colors.ENDC}")
        print(f"{'-' * 80}")

        if result['error']:
            print(f"{Colors.FAIL}ERROR: {result['error']}{Colors.ENDC}\n")
            continue

        if result['match']:
            print(f"{Colors.OKGREEN}✓ Packets match{Colors.ENDC}\n")
        else:
            print(f"{Colors.WARNING}✗ Packets differ{Colors.ENDC}")
            print(result['diff'])

        print()


def main():
    """Main entry point for the packet comparison tool"""
    parser = argparse.ArgumentParser(
        description='Compare PM4 packets between original and new JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s GL2C 41        # Compare GL2C block, event 41 (decimal)
  %(prog)s SQ 0x42        # Compare SQ block, event 0x42 (hex)
  %(prog)s TA 100         # Compare TA block, event 100 (decimal)
        """
    )

    parser.add_argument('block_name',
                       help='Block name (e.g., GL2C, GRBM, SQ, TA)')
    parser.add_argument('event_id',
                       help='Event ID in decimal (e.g., 41) or hex (e.g., 0x29)')
    parser.add_argument('--decoder',
                       default='./pm4_decoder',
                       help='Path to pm4_decoder executable (default: ./pm4_decoder)')
    parser.add_argument('--original',
                       default='original_packets.json',
                       help='Path to original packets JSON (default: original_packets.json)')
    parser.add_argument('--new',
                       default='new_packets.json',
                       help='Path to new packets JSON (default: new_packets.json)')

    args = parser.parse_args()

    try:
        # Parse event ID
        event_id = parse_event_id(args.event_id)

        # Get script directory for relative paths
        script_dir = Path(__file__).parent.absolute()

        # Resolve paths
        decoder_path = Path(args.decoder)
        if not decoder_path.is_absolute():
            decoder_path = script_dir / decoder_path

        original_json_path = Path(args.original)
        if not original_json_path.is_absolute():
            original_json_path = script_dir / original_json_path

        new_json_path = Path(args.new)
        if not new_json_path.is_absolute():
            new_json_path = script_dir / new_json_path

        # Load JSON files
        print(f"Loading {original_json_path}...")
        original_data = load_json_file(original_json_path)

        print(f"Loading {new_json_path}...")
        new_data = load_json_file(new_json_path)

        # Get block/event data from both files
        original_block = get_block_data(original_data, args.block_name, event_id)
        new_block = get_block_data(new_data, args.block_name, event_id)

        # Check if block/event exists in both files
        if original_block is None:
            print(f"{Colors.FAIL}ERROR: Block '{args.block_name}' with event '{event_id}' "
                  f"not found in {original_json_path}{Colors.ENDC}")
            sys.exit(1)

        if new_block is None:
            print(f"{Colors.FAIL}ERROR: Block '{args.block_name}' with event '{event_id}' "
                  f"not found in {new_json_path}{Colors.ENDC}")
            sys.exit(1)

        # Create temporary directory for hex files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Compare each packet type
            results = []

            for packet_type in ['START', 'STOP', 'READ']:
                packet_key = packet_type.lower() + '_packets'

                print(f"Comparing {packet_type.lower()} packets...")
                result = compare_packet_type(
                    packet_type,
                    original_block[packet_key],
                    new_block[packet_key],
                    decoder_path,
                    temp_path
                )
                results.append(result)

            # Print results
            print_comparison_results(args.block_name, event_id, results)

            # Exit with appropriate code
            all_match = all(r['match'] for r in results if r['error'] is None)
            has_errors = any(r['error'] is not None for r in results)

            if has_errors:
                sys.exit(2)
            elif not all_match:
                sys.exit(1)
            else:
                sys.exit(0)

    except FileNotFoundError as e:
        print(f"{Colors.FAIL}ERROR: {str(e)}{Colors.ENDC}")
        sys.exit(1)
    except ValueError as e:
        print(f"{Colors.FAIL}ERROR: {str(e)}{Colors.ENDC}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{Colors.FAIL}ERROR: {str(e)}{Colors.ENDC}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print(f"{Colors.FAIL}UNEXPECTED ERROR: {str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
