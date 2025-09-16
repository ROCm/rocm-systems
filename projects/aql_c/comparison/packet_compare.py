#!/usr/bin/env python3
"""
AQL Packet Comparison Tool

This tool compares outputs from aql_c_dumper and aqlprofile_v2_dumper
to identify differences in generated AQL packets.
"""

import argparse
import json
import sys
import os
import difflib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import struct


class PacketComparer:
    """Main class for comparing AQL packets between implementations."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.differences = []

    def load_json_file(self, filepath: str) -> Dict[str, Any]:
        """Load and parse JSON packet dump file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            sys.exit(1)

    def load_binary_file(self, filepath: str) -> Dict[str, Any]:
        """Load and parse binary packet dump file."""
        try:
            with open(filepath, 'rb') as f:
                data = f.read()

            # Parse binary format
            offset = 0
            magic = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4

            if magic == 0x41514C43:  # "AQLC"
                tool = "aql_c_dumper"
            elif magic == 0x41514C56:  # "AQLV"
                tool = "aqlprofile_v2_dumper"
            else:
                raise ValueError(f"Invalid magic number: 0x{magic:08x}")

            version = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4

            arch_len = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4

            arch_name = data[offset:offset+arch_len].decode('utf-8')
            offset += arch_len

            event_count = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4

            # Parse events (simplified - actual structure depends on implementation)
            events = []
            for i in range(event_count):
                # This would need to match the actual event structure
                offset += 32  # Skip event data for now

            return {
                'tool': tool,
                'architecture': arch_name,
                'event_count': event_count,
                'events': events,
                'binary_data': data,
                'binary_size': len(data)
            }

        except Exception as e:
            print(f"Error loading binary file {filepath}: {e}")
            sys.exit(1)

    def compare_basic_info(self, data1: Dict, data2: Dict) -> bool:
        """Compare basic packet information."""
        differences = []

        if data1.get('architecture') != data2.get('architecture'):
            differences.append(f"Architecture mismatch: {data1.get('architecture')} vs {data2.get('architecture')}")

        if data1.get('event_count') != data2.get('event_count'):
            differences.append(f"Event count mismatch: {data1.get('event_count')} vs {data2.get('event_count')}")

        self.differences.extend(differences)
        return len(differences) == 0

    def compare_events(self, events1: List[Dict], events2: List[Dict]) -> bool:
        """Compare event configurations."""
        differences = []

        if len(events1) != len(events2):
            differences.append(f"Event list length mismatch: {len(events1)} vs {len(events2)}")
            return False

        for i, (event1, event2) in enumerate(zip(events1, events2)):
            if event1.get('block_name') != event2.get('block_name'):
                differences.append(f"Event {i} block name mismatch: {event1.get('block_name')} vs {event2.get('block_name')}")

            if event1.get('block_instance') != event2.get('block_instance'):
                differences.append(f"Event {i} instance mismatch: {event1.get('block_instance')} vs {event2.get('block_instance')}")

            if event1.get('event_id') != event2.get('event_id'):
                differences.append(f"Event {i} ID mismatch: {event1.get('event_id')} vs {event2.get('event_id')}")

            if event1.get('flags') != event2.get('flags'):
                differences.append(f"Event {i} flags mismatch: {event1.get('flags')} vs {event2.get('flags')}")

        self.differences.extend(differences)
        return len(differences) == 0

    def compare_packet_field(self, packet1: Dict, packet2: Dict, field: str, packet_name: str) -> bool:
        """Compare a specific field in packet structures."""
        val1 = packet1.get(field)
        val2 = packet2.get(field)

        if val1 != val2:
            self.differences.append(f"{packet_name} {field} mismatch: {val1} vs {val2}")
            return False
        return True

    def compare_packets(self, data1: Dict, data2: Dict) -> bool:
        """Compare packet structures."""
        all_match = True

        # Compare start packets
        start1 = data1.get('start_packet', {})
        start2 = data2.get('start_packet', {})

        packet_fields = ['header', 'size']

        # AQL_C uses pm4_ib_command structure
        if 'pm4_ib_command' in start1:
            all_match &= self.compare_packet_field(start1, start2, 'pm4_ib_format', 'start_packet')
            all_match &= self.compare_packet_field(start1, start2, 'dw_count_remain', 'start_packet')
            all_match &= self.compare_packet_field(start1, start2, 'completion_signal', 'start_packet')

            # Compare PM4 command arrays
            cmd1 = start1.get('pm4_ib_command', [])
            cmd2 = start2.get('pm4_ib_command', [])
            if cmd1 != cmd2:
                self.differences.append(f"start_packet PM4 command mismatch")
                all_match = False

        # AQLProfile v2 uses different structure
        elif 'type' in start1:
            v2_fields = ['type', 'source', 'format', 'control', 'address', 'size_field', 'connection']
            for field in v2_fields:
                all_match &= self.compare_packet_field(start1, start2, field, 'start_packet')

        # Compare common fields
        for field in packet_fields:
            all_match &= self.compare_packet_field(start1, start2, field, 'start_packet')

        # Similar comparison for stop and read packets
        # (Simplified for brevity - would include full comparison)

        return all_match

    def analyze_semantic_differences(self, data1: Dict, data2: Dict) -> List[str]:
        """Analyze semantic differences between implementations."""
        analysis = []

        # Check if packet structures are fundamentally different
        start1 = data1.get('start_packet', {})
        start2 = data2.get('start_packet', {})

        if 'pm4_ib_command' in start1 and 'type' in start2:
            analysis.append("Different packet structures detected:")
            analysis.append("  - AQL_C uses PM4 Indirect Buffer format")
            analysis.append("  - AQLProfile v2 uses HSA AQL PM4 format")
            analysis.append("  - This is expected due to different implementations")

        # Check for memory buffer differences
        cmd_buf1 = data1.get('command_buffer', {})
        cmd_buf2 = data2.get('command_buffer', {})

        if cmd_buf1.get('present') != cmd_buf2.get('present'):
            analysis.append("Command buffer handling differs between implementations")

        return analysis

    def generate_diff_report(self, file1: str, file2: str, data1: Dict, data2: Dict) -> str:
        """Generate a detailed difference report."""
        report = []
        report.append("=" * 80)
        report.append("AQL PACKET COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"File 1: {file1} ({data1.get('tool', 'unknown')})")
        report.append(f"File 2: {file2} ({data2.get('tool', 'unknown')})")
        report.append(f"Timestamp: {data1.get('timestamp', 'unknown')} vs {data2.get('timestamp', 'unknown')}")
        report.append("")

        # Basic comparison
        report.append("BASIC INFORMATION:")
        self.compare_basic_info(data1, data2)

        # Event comparison
        report.append("EVENT CONFIGURATION:")
        self.compare_events(data1.get('events', []), data2.get('events', []))

        # Packet comparison
        report.append("PACKET COMPARISON:")
        self.compare_packets(data1, data2)

        # Report differences
        if self.differences:
            report.append("DIFFERENCES FOUND:")
            for diff in self.differences:
                report.append(f"  - {diff}")
        else:
            report.append("NO DIFFERENCES FOUND")

        # Semantic analysis
        semantic_analysis = self.analyze_semantic_differences(data1, data2)
        if semantic_analysis:
            report.append("")
            report.append("SEMANTIC ANALYSIS:")
            report.extend(semantic_analysis)

        return "\n".join(report)

    def compare_files(self, file1: str, file2: str) -> bool:
        """Compare two packet dump files."""
        self.differences = []  # Reset differences

        # Determine file types
        is_json1 = file1.endswith('.json')
        is_json2 = file2.endswith('.json')

        # Load files
        if is_json1:
            data1 = self.load_json_file(file1)
        else:
            data1 = self.load_binary_file(file1)

        if is_json2:
            data2 = self.load_json_file(file2)
        else:
            data2 = self.load_binary_file(file2)

        # Generate comparison report
        report = self.generate_diff_report(file1, file2, data1, data2)
        print(report)

        return len(self.differences) == 0

    def batch_compare(self, dir1: str, dir2: str) -> Dict[str, bool]:
        """Compare all matching files in two directories."""
        results = {}

        dir1_path = Path(dir1)
        dir2_path = Path(dir2)

        if not dir1_path.exists() or not dir2_path.exists():
            print("One or both directories don't exist")
            return results

        # Find matching files
        files1 = set(f.name for f in dir1_path.iterdir() if f.is_file())
        files2 = set(f.name for f in dir2_path.iterdir() if f.is_file())

        common_files = files1.intersection(files2)

        if not common_files:
            print("No matching files found")
            return results

        print(f"Comparing {len(common_files)} file pairs...")

        for filename in sorted(common_files):
            file1 = dir1_path / filename
            file2 = dir2_path / filename

            print(f"\n{'='*60}")
            print(f"Comparing: {filename}")
            print('='*60)

            match = self.compare_files(str(file1), str(file2))
            results[filename] = match

        # Summary
        print(f"\n{'='*60}")
        print("BATCH COMPARISON SUMMARY")
        print('='*60)

        matches = sum(1 for result in results.values() if result)
        total = len(results)

        print(f"Total files compared: {total}")
        print(f"Perfect matches: {matches}")
        print(f"Differences found: {total - matches}")

        if total - matches > 0:
            print("\nFiles with differences:")
            for filename, match in results.items():
                if not match:
                    print(f"  - {filename}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Compare AQL packet dumps")
    parser.add_argument('file1', nargs='?', help='First file to compare')
    parser.add_argument('file2', nargs='?', help='Second file to compare')
    parser.add_argument('--batch', nargs=2, metavar=('DIR1', 'DIR2'),
                       help='Batch compare all files in two directories')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed differences')

    args = parser.parse_args()

    comparer = PacketComparer(verbose=args.verbose)

    if args.batch:
        results = comparer.batch_compare(args.batch[0], args.batch[1])
        # Exit with error code if any differences found
        sys.exit(0 if all(results.values()) else 1)
    elif args.file1 and args.file2:
        match = comparer.compare_files(args.file1, args.file2)
        sys.exit(0 if match else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()