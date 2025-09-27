#!/usr/bin/env python3
"""
PCIe Trace Log Analyzer

Analyzes PCIe trace log files for memory read/write transactions,
ordering attributes, and timing patterns.

Usage: python3 pcie_trace_analyzer.py [directory_path]
"""

import os
import re
import sys
import glob
from typing import Dict, List, Tuple, Optional

class PCIeTraceAnalyzer:
    def __init__(self, directory_path: str = "."):
        self.directory_path = directory_path
        self.results = {}

    def find_log_files(self) -> List[str]:
        """Find all .log and .txt files in the directory."""
        patterns = ["*.log", "*.txt"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(self.directory_path, pattern)))
        return sorted(files)

    def extract_memory_transactions(self, filepath: str) -> Tuple[List[int], Dict[str, int]]:
        """Extract memory transaction timestamps and ordering attributes."""
        timestamps = []
        ordering_counts = {'Attr=0': 0, 'Attr=1': 0, 'Attr=2': 0, 'Attr=3': 0}
        read_count = 0
        write_count = 0

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                # Look for timestamp lines followed by memory transactions
                if re.match(r'^[0-9A-F]{16}', line) and i+1 < len(lines):
                    next_line = lines[i+1]

                    # Check if next line contains memory read/write transactions
                    if any(x in next_line for x in ['MR64', 'MR32', 'MW64', 'MW32']):
                        # Extract timestamp
                        ts_str = line.split()[0].rstrip('*')
                        try:
                            timestamps.append(int(ts_str, 16))
                        except ValueError:
                            continue

                        # Count reads vs writes
                        if any(x in next_line for x in ['MR64', 'MR32']):
                            read_count += 1
                        elif any(x in next_line for x in ['MW64', 'MW32']):
                            write_count += 1

                        # Extract ordering attributes
                        attr_match = re.search(r'Attr=([0-9])', next_line)
                        if attr_match:
                            attr_key = f'Attr={attr_match.group(1)}'
                            if attr_key in ordering_counts:
                                ordering_counts[attr_key] += 1

            # Add read/write counts to ordering_counts for convenience
            ordering_counts['reads'] = read_count
            ordering_counts['writes'] = write_count

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

        return timestamps, ordering_counts

    def calculate_timing_stats(self, timestamps: List[int]) -> Dict[str, float]:
        """Calculate timing statistics from timestamps."""
        if len(timestamps) < 2:
            return {}

        duration = timestamps[-1] - timestamps[0]
        rate_per_1k = (len(timestamps) / duration * 1000) if duration > 0 else 0
        avg_interval = duration / len(timestamps) if len(timestamps) > 0 else 0

        # Calculate gaps between consecutive transactions
        gaps = [timestamps[i] - timestamps[i-1] for i in range(1, min(20, len(timestamps)))]

        return {
            'duration': duration,
            'rate_per_1k': rate_per_1k,
            'avg_interval': avg_interval,
            'min_gap': min(gaps) if gaps else 0,
            'max_gap': max(gaps) if gaps else 0,
            'avg_gap': sum(gaps) / len(gaps) if gaps else 0
        }

    def analyze_file(self, filepath: str) -> Dict:
        """Analyze a single log file."""
        filename = os.path.basename(filepath)
        timestamps, ordering_counts = self.extract_memory_transactions(filepath)

        total_transactions = sum(ordering_counts[key] for key in ['Attr=0', 'Attr=1', 'Attr=2', 'Attr=3'])

        if total_transactions == 0:
            return {
                'filename': filename,
                'total_transactions': 0,
                'reads': 0,
                'writes': 0,
                'ordering': ordering_counts,
                'timing': {}
            }

        timing_stats = self.calculate_timing_stats(timestamps)

        return {
            'filename': filename,
            'total_transactions': total_transactions,
            'reads': ordering_counts['reads'],
            'writes': ordering_counts['writes'],
            'ordering': ordering_counts,
            'timing': timing_stats
        }

    def analyze_all_files(self):
        """Analyze all log files in the directory."""
        files = self.find_log_files()

        if not files:
            print(f"No .log or .txt files found in {self.directory_path}")
            return

        print(f"Found {len(files)} log files to analyze...")

        for filepath in files:
            result = self.analyze_file(filepath)
            self.results[result['filename']] = result

    def print_transaction_summary(self):
        """Print transaction count and read/write breakdown table."""
        print("\n" + "="*80)
        print("TRANSACTION SUMMARY")
        print("="*80)

        print(f"{'File':<20} {'Total':<8} {'Reads':<8} {'Writes':<8} {'R/W Ratio':<12}")
        print("-" * 80)

        for filename, data in sorted(self.results.items()):
            if data['total_transactions'] > 0:
                rw_ratio = f"{data['reads']}/{data['writes']}"
                print(f"{filename:<20} {data['total_transactions']:<8} {data['reads']:<8} {data['writes']:<8} {rw_ratio:<12}")

    def print_ordering_analysis(self):
        """Print ordering attributes analysis table."""
        print("\n" + "="*100)
        print("ORDERING ATTRIBUTES ANALYSIS")
        print("="*100)

        print(f"{'File':<20} {'Total':<8} {'Attr=0':<8} {'Attr=1':<8} {'Attr=2':<8} {'Attr=3':<8} {'Strict%':<10} {'Relaxed%':<10}")
        print("-" * 100)

        for filename, data in sorted(self.results.items()):
            if data['total_transactions'] > 0:
                ordering = data['ordering']
                total = data['total_transactions']
                strict = ordering['Attr=0'] + ordering['Attr=1']
                relaxed = ordering['Attr=2'] + ordering['Attr=3']
                strict_pct = (strict / total * 100) if total > 0 else 0
                relaxed_pct = (relaxed / total * 100) if total > 0 else 0

                print(f"{filename:<20} {total:<8} {ordering['Attr=0']:<8} {ordering['Attr=1']:<8} "
                      f"{ordering['Attr=2']:<8} {ordering['Attr=3']:<8} {strict_pct:<10.1f} {relaxed_pct:<10.1f}")

    def print_timing_analysis(self):
        """Print timing analysis table."""
        print("\n" + "="*120)
        print("TIMING ANALYSIS")
        print("="*120)

        print(f"{'File':<20} {'Transactions':<12} {'Duration':<12} {'Rate/1K':<10} {'Avg Interval':<12} {'Pattern':<15}")
        print("-" * 120)

        for filename, data in sorted(self.results.items()):
            if data['total_transactions'] > 0 and data['timing']:
                timing = data['timing']
                duration_k = timing['duration'] / 1000

                # Classify pattern based on rate
                if timing['rate_per_1k'] > 15:
                    pattern = "High-freq burst"
                elif timing['rate_per_1k'] > 5:
                    pattern = "Medium-freq"
                else:
                    pattern = "Low-freq extended"

                print(f"{filename:<20} {data['total_transactions']:<12} {duration_k:<12.1f}K "
                      f"{timing['rate_per_1k']:<10.2f} {timing['avg_interval']:<12.1f} {pattern:<15}")

    def print_strict_ordering_ranking(self):
        """Print strict ordering operations ranking."""
        print("\n" + "="*80)
        print("STRICT ORDERING OPERATIONS RANKING")
        print("="*80)

        # Calculate strict ordering counts and sort
        strict_data = []
        for filename, data in self.results.items():
            if data['total_transactions'] > 0:
                ordering = data['ordering']
                strict_count = ordering['Attr=0'] + ordering['Attr=1']
                strict_pct = (strict_count / data['total_transactions'] * 100)
                strict_data.append((filename, strict_count, data['total_transactions'], strict_pct))

        strict_data.sort(key=lambda x: x[1], reverse=True)

        print(f"{'Rank':<6} {'File':<20} {'Strict Ops':<12} {'Total':<8} {'Percentage':<12}")
        print("-" * 80)

        for i, (filename, strict_count, total, strict_pct) in enumerate(strict_data, 1):
            print(f"{i:<6} {filename:<20} {strict_count:<12} {total:<8} {strict_pct:<12.1f}%")

    def print_summary_statistics(self):
        """Print overall summary statistics."""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)

        total_transactions = sum(data['total_transactions'] for data in self.results.values())
        total_reads = sum(data['reads'] for data in self.results.values())
        total_writes = sum(data['writes'] for data in self.results.values())

        total_strict = sum(data['ordering']['Attr=0'] + data['ordering']['Attr=1']
                          for data in self.results.values() if data['total_transactions'] > 0)
        total_relaxed = sum(data['ordering']['Attr=2'] + data['ordering']['Attr=3']
                           for data in self.results.values() if data['total_transactions'] > 0)

        print(f"Files analyzed: {len([d for d in self.results.values() if d['total_transactions'] > 0])}")
        print(f"Total transactions: {total_transactions:,}")
        print(f"Total reads: {total_reads:,} ({total_reads/total_transactions*100:.1f}%)")
        print(f"Total writes: {total_writes:,} ({total_writes/total_transactions*100:.1f}%)")
        print(f"Total strict ordering: {total_strict:,} ({total_strict/total_transactions*100:.1f}%)")
        print(f"Total relaxed ordering: {total_relaxed:,} ({total_relaxed/total_transactions*100:.1f}%)")

    def generate_report(self):
        """Generate complete analysis report."""
        self.analyze_all_files()

        if not any(data['total_transactions'] > 0 for data in self.results.values()):
            print("No PCIe memory transactions found in any files.")
            return

        print("PCIe TRACE LOG ANALYSIS REPORT")
        print("=" * 80)
        print(f"Directory: {os.path.abspath(self.directory_path)}")

        self.print_transaction_summary()
        self.print_ordering_analysis()
        self.print_timing_analysis()
        self.print_strict_ordering_ranking()
        self.print_summary_statistics()

        print("\n" + "="*80)
        print("LEGEND:")
        print("Attr=0: Strict ordering + cache coherent")
        print("Attr=1: Strict ordering + no snoop")
        print("Attr=2: Relaxed ordering + cache coherent")
        print("Attr=3: Relaxed ordering + no snoop")
        print("Rate/1K: Transactions per 1000 cycles")
        print("="*80)


def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)

    analyzer = PCIeTraceAnalyzer(directory)
    analyzer.generate_report()


if __name__ == "__main__":
    main()