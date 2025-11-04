#!/usr/bin/env python3
# validate_stream_correlation.py

import sys
import os
from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig

def load_trace(trace_file, bin_path=None):
    """Load trace using the correct Perfetto API"""
    try:
        if bin_path and os.path.isfile(bin_path):
            config = TraceProcessorConfig(bin_path=bin_path)
            tp = TraceProcessor(trace=trace_file, config=config)
        else:
            tp = TraceProcessor(trace=trace_file)
        return tp
    except Exception as ex:
        print(f"Error loading trace: {ex}")
        return None

def validate_stream_correlation(trace_file):
    """Validate that kernels appear under correct stream tracks"""
    print(f"Loading trace file: {trace_file}")
    
    tp = load_trace(trace_file)
    if tp is None:
        return ["Failed to load trace"], []
    
    # Query for all kernel events and their parent tracks
    try:
        kernel_events = tp.query("""
            SELECT 
                slice.name as kernel_name,
                track.name as track_name,
                slice.ts,
                slice.dur
            FROM slice 
            JOIN track ON slice.track_id = track.id 
            WHERE slice.name LIKE '%kernel%' 
               OR slice.name LIKE '%simple_kernel%'
            ORDER BY slice.ts
        """)
    except Exception as ex:
        print(f"Query failed: {ex}")
        return [f"Query error: {ex}"], []
    
    stream_violations = []
    queue_violations = []
    track_summary = {}
    
    print("\n📊 Kernel-to-Track Analysis:")
    for row in kernel_events:
        kernel_name = row.kernel_name
        track_name = row.track_name
        
        print(f"  Kernel: {kernel_name} -> Track: {track_name}")
        
        # Count kernels per track
        if track_name not in track_summary:
            track_summary[track_name] = 0
        track_summary[track_name] += 1
        
        # Check for incorrect grouping patterns (this would indicate the bug)
        if "Stream" in track_name and "Queue" in track_name:
            stream_violations.append(f"Mixed track naming: {track_name}")
        
        # Validate stream assignments based on your test
        if "simple_kernel" in kernel_name:
            # Check if kernel is on expected tracks
            expected_tracks = ["HIP Activity Stream 0", "HIP Activity Stream 1", 
                             "HIP Activity Stream 2"]
            queue_tracks = [t for t in [track_name] if "Queue" in t]
            
            if track_name not in expected_tracks and not queue_tracks:
                queue_violations.append(f"Unexpected track for {kernel_name}: {track_name}")
    
    # Print track summary
    print("\n📈 Track Summary:")
    for track, count in track_summary.items():
        print(f"  {track}: {count} kernels")
    
    # Additional validation: Check for proper stream distribution
    stream_tracks = [t for t in track_summary.keys() if "HIP Activity Stream" in t]
    queue_tracks = [t for t in track_summary.keys() if "Queue" in t]
    
    print(f"\n🔍 Found {len(stream_tracks)} stream tracks, {len(queue_tracks)} queue tracks")
    
    # Query for stream_id annotations to verify correlation
    try:
        stream_annotations = tp.query("""
            SELECT 
                slice.name as kernel_name, 
                args.key, 
                args.string_value,
                track.name as track_name
            FROM slice 
            JOIN track ON slice.track_id = track.id 
            LEFT JOIN args ON slice.arg_set_id = args.arg_set_id 
            WHERE args.key = 'stream_id' OR args.key LIKE '%stream%'
            ORDER BY slice.ts
        """)
        
        print("\n🏷️  Stream ID Annotations:")
        for row in stream_annotations:
            print(f"  {row.kernel_name} ({row.track_name}): {row.key} = {row.string_value}")
            
    except Exception as ex:
        print(f"Stream annotation query failed: {ex}")
    
    return stream_violations, queue_violations

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_stream_correlation.py <trace.proto>")
        print("Example: python validate_stream_correlation.py test1/perfetto-trace.proto")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    
    if not os.path.exists(trace_file):
        print(f"❌ Trace file not found: {trace_file}")
        sys.exit(1)
    
    print(f"🔍 Validating stream correlation in: {trace_file}")
    violations1, violations2 = validate_stream_correlation(trace_file)
    
    if violations1 or violations2:
        print("\n❌ VALIDATION FAILED:")
        for v in violations1 + violations2:
            print(f"  • {v}")
        sys.exit(1)
    else:
        print("\n✅ Stream correlation validation PASSED")
        print("   All kernels appear under correct tracks!")
        sys.exit(0)

if __name__ == "__main__":
    main()