#!/usr/bin/env python3
# validate_stream_correlation_simple.py
# Alternative validation using trace_processor_shell command line tool

import sys
import os
import subprocess
import json
import tempfile

def validate_with_shell(trace_file):
    """Use trace_processor_shell command line tool for validation"""
    
    # Create SQL query file
    query = """
    SELECT 
        slice.name as kernel_name,
        track.name as track_name,
        slice.ts,
        slice.dur
    FROM slice 
    JOIN track ON slice.track_id = track.id 
    WHERE slice.name LIKE '%kernel%' 
       OR slice.name LIKE '%simple_kernel%'
    ORDER BY slice.ts;
    """
    
    try:
        # Write query to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write(query)
            query_file = f.name
        
        # Run trace_processor_shell
        cmd = ['trace_processor_shell', '--query-file', query_file, trace_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Clean up
        os.unlink(query_file)
        
        if result.returncode != 0:
            print(f"❌ trace_processor_shell failed: {result.stderr}")
            return False
        
        # Parse output
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:  # Header + at least one data row
            print("❌ No kernel events found in trace")
            return False
        
        print(f"\n📊 Found {len(lines)-1} kernel events:")
        track_summary = {}
        
        # Skip header line
        for line in lines[1:]:
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 2:
                    kernel_name = parts[0]
                    track_name = parts[1]
                    
                    print(f"  {kernel_name} -> {track_name}")
                    
                    # Count kernels per track
                    if track_name not in track_summary:
                        track_summary[track_name] = 0
                    track_summary[track_name] += 1
        
        print(f"\n📈 Track Summary:")
        for track, count in track_summary.items():
            print(f"  {track}: {count} kernels")
        
        # Validate tracks
        stream_tracks = [t for t in track_summary.keys() if "HIP Activity Stream" in t]
        queue_tracks = [t for t in track_summary.keys() if "Queue" in t]
        
        print(f"\n🔍 Found {len(stream_tracks)} stream tracks, {len(queue_tracks)} queue tracks")
        
        if len(stream_tracks) == 0 and len(queue_tracks) == 0:
            print("❌ No HIP stream or queue tracks found - possible tracing issue")
            return False
        
        print("✅ Basic validation passed")
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ trace_processor_shell timed out")
        return False
    except FileNotFoundError:
        print("❌ trace_processor_shell not found in PATH")
        print("   Try: export PATH=$PATH:/path/to/perfetto/tools")
        return False
    except Exception as ex:
        print(f"❌ Validation error: {ex}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_stream_correlation_simple.py <trace.proto>")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    
    if not os.path.exists(trace_file):
        print(f"❌ Trace file not found: {trace_file}")
        sys.exit(1)
    
    print(f"🔍 Validating stream correlation in: {trace_file}")
    
    if validate_with_shell(trace_file):
        print("\n✅ Stream correlation validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ Stream correlation validation FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()