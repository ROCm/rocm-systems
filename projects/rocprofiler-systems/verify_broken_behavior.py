#!/usr/bin/env python3
# verify_broken_behavior.py - Detect the stream correlation bug

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

def detect_stream_correlation_bug(trace_file):
    """Detect the specific _group_by_queue variable contamination bug"""
    print(f"🔍 Analyzing trace for stream correlation bug: {trace_file}")
    
    tp = load_trace(trace_file)
    if tp is None:
        return False
    
    # Query for kernel events in chronological order
    try:
        kernel_events = tp.query("""
            SELECT 
                slice.name as kernel_name,
                track.name as track_name,
                slice.ts as timestamp,
                slice.dur,
                row_number() OVER (ORDER BY slice.ts) as event_order
            FROM slice 
            JOIN track ON slice.track_id = track.id 
            WHERE slice.name LIKE '%kernel%' 
               OR slice.name LIKE '%simple_kernel%'
            ORDER BY slice.ts
        """)
    except Exception as ex:
        print(f"Query failed: {ex}")
        return False
    
    print(f"\n📊 Found {len(kernel_events)} kernel events:")
    
    bug_detected = False
    stream_events = []
    queue_events = []
    
    for row in kernel_events:
        event_info = {
            'order': row.event_order,
            'name': row.kernel_name,
            'track': row.track_name,
            'timestamp': row.timestamp
        }
        
        print(f"  {row.event_order}: {row.kernel_name} -> {row.track_name}")
        
        if "HIP Activity Stream" in row.track_name:
            stream_events.append(event_info)
        elif "Queue" in row.track_name:
            queue_events.append(event_info)
    
    print(f"\n🎯 Track Classification:")
    print(f"  Stream tracks: {len(stream_events)} events")
    print(f"  Queue tracks: {len(queue_events)} events")
    
    # BUG DETECTION LOGIC:
    # The bug manifests when:
    # 1. Early kernels appear on stream tracks (correct)
    # 2. A default stream (stream_id=0) kernel appears 
    # 3. ALL subsequent kernels switch to queue tracks (incorrect)
    
    if len(stream_events) > 0 and len(queue_events) > 0:
        # Check if there's a suspicious pattern where queue events 
        # come after stream events (indicating variable contamination)
        first_queue_order = min([e['order'] for e in queue_events])
        last_stream_order = max([e['order'] for e in stream_events])
        
        if first_queue_order < last_stream_order:
            print(f"\n✅ Mixed stream/queue ordering detected - this suggests correct behavior")
        else:
            print(f"\n🚨 BUG DETECTED: All queue events ({first_queue_order}+) come after stream events (1-{last_stream_order})")
            print(f"   This indicates _group_by_queue variable contamination!")
            bug_detected = True
            
            # Show the problematic transition
            print(f"\n📈 Problematic sequence:")
            all_events = stream_events + queue_events
            all_events.sort(key=lambda x: x['order'])
            
            transition_found = False
            for i in range(len(all_events) - 1):
                current = all_events[i]
                next_event = all_events[i + 1]
                
                if "Stream" in current['track'] and "Queue" in next_event['track']:
                    print(f"   {current['order']}: {current['name']} -> {current['track']} ✅")
                    print(f"   {next_event['order']}: {next_event['name']} -> {next_event['track']} ❌ (should be Stream!)")
                    transition_found = True
                    break
            
            if transition_found:
                print(f"\n💡 Expected behavior: Each kernel should appear under its correct stream track")
                print(f"   Actual behavior: _group_by_queue=true leaked to subsequent records")
    
    # Additional check: Look for stream_id annotations vs track placement
    try:
        stream_annotations = tp.query("""
            SELECT 
                slice.name as kernel_name,
                track.name as track_name,
                args.string_value as stream_id,
                slice.ts
            FROM slice 
            JOIN track ON slice.track_id = track.id 
            JOIN args ON slice.arg_set_id = args.arg_set_id 
            WHERE args.key = 'stream_id'
            ORDER BY slice.ts
        """)
        
        print(f"\n🏷️  Stream ID vs Track Validation:")
        mismatches = 0
        for row in stream_annotations:
            expected_track = f"HIP Activity Stream {row.stream_id}"
            actual_track = row.track_name
            
            if row.stream_id != "0":  # Non-default streams should always be on stream tracks
                if expected_track != actual_track and "Queue" in actual_track:
                    print(f"   ❌ {row.kernel_name}: stream_id={row.stream_id} but on {actual_track}")
                    mismatches += 1
                    bug_detected = True
                else:
                    print(f"   ✅ {row.kernel_name}: stream_id={row.stream_id} correctly on {actual_track}")
            else:
                # Default stream (0) can legitimately be on queue tracks if group_by_queue=false
                print(f"   ⚪ {row.kernel_name}: stream_id={row.stream_id} on {actual_track} (default stream)")
        
        if mismatches > 0:
            print(f"\n🚨 Found {mismatches} stream/track mismatches indicating the correlation bug!")
        
    except Exception as ex:
        print(f"Stream annotation analysis failed: {ex}")
    
    return bug_detected

def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_broken_behavior.py <trace.proto>")
        print("\nThis script detects the _group_by_queue variable contamination bug")
        print("Expected pattern for BROKEN behavior:")
        print("  - First few kernels on 'HIP Activity Stream X' tracks")  
        print("  - After stream_id=0 kernel, ALL subsequent kernels on 'Queue' tracks")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    
    if not os.path.exists(trace_file):
        print(f"❌ Trace file not found: {trace_file}")
        sys.exit(1)
    
    bug_detected = detect_stream_correlation_bug(trace_file)
    
    if bug_detected:
        print(f"\n❌ BUG CONFIRMED: Stream correlation issue detected!")
        print(f"   The _group_by_queue variable contamination bug is present.")
        print(f"   Kernels are incorrectly grouped due to leaked state.")
        sys.exit(1)
    else:
        print(f"\n✅ No obvious correlation bug detected")
        print(f"   Stream grouping appears to be working correctly.")
        sys.exit(0)

if __name__ == "__main__":
    main()