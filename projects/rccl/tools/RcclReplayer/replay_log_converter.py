"""
RCCLReplayer Log Format Converter
Converts between JSON and binary log formats for RCCL Replayer

Usage:
    python3 replay_log_converter.py <base_logname> tobin [new_basename]
    python3 replay_log_converter.py <base_logname> tojson [new_basename]
    python3 replay_log_converter.py <base_logname> tojson [new_basename] --standardize
    python3 replay_log_converter.py <base_logname> --standardize
    python3 replay_log_converter.py <base_logname> --sanitize
    python3 replay_log_converter.py <base_logname> --sanitize --nts
"""

import sys
import os
import struct
import re
import glob
import json
import argparse

# rcclCall_t enum mapping (from recorder.h)
RCCL_CALL_TYPES = {
    "Broadcast": 0,
    "Reduce": 1,
    "AllGather": 2,
    "ReduceScatter": 3,
    "AllReduce": 4,
    "AllReduceWithBias": 5,
    "Send": 6,
    "Recv": 7,
    "AllToAll": 8,
    "AllToAllv": 9,
    "Gather": 10,
    "Scatter": 11,
    "Bcast": 12,
    "GroupStart": 13,
    "GroupEnd": 14,
    "GroupSimulatedEnd": 15,
    "GetUniqueId": 16,
    "CommInitDev": 17,
    "CommInitRank": 18,
    "CommInitAll": 19,
    "CommInitRankConfig": 20,
    "CommSplit": 21,
    "CommFinalize": 22,
    "CommDestroy": 23,
    "CommAbort": 24,
    "CommRegister": 25,
    "CommDeregister": 26,
    "MemAlloc": 27,
    "MemFree": 28,
    "RedOpCreatePreMulSum": 29,
    "RedOpDestroy": 30,
    "OtherCall": 31
}

# Reverse mapping. This is needed to convert the binary log to JSON.
RCCL_CALL_NAMES = {v: k for k, v in RCCL_CALL_TYPES.items()}

# Call type groups for cleaner code
COMM_LIFECYCLE_CALLS = ["CommDestroy", "CommFinalize", "CommAbort"]
COLLECTIVE_CALLS = ["AllReduce", "Reduce", "AllGather", "ReduceScatter", "Broadcast", 
                    "Scatter", "Gather", "Bcast", "Send", "Recv", "AllToAll", "AllToAllv"]

# rcclApiCall struct format (based on recorder.h)
# Corresponds to the C struct rcclApiCall
STRUCT_FORMAT = (
    'i'  # pid               - 4 bytes
    'i'  # tid               - 4 bytes
    'i'  # hipDev            - 4 bytes
    'i'  # groupDepth        - 4 bytes
    'd'  # timestamp         - 8 bytes
    'Q'  # graphID           - 8 bytes
    'i'  # graphCaptured     - 4 bytes
    'i'  # type (rcclCall_t) - 4 bytes
    'Q'  # opCount           - 8 bytes
    'Q'  # sendbuff          - 8 bytes
    'Q'  # recvbuff          - 8 bytes
    'Q'  # acc               - 8 bytes
    'Q'  # sendPtrBase       - 8 bytes
    'Q'  # recvPtrBase       - 8 bytes
    'Q'  # sendPtrExtent     - 8 bytes
    'Q'  # recvPtrExtent     - 8 bytes
    'Q'  # count             - 8 bytes
    'i'  # datatype          - 4 bytes
    'i'  # op                - 4 bytes
    'i'  # root              - 4 bytes
    'i'  # nRanks            - 4 bytes
    'Q'  # comm              - 8 bytes
    'Q'  # stream            - 8 bytes
    'i'  # nTasks            - 4 bytes
    'i'  # globalRank        - 4 bytes
    'Q'  # commId            - 8 bytes
)  # Total: 160 bytes 

# Calculate the size of the struct in bytes
STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT) 

class Sanitizer:
    # Tracks and remaps pointer values to human-readable identifiers.
    
    # Regex patterns for sanitization (compiled once at class level)
    HEX_PATTERN = re.compile(r'0x[0-9a-fA-F]+')
    UNIQUEID_PATTERN = re.compile(r'uniqueID\s*:\s*(\d+)')
    TIME_PATTERN = re.compile(r'time\s*:\s*([\d.]+)')
    THREAD_PATTERN = re.compile(r'thread\s*:\s*(\d+)')
    PID_PATTERN = re.compile(r'pid\s*:\s*(\d+)')
    CONTEXT_LOOKBACK = 20  # Characters to look back for context detection
    
    def __init__(self):
        self.comm_map = {}
        self.uniqueid_map = {}
        self.stream_map = {}
        self.buffer_map = {}
        self.handle_map = {}
        self.thread_map = {}
        self.pid_map = {}
    
    def _sanitize_value(self, value, mapping, prefix):
        if value is None or value == 0 or value == -1:
            return value
        if value not in mapping:
            mapping[value] = f"{prefix}_{len(mapping) + 1:03d}"
        return mapping[value]
    
    def sanitize_comm(self, value):
        # Sanitize communicator pointer.
        return self._sanitize_value(value, self.comm_map, "comm")
    
    def sanitize_uniqueid(self, value):
        # Sanitize unique ID / commId.
        return self._sanitize_value(value, self.uniqueid_map, "uniqueid")
    
    def sanitize_stream(self, value):
        # Sanitize stream pointer.
        return self._sanitize_value(value, self.stream_map, "stream")
    
    def sanitize_buffer(self, value):
        # Sanitize buffer pointer (sendbuff, recvbuff, acc, base addresses).
        return self._sanitize_value(value, self.buffer_map, "buf")
    
    def sanitize_handle(self, value):
        # Sanitize handle (CommRegister/Deregister).
        return self._sanitize_value(value, self.handle_map, "handle")
    
    def sanitize_thread(self, value):
        # Sanitize thread ID.
        return self._sanitize_value(value, self.thread_map, "thread")
    
    def sanitize_pid(self, value):
        # Sanitize process ID.
        return self._sanitize_value(value, self.pid_map, "pid")
    
    def sanitize_by_type(self, value, value_type):
        # Sanitize a value based on its type.
        if value_type == 'comm':
            return self.sanitize_comm(value)
        elif value_type == 'uniqueid':
            return self.sanitize_uniqueid(value)
        elif value_type == 'stream':
            return self.sanitize_stream(value)
        elif value_type == 'buffer':
            return self.sanitize_buffer(value)
        elif value_type == 'handle':
            return self.sanitize_handle(value)
        return value
    
    def determine_hex_type(self, line, match_start):
        # Determine the type of hex value based on context.
        start = max(0, match_start - self.CONTEXT_LOOKBACK)
        context = line[start:match_start]
        
        if 'comm :' in context or 'newcomm :' in context:
            return 'comm'
        elif 'uniqueID :' in context:
            return 'uniqueid'
        elif 'stream :' in context:
            return 'stream'
        elif any(kw in context for kw in ['addr :', 'base :', 'ptr :', 'acc :']):
            return 'buffer'
        elif 'handle :' in context:
            return 'handle'
        return None

def parse_hex_or_int(value_str):
    # Parse hex (0x...) or decimal integer, handle (nil), or keep sanitized strings
    value_str = value_str.strip()
    if value_str == "(nil)" or value_str == "":
        return 0
    try:
        if value_str.startswith("0x"):
            return int(value_str, 16)
        else:
            return int(value_str)
    except ValueError:
        # If it's not a valid number, return the string as-is (e.g., "comm_001", "uniqueid_001")
        return value_str

def format_hex_or_string(value):
    # Format value as hex if it's an integer, or return as-is if it's a string (sanitized)
    if isinstance(value, str):
        return value  # Already a sanitized string like "comm_001"
    elif isinstance(value, int):
        return hex(value) if value != 0 else "0x0"
    return "0x0"

def parse_buffer_field(line, field_name):
    # Matches "field : [... addr : XXX ... base : YYY ... size : ZZZ]" pattern
    pattern = rf'{field_name}\s*:\s*\[.*?addr\s*:\s*(\w+).*?base\s*:\s*(\w+).*?size\s*:\s*(\d+)'
    match = re.search(pattern, line)
    
    if match:
        return {
            'addr': parse_hex_or_int(match.group(1)),
            'base': parse_hex_or_int(match.group(2)),
            'size': parse_hex_or_int(match.group(3))
        }
    return None

def parse_array_field(line, field_name):
    # Parse array of integers from JSON format like 'sendcounts : []'
    # Matches "field : [num1, num2, num3, ...]" pattern
    pattern = rf'{field_name}\s*:\s*\[([\d\s,]+)\]'
    match = re.search(pattern, line)
    
    if match:
        # Split by comma and convert to integers
        return [int(x.strip()) for x in match.group(1).split(',') if x.strip()]
    return None

def format_pointer(val, none_str='(nil)'):
    # Format pointer value: 0/None -> none_str, string as-is, else hex.
    if val is None or val == 0:
        return none_str
    if isinstance(val, str):
        return val  # Already a sanitized string like "comm_001"
    return hex(val)

def format_context(call_data):
    # Format context string for binary->JSON output.
    return (f"context : [pid : {call_data['pid']}, "
            f"time : {call_data['timestamp']:.6f}, "
            f"thread : {call_data['tid']}, "
            f"device : {call_data['hipDev']}, "
            f"groupDepth : {call_data['groupDepth']}, "
            f"captured : {call_data['graphCaptured']}, "
            f"graphID : {call_data['graphID']} ]")

def extract_value(line, key):
    # Extract value after 'key :' in the line
    # Matches "key : value" pattern, capturing everything until comma/bracket/brace
    pattern = rf'{re.escape(key)}\s*:\s*([^,\]}}]+)'
    match = re.search(pattern, line)
    if match:
        return match.group(1).strip()
    return None

def extract_context(line):
    # Extract context section from line
    # Matches "context : [...]]]" and captures the content between first [ and ]]
    match = re.search(r'context\s*:\s*\[(.*?)\]\]', line)
    if match:
        context_str = match.group(1)
        context = {}
        context['pid'] = int(extract_value(context_str, 'pid') or '-1')
        context['time'] = float(extract_value(context_str, 'time') or '-1')
        context['thread'] = int(extract_value(context_str, 'thread') or '-1')
        context['device'] = int(extract_value(context_str, 'device') or '-1')
        context['groupDepth'] = int(extract_value(context_str, 'groupDepth') or '-1')
        context['captured'] = int(extract_value(context_str, 'captured') or '-1')
        context['graphID'] = int(extract_value(context_str, 'graphID') or '0')
        return context
    return None

def parse_json_line(line):
    # Parse a single line of recorder's JSON format into struct field dict.
    line = line.strip()
    
    # Skip empty, braces, version
    if not line or line in ['{', '}', '},'] or 'version' in line:
        return None
    
    # Extract call type
    # Matches word characters at start of line followed by colon (e.g., "CommInitAll :")
    match = re.match(r'(\w+)\s*:', line)
    if not match:
        return None
    
    call_type_str = match.group(1)
    if call_type_str not in RCCL_CALL_TYPES:
        return None
    
    call_type = RCCL_CALL_TYPES[call_type_str]
    
    # Initialize struct data with defaults
    data = {
        'pid': -1,
        'tid': -1,
        'hipDev': -1,
        'groupDepth': -1,
        'timestamp': -1.0,
        'graphID': 0,
        'graphCaptured': -1,
        'type': call_type,
        'opCount': 0,
        'sendbuff': 0,
        'recvbuff': 0,
        'acc': 0,
        'sendPtrBase': 0,
        'recvPtrBase': 0,
        'sendPtrExtent': 0,
        'recvPtrExtent': 0,
        'count': 0,
        'datatype': 0,
        'op': 0,
        'root': -1,
        'nRanks': -1,
        'comm': 0,
        'stream': 0,
        'nTasks': -1,
        'globalRank': -1,
        'commId': 0
    }
    
    # Extract context
    context = extract_context(line)
    if context:
        data['pid'] = context['pid']
        data['timestamp'] = context['time']
        data['tid'] = context['thread']
        data['hipDev'] = context['device']
        data['groupDepth'] = context['groupDepth']
        data['graphCaptured'] = context['captured']
        data['graphID'] = context['graphID']
    
    # Parse call-specific fields
    if call_type_str == "GetUniqueId":
        val = extract_value(line, 'uniqueID')
        if val:
            data['commId'] = parse_hex_or_int(val)
    
    elif call_type_str == "CommInitRank":
        if val := extract_value(line, 'size'):
            data['nRanks'] = parse_hex_or_int(val)
        if val := extract_value(line, 'uniqueID'):
            data['commId'] = parse_hex_or_int(val)
        if val := extract_value(line, 'rank'):
            data['globalRank'] = parse_hex_or_int(val)
    
    elif call_type_str == "CommInitDev":
        if val := extract_value(line, 'comm'):
            data['comm'] = parse_hex_or_int(val)
        if val := extract_value(line, 'size'):
            data['nRanks'] = parse_hex_or_int(val)
        if val := extract_value(line, 'uniqueID'):
            data['commId'] = parse_hex_or_int(val)
        if val := extract_value(line, 'rank'):
            data['globalRank'] = parse_hex_or_int(val)
        if val := extract_value(line, 'dev'):
            data['root'] = parse_hex_or_int(val)
    
    elif call_type_str == "CommInitAll":
        if val := extract_value(line, '# of device'):
            data['root'] = parse_hex_or_int(val)
    
    elif call_type_str == "CommSplit":
        if val := extract_value(line, 'comm'):
            data['commId'] = parse_hex_or_int(val)
        if val := extract_value(line, 'color'):
            data['nRanks'] = parse_hex_or_int(val)
        if val := extract_value(line, 'key'):
            data['globalRank'] = parse_hex_or_int(val)
        if val := extract_value(line, 'newcomm'):
            data['comm'] = parse_hex_or_int(val)
    
    elif call_type_str in COMM_LIFECYCLE_CALLS:
        if val := extract_value(line, 'comm'):
            data['comm'] = parse_hex_or_int(val)
    
    elif call_type_str == "MemAlloc":
        if val := extract_value(line, 'returned ptr'):
            data['recvbuff'] = parse_hex_or_int(val)
        if val := extract_value(line, 'size'):
            data['count'] = parse_hex_or_int(val)
    
    elif call_type_str == "MemFree":
        if val := extract_value(line, 'ptr'):
            data['recvbuff'] = parse_hex_or_int(val)
    
    elif call_type_str == "CommRegister":
        if val := extract_value(line, 'comm'):
            data['comm'] = parse_hex_or_int(val)
        # Extract buffer info
        buff_info = parse_buffer_field(line, 'buff')
        if buff_info:
            data['sendbuff'] = buff_info['addr']
            data['sendPtrBase'] = buff_info['base']
            data['sendPtrExtent'] = buff_info['size']
        if val := extract_value(line, 'returned handle'):
            data['recvbuff'] = parse_hex_or_int(val)
    
    elif call_type_str == "CommDeregister":
        if val := extract_value(line, 'comm'):
            data['comm'] = parse_hex_or_int(val)
        if val := extract_value(line, 'handle'):
            data['recvbuff'] = parse_hex_or_int(val)
    
    # Collective operations
    elif call_type < RCCL_CALL_TYPES["GroupStart"]:
        # Extract buffer sections
        sendbuff_info = parse_buffer_field(line, 'sendbuff')
        if sendbuff_info:
            data['sendbuff'] = sendbuff_info['addr']
            data['sendPtrBase'] = sendbuff_info['base']
            data['sendPtrExtent'] = sendbuff_info['size']
        
        recvbuff_info = parse_buffer_field(line, 'recvbuff')
        if recvbuff_info:
            data['recvbuff'] = recvbuff_info['addr']
            data['recvPtrBase'] = recvbuff_info['base']
            data['recvPtrExtent'] = recvbuff_info['size']
        
        if val := extract_value(line, 'opCount'):
            data['opCount'] = parse_hex_or_int(val)
        if val := extract_value(line, 'acc'):
            data['acc'] = parse_hex_or_int(val)
        if val := extract_value(line, 'count'):
            data['count'] = parse_hex_or_int(val)
        if val := extract_value(line, 'datatype'):
            data['datatype'] = parse_hex_or_int(val)
        if val := extract_value(line, 'op'):
            data['op'] = parse_hex_or_int(val)
        if val := extract_value(line, 'root'):
            data['root'] = parse_hex_or_int(val)
        if val := extract_value(line, 'comm'):
            data['comm'] = parse_hex_or_int(val)
        if val := extract_value(line, 'nranks'):
            data['nRanks'] = parse_hex_or_int(val)
        if val := extract_value(line, 'stream'):
            data['stream'] = parse_hex_or_int(val)
        if val := extract_value(line, 'task'):
            data['nTasks'] = parse_hex_or_int(val)
        if val := extract_value(line, 'globalrank'):
            data['globalRank'] = parse_hex_or_int(val)
        
        # AllToAllv has 4 extra arrays appended after the struct
        if call_type_str == "AllToAllv":
            data['alltoallv_arrays'] = {
                'sendcounts': parse_array_field(line, 'sendcounts'),
                'sdispls': parse_array_field(line, 'sdispls'),
                'recvcounts': parse_array_field(line, 'recvcounts'),
                'rdispls': parse_array_field(line, 'rdispls')
            }
    
    return data

def json_to_bin(json_file, bin_file):
    # Convert JSON log to binary format
    print(f"Converting {json_file} to binary format...")
    
    # Validate input file
    if not os.path.exists(json_file):
        print(f"Error: Input file not found: {json_file}")
        return False
    
    if os.path.getsize(json_file) == 0:
        print(f"Warning: Input file is empty: {json_file}")
        return False
    
    try:
        with open(json_file, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error: Failed to read input file: {json_file}")
        print(f"  {e}")
        return False
    
    call_count = 0
    with open(bin_file, 'wb') as f:
        for line in lines:
            # Parse the line
            data = parse_json_line(line)
            
            # Skip lines that are just braces or empty
            if not data:
                continue
            
            # Pack struct
            packed = struct.pack(
                STRUCT_FORMAT,
                data['pid'],
                data['tid'],
                data['hipDev'],
                data['groupDepth'],
                data['timestamp'],
                data['graphID'],
                data['graphCaptured'],
                data['type'],
                data['opCount'],
                data['sendbuff'],
                data['recvbuff'],
                data['acc'],
                data['sendPtrBase'],
                data['recvPtrBase'],
                data['sendPtrExtent'],
                data['recvPtrExtent'],
                data['count'],
                data['datatype'],
                data['op'],
                data['root'],
                data['nRanks'],
                data['comm'],
                data['stream'],
                data['nTasks'],
                data['globalRank'],
                data['commId']
            )
            f.write(packed)
            
            # Write extra data for AllToAllv (4 arrays of int32)
            if data['type'] == RCCL_CALL_TYPES["AllToAllv"] and 'alltoallv_arrays' in data:
                arrays = data['alltoallv_arrays']
                nRanks = data.get('nRanks', -1)
                
                # Validate nRanks
                if nRanks <= 0:
                    print(f"Warning: Invalid nRanks={nRanks} for AllToAllv at call {call_count + 1}")
                    print(f"Skipping array data write.")
                else:
                    # Write sendcounts, sdispls, recvcounts, rdispls (each is nRanks * int32)
                    for array_name in ['sendcounts', 'sdispls', 'recvcounts', 'rdispls']:
                        if arrays.get(array_name):
                            array_len = len(arrays[array_name])
                            if array_len != nRanks:
                                print(f"Warning: {array_name} length mismatch for AllToAllv at call {call_count + 1}")
                                print(f"Expected {nRanks} elements, got {array_len} elements")
                            # Pack as array of int32 (signed 32-bit integers, 4 bytes each)
                            for val in arrays[array_name]:
                                f.write(struct.pack('i', val))
            
            call_count += 1
    
    print(f"Converted {call_count} calls to binary format: {bin_file} \n")

def bin_to_json(bin_file, json_file):
    # Convert binary log to JSON format
    print(f"Converting {bin_file} to JSON format...")
    
    # Validate input file
    if not os.path.exists(bin_file):
        print(f"Error: Input file not found: {bin_file}")
        return False
    
    if os.path.getsize(bin_file) == 0:
        print(f"Warning: Input file is empty: {bin_file}")
        return False
    
    file_size = os.path.getsize(bin_file)
    if file_size < STRUCT_SIZE:
        print(f"Error: Input file too small: {bin_file}")
        print(f"  File size: {file_size} bytes, minimum expected: {STRUCT_SIZE} bytes")
        return False
    
    # Read binary file
    # Parse all calls first - read sequentially to handle variable-size records
    all_calls = []
    record_num = 0
    try:
        with open(bin_file, 'rb') as f:
            while True:
                # Read the fixed-size struct
                chunk = f.read(STRUCT_SIZE)
                if len(chunk) < STRUCT_SIZE:
                    break  # End of file
                
                try:
                    unpacked = struct.unpack(STRUCT_FORMAT, chunk)
                except struct.error as e:
                    print(f"Error: Failed to unpack record {record_num}")
                    print(f"  Expected {STRUCT_SIZE} bytes, got {len(chunk)} bytes")
                    print(f"  Error: {e}")
                    break
                
                record_num += 1
                
                call_data = {
                    'pid': unpacked[0],
                    'tid': unpacked[1],
                    'hipDev': unpacked[2],
                    'groupDepth': unpacked[3],
                    'timestamp': unpacked[4],
                    'graphID': unpacked[5],
                    'graphCaptured': unpacked[6],
                    'type': unpacked[7],
                    'opCount': unpacked[8],
                    'sendbuff': unpacked[9],
                    'recvbuff': unpacked[10],
                    'acc': unpacked[11],
                    'sendPtrBase': unpacked[12],
                    'recvPtrBase': unpacked[13],
                    'sendPtrExtent': unpacked[14],
                    'recvPtrExtent': unpacked[15],
                    'count': unpacked[16],
                    'datatype': unpacked[17],
                    'op': unpacked[18],
                    'root': unpacked[19],
                    'nRanks': unpacked[20],
                    'comm': unpacked[21],
                    'stream': unpacked[22],
                    'nTasks': unpacked[23],
                    'globalRank': unpacked[24],
                    'commId': unpacked[25]
                }
                
                # Read extra data for AllToAllv (4 arrays of nRanks * int32)
                if call_data['type'] == RCCL_CALL_TYPES["AllToAllv"]:
                    nRanks = call_data['nRanks']
                    
                    # Validate nRanks before reading arrays
                    if nRanks <= 0:
                        print(f"Warning: Invalid nRanks={nRanks} for AllToAllv at record {record_num}")
                        print(f"  Skipping array data read. Arrays will be empty.")
                        call_data['alltoallv_arrays'] = {
                            'sendcounts': [], 'sdispls': [], 'recvcounts': [], 'rdispls': []
                        }
                    else:
                        call_data['alltoallv_arrays'] = {}
                        for array_name in ['sendcounts', 'sdispls', 'recvcounts', 'rdispls']:
                            array_data = []
                            for i in range(nRanks):
                                val_bytes = f.read(4)  # int32 is 4 bytes
                                if len(val_bytes) < 4:
                                    print(f"Warning: Incomplete {array_name} array for AllToAllv at record {record_num}")
                                    print(f"  Expected {nRanks} elements, got {i} elements")
                                    break
                                array_data.append(struct.unpack('i', val_bytes)[0])
                            call_data['alltoallv_arrays'][array_name] = array_data
                
                all_calls.append(call_data)
    except IOError as e:
        print(f"Error: Failed to read binary file: {bin_file}")
        print(f"  {e}")
        return False
    
    num_calls = len(all_calls)
    
    # Write JSON file
    with open(json_file, 'w') as f:
        f.write("{\n")
        f.write("  version : 1,\n")
        
        # Track depth for proper indentation (2 + 2*depth spaces)
        # GroupStart adds opening brace and increments depth
        # GroupEnd decrements depth and adds closing brace
        depth = 0
        
        i = 0
        while i < num_calls:
            call_data = all_calls[i]
            call_type_name = RCCL_CALL_NAMES.get(call_data['type'], 'OtherCall')
            
            # Handle GroupEnd: decrement depth BEFORE writing
            if call_type_name == "GroupEnd":
                depth -= 1
                # Write closing brace before GroupEnd
                indent_for_brace = ' ' * (2 + 2 * depth)
                f.write(f"{indent_for_brace}}},\n")
            
            # Calculate indentation based on current depth
            indent = ' ' * (2 + 2 * depth)
            
            # Format output based on call type
            context = format_context(call_data)
            
            if call_type_name == "GetUniqueId":
                f.write(f"{indent}{call_type_name} : [uniqueID : {call_data['commId']}, {context}]")
            
            elif call_type_name == "CommInitRank":
                f.write(f"{indent}{call_type_name} : [size : {call_data['nRanks']}, uniqueID : {call_data['commId']}, rank : {call_data['globalRank']}, {context}]")
            
            elif call_type_name == "CommInitDev":
                f.write(f"{indent}{call_type_name} : [comm : {hex(call_data['comm'])}, size : {call_data['nRanks']}, uniqueID : {call_data['commId']}, rank : {call_data['globalRank']}, dev : {call_data['root']}, {context}]")
            
            elif call_type_name == "CommInitAll":
                f.write(f"{indent}{call_type_name} : [# of device : {call_data['root']}, {context}]")
            
            elif call_type_name == "CommSplit":
                # Single CommSplit (shouldn't happen if grouped correctly, but handle it)
                f.write(f"{indent}{call_type_name} : [comm : {hex(call_data['commId'])}, color : {call_data['nRanks']}, key : {call_data['globalRank']}, newcomm : {format_pointer(call_data['comm'])}, {context}]")
            
            elif call_type_name in COMM_LIFECYCLE_CALLS:
                f.write(f"{indent}{call_type_name} : [comm : {format_pointer(call_data['comm'])}, {context}]")
            
            elif call_type_name == "MemAlloc":
                f.write(f"{indent}{call_type_name} : [returned ptr : {hex(call_data['recvbuff'])}, size : {call_data['count']}, {context}]")
            
            elif call_type_name == "MemFree":
                f.write(f"{indent}{call_type_name} : [ptr : {hex(call_data['recvbuff'])}, {context}]")
            
            elif call_type_name == "CommRegister":
                f.write(f"{indent}{call_type_name} : [comm : {hex(call_data['comm'])}, buff : [addr : {hex(call_data['sendbuff'])}, base : {hex(call_data['sendPtrBase'])}, size : {call_data['sendPtrExtent']}], returned handle : {hex(call_data['recvbuff'])}, {context}]")
            
            elif call_type_name == "CommDeregister":
                f.write(f"{indent}{call_type_name} : [comm : {hex(call_data['comm'])}, handle : {hex(call_data['recvbuff'])}, {context}]")
            
            # Collective operations
            elif call_data['type'] < RCCL_CALL_TYPES["GroupStart"]:
                # Format acc field using format_pointer
                acc_str = format_pointer(call_data['acc'])
                f.write(f"{indent}{call_type_name} : [opCount : {call_data['opCount']}, ")
                f.write(f"sendbuff : [addr : {hex(call_data['sendbuff'])}, base : {hex(call_data['sendPtrBase'])}, size : {call_data['sendPtrExtent']}], ")
                f.write(f"recvbuff : [addr : {hex(call_data['recvbuff'])}, base : {hex(call_data['recvPtrBase'])}, size : {call_data['recvPtrExtent']}], ")
                f.write(f"acc : {acc_str}, ")
                f.write(f"count : {call_data['count']}, datatype : {call_data['datatype']}, op : {call_data['op']}, ")
                f.write(f"root : {call_data['root']}, comm : {hex(call_data['comm'])}, nranks : {call_data['nRanks']}, ")
                f.write(f"stream : {hex(call_data['stream'])}, task : {call_data['nTasks']}, globalrank : {call_data['globalRank']}, {context}]")
                
                # Write AllToAllv arrays if present
                if call_type_name == "AllToAllv" and 'alltoallv_arrays' in call_data:
                    arrays = call_data['alltoallv_arrays']
                    for array_name in ['sendcounts', 'sdispls', 'recvcounts', 'rdispls']:
                        if arrays[array_name]:
                            array_str = ', '.join(str(x) for x in arrays[array_name])
                            f.write(f", {array_name} : [{array_str}]")
            
            else:
                f.write(f"{indent}{call_type_name} : [{context}]")
            
            # Add comma and handle GroupStart opening brace
            if call_type_name == "GroupStart":
                # Write comma after GroupStart, then opening brace on next line
                f.write(",\n")
                f.write(f"{indent}{{\n")
                # Increment depth for content inside the group
                depth += 1
            elif i < num_calls - 1:
                # Regular comma for non-GroupStart calls
                f.write(",\n")
            else:
                f.write("\n")
            
            i += 1
        
        f.write("}\n")
    
    print(f"Converted {num_calls} calls to JSON format: {json_file} \n")

def find_log_files(base_name, extension=None):
    # Find all log files matching the base name pattern.
    # Pattern: <basename>.<pid>.<hostname>[.json]
    
    # Handle if user provides extension in base_name
    if base_name.endswith('.json'):
        base_name = base_name[:-5]
    
    # Try different patterns to find matching files
    if extension == ".json":
        # Looking for JSON files: <basename>.*.*.<extension>
        pattern = f"{base_name}.*.*.json"
    else:
        # Looking for binary files: <basename>.*.*
        # But exclude .json files
        pattern = f"{base_name}.*.*"
    
    files = []
    for f in glob.glob(pattern):
        # For binary mode, skip .json files
        if extension is None and f.endswith('.json'):
            continue
        # Verify it matches the expected pattern: basename.number.text[.json]
        # This filters out files like basename.json without the pid.hostname
        base_without_ext = f[:-5] if f.endswith('.json') else f
        parts = base_without_ext.split('.')
        if len(parts) >= 3:  # basename.pid.hostname (at least)
            files.append(f)
    
    return sorted(files)

def sanitize_json_file(input_file, output_file, zero_timestamps=False):
    # Sanitize JSON log file for easier comparison.
    print(f"Sanitizing {input_file} to {output_file}")
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error reading {input_file}: {e}")
        return
    
    sanitizer = Sanitizer()
    min_timestamp = float('inf')
    
    # First pass: collect all unique values to build mappings
    for line in lines:
        # Collect hex values
        for match in Sanitizer.HEX_PATTERN.finditer(line):
            hex_val = int(match.group(), 16)
            if hex_val > 0:
                value_type = sanitizer.determine_hex_type(line, match.start())
                if value_type:
                    sanitizer.sanitize_by_type(hex_val, value_type)
        
        # Collect decimal uniqueID values
        for match in Sanitizer.UNIQUEID_PATTERN.finditer(line):
            uniqueid_val = int(match.group(1))
            if uniqueid_val > 0:
                sanitizer.sanitize_uniqueid(uniqueid_val)
        
        # Collect thread IDs
        for match in Sanitizer.THREAD_PATTERN.finditer(line):
            thread_val = int(match.group(1))
            if thread_val > 0:
                sanitizer.sanitize_thread(thread_val)
        
        # Collect PIDs
        for match in Sanitizer.PID_PATTERN.finditer(line):
            pid_val = int(match.group(1))
            if pid_val > 0:
                sanitizer.sanitize_pid(pid_val)
        
        # Find minimum timestamp (only if not zero_timestamps mode)
        if not zero_timestamps:
            time_match = Sanitizer.TIME_PATTERN.search(line)
            if time_match:
                timestamp = float(time_match.group(1))
                if timestamp > 0:
                    min_timestamp = min(min_timestamp, timestamp)
    
    if min_timestamp == float('inf'):
        min_timestamp = 0.0
    
    # Second pass: replace values in-place
    try:
        with open(output_file, 'w') as f:
            for line in lines:
                new_line = line
                
                # Replace hex values
                for match in reversed(list(Sanitizer.HEX_PATTERN.finditer(line))):
                    hex_val = int(match.group(), 16)
                    if hex_val == 0:
                        replacement = '(nil)'
                    else:
                        value_type = sanitizer.determine_hex_type(line, match.start())
                        if value_type:
                            replacement = sanitizer.sanitize_by_type(hex_val, value_type)
                        else:
                            replacement = match.group()
                    
                    # Replace from end to start to preserve match positions
                    new_line = new_line[:match.start()] + replacement + new_line[match.end():]
                
                # Replace decimal uniqueID values
                for match in reversed(list(Sanitizer.UNIQUEID_PATTERN.finditer(new_line))):
                    uniqueid_val = int(match.group(1))
                    if uniqueid_val > 0:
                        sanitized = sanitizer.sanitize_uniqueid(uniqueid_val)
                        replacement = f"uniqueID : {sanitized}"
                        new_line = new_line[:match.start()] + replacement + new_line[match.end():]
                
                # Replace thread IDs
                for match in reversed(list(Sanitizer.THREAD_PATTERN.finditer(new_line))):
                    thread_val = int(match.group(1))
                    if thread_val > 0:
                        sanitized = sanitizer.sanitize_thread(thread_val)
                        replacement = f"thread : {sanitized}"
                        new_line = new_line[:match.start()] + replacement + new_line[match.end():]
                
                # Replace PIDs
                for match in reversed(list(Sanitizer.PID_PATTERN.finditer(new_line))):
                    pid_val = int(match.group(1))
                    if pid_val > 0:
                        sanitized = sanitizer.sanitize_pid(pid_val)
                        replacement = f"pid : {sanitized}"
                        new_line = new_line[:match.start()] + replacement + new_line[match.end():]
                
                # Replace timestamps
                for match in reversed(list(Sanitizer.TIME_PATTERN.finditer(new_line))):
                    timestamp = float(match.group(1))
                    if zero_timestamps:
                        normalized = 0.0
                    else:
                        normalized = timestamp - min_timestamp
                    replacement = f"time : {normalized:.6f}"
                    new_line = new_line[:match.start()] + replacement + new_line[match.end():]
                
                f.write(new_line)
    except IOError as e:
        print(f"Error writing to {output_file}: {e}")
        return

def standardize_json_file(input_file, output_file):
    # Convert non-standard JSON to standard parseable JSON
    print(f"Standardizing {input_file} to {output_file}...")
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse the non-standard JSON and convert to standard
    result = {
        "version": 1,
        "calls": []
    }
    
    in_group = False
    current_group = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip root braces and version line
        if stripped == '{' and not result["calls"]:  # Root opening brace
            continue
        if stripped.startswith('version'):
            continue
        if stripped == '}' and not in_group:  # Root closing brace
            continue
        
        # Detect group start (for CommSplit grouping)
        if stripped == '{':
            in_group = True
            current_group = []
            continue
        
        # Detect group end
        if stripped in ['}', '},']:
            if in_group and current_group:
                result["calls"].append({
                    "type": "GroupedCalls",
                    "calls": current_group
                })
                current_group = []
                in_group = False
            continue
        
        # Parse call line
        call_dict = parse_nonstandard_json_line(stripped)
        if call_dict:
            if in_group:
                current_group.append(call_dict)
            else:
                result["calls"].append(call_dict)
    
    # Write standard JSON
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # print(f"Standardized to {output_file}")

def transform_struct_to_standard_json(struct_data, call_type_str):
    # Transform struct-format dict (from parse_json_line) to standard JSON format dict.
    if not struct_data:
        return None
    
    call_dict = {"type": call_type_str}
    
    # Transform context - include all available fields
    context = {}
    if struct_data.get('pid') is not None:
        context["pid"] = struct_data['pid']
    if struct_data.get('timestamp') is not None:
        context["time"] = struct_data['timestamp']
    if struct_data.get('tid') is not None:
        context["thread"] = struct_data['tid']
    if struct_data.get('hipDev') is not None:
        context["device"] = struct_data['hipDev']
    if struct_data.get('groupDepth') is not None:
        context["groupDepth"] = struct_data['groupDepth']
    if struct_data.get('graphCaptured') is not None:
        context["captured"] = struct_data['graphCaptured']
    if struct_data.get('graphID') is not None:
        context["graphID"] = struct_data['graphID']
    
    # Only add context if we have fields
    if context:
        call_dict["context"] = context
    
    # Call-specific field transformations
    if call_type_str == "GetUniqueId":
        if struct_data.get('commId'):
            call_dict["uniqueID"] = struct_data['commId']
    
    elif call_type_str == "CommInitAll":
        if struct_data.get('root') != -1:
            call_dict["num_devices"] = struct_data['root']
    
    elif call_type_str == "CommInitRank":
        if struct_data.get('nRanks') != -1:
            call_dict["size"] = struct_data['nRanks']
        if struct_data.get('commId'):
            call_dict["uniqueID"] = struct_data['commId']
        if struct_data.get('globalRank') != -1:
            call_dict["rank"] = struct_data['globalRank']
    
    elif call_type_str == "CommInitDev":
        if struct_data.get('comm'):
            call_dict["comm"] = format_pointer(struct_data['comm'], None)
        if struct_data.get('nRanks') != -1:
            call_dict["size"] = struct_data['nRanks']
        if struct_data.get('commId'):
            call_dict["uniqueID"] = struct_data['commId']
        if struct_data.get('globalRank') != -1:
            call_dict["rank"] = struct_data['globalRank']
        if struct_data.get('root') != -1:
            call_dict["dev"] = struct_data['root']
    
    elif call_type_str == "CommSplit":
        if struct_data.get('commId'):
            call_dict["comm"] = format_hex_or_string(struct_data['commId'])
        if struct_data.get('nRanks') != -1:
            call_dict["color"] = struct_data['nRanks']
        if struct_data.get('globalRank') != -1:
            call_dict["key"] = struct_data['globalRank']
        if struct_data.get('comm'):
            call_dict["newcomm"] = format_pointer(struct_data['comm'], None)
    
    elif call_type_str in COMM_LIFECYCLE_CALLS:
        if struct_data.get('comm'):
            call_dict["comm"] = format_pointer(struct_data['comm'], None)
    
    elif call_type_str == "MemAlloc":
        if struct_data.get('recvbuff'):
            call_dict["returned_ptr"] = format_hex_or_string(struct_data['recvbuff'])
        if struct_data.get('count'):
            call_dict["size"] = struct_data['count']
    
    elif call_type_str == "MemFree":
        if struct_data.get('recvbuff'):
            call_dict["ptr"] = format_hex_or_string(struct_data['recvbuff'])
    
    elif call_type_str == "CommRegister":
        if struct_data.get('comm'):
            call_dict["comm"] = format_hex_or_string(struct_data['comm'])
        if struct_data.get('sendbuff') or struct_data.get('sendPtrBase') or struct_data.get('sendPtrExtent'):
            call_dict["buffer"] = {
                "addr": format_pointer(struct_data.get('sendbuff'), "0x0"),
                "base": format_pointer(struct_data.get('sendPtrBase'), "0x0"),
                "size": struct_data.get('sendPtrExtent', 0)
            }
        if struct_data.get('recvbuff'):
            call_dict["returned_handle"] = format_hex_or_string(struct_data['recvbuff'])
    
    elif call_type_str == "CommDeregister":
        if struct_data.get('comm'):
            call_dict["comm"] = format_hex_or_string(struct_data['comm'])
        if struct_data.get('recvbuff'):
            call_dict["handle"] = format_hex_or_string(struct_data['recvbuff'])
    
    # Collective operations
    elif call_type_str in COLLECTIVE_CALLS:
        if struct_data.get('opCount') is not None:
            call_dict["opCount"] = str(struct_data['opCount'])
        
        # sendbuff
        if struct_data.get('sendbuff') or struct_data.get('sendPtrBase') or struct_data.get('sendPtrExtent'):
            call_dict["sendbuff"] = {
                "addr": format_hex_or_string(struct_data.get('sendbuff', 0)),
                "base": format_hex_or_string(struct_data.get('sendPtrBase', 0)),
                "size": struct_data.get('sendPtrExtent', 0)
            }
        
        # recvbuff
        if struct_data.get('recvbuff') or struct_data.get('recvPtrBase') or struct_data.get('recvPtrExtent'):
            call_dict["recvbuff"] = {
                "addr": format_hex_or_string(struct_data.get('recvbuff', 0)),
                "base": format_hex_or_string(struct_data.get('recvPtrBase', 0)),
                "size": struct_data.get('recvPtrExtent', 0)
            }
        
        if 'acc' in struct_data:
            call_dict["acc"] = format_pointer(struct_data['acc'], None)
        if struct_data.get('count') is not None:
            call_dict["count"] = struct_data['count']
        if struct_data.get('datatype') is not None:
            call_dict["datatype"] = struct_data['datatype']
        if struct_data.get('op') is not None:
            call_dict["op"] = struct_data['op']
        if struct_data.get('root') != -1:
            call_dict["root"] = struct_data['root']
        if struct_data.get('comm'):
            call_dict["comm"] = format_hex_or_string(struct_data['comm'])
        if struct_data.get('nRanks') != -1:
            call_dict["nranks"] = struct_data['nRanks']
        if struct_data.get('stream'):
            call_dict["stream"] = format_hex_or_string(struct_data['stream'])
        if struct_data.get('nTasks') != -1:
            call_dict["task"] = struct_data['nTasks']
        if struct_data.get('globalRank') != -1:
            call_dict["globalrank"] = struct_data['globalRank']
        
        # AllToAllv arrays
        if call_type_str == "AllToAllv" and 'alltoallv_arrays' in struct_data:
            arrays = struct_data['alltoallv_arrays']
            for array_name in ['sendcounts', 'sdispls', 'recvcounts', 'rdispls']:
                if arrays.get(array_name):
                    call_dict[array_name] = arrays[array_name]
    
    return call_dict

def parse_nonstandard_json_line(line):
    # Parse a single line of recorder's JSON into standard JSON format dict.
    # First parse using the main parsing function
    struct_data = parse_json_line(line)
    if not struct_data:
        return None
    
    # Get the call type string
    call_type_str = RCCL_CALL_NAMES.get(struct_data['type'], 'OtherCall')
    
    # Transform struct format to standard JSON format
    return transform_struct_to_standard_json(struct_data, call_type_str)

def main():
    parser = argparse.ArgumentParser(
        description='RCCLReplayer Log Format Converter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('base_name', help='Base log name (e.g., std-logs-8ranks)')
    parser.add_argument('mode', nargs='?', choices=['tobin', 'tojson'], 
                       help='Conversion mode (optional if using --standardize or --sanitize)')
    parser.add_argument('new_base_name', nargs='?', help='New base name for output files (optional)')
    parser.add_argument('--standardize', action='store_true',
                       help='Generate standard JSON format (parseable by standard JSON parsers)')
    parser.add_argument('--sanitize', action='store_true',
                       help='Sanitize JSON logs in-place (normalize pointers and timestamps)')
    parser.add_argument('--no-timestamp', '--nts', action='store_true',
                       help='Set all timestamps to 0.0 (use with --sanitize)')
    
    args = parser.parse_args()
    
    base_name = args.base_name
    mode = args.mode.lower() if args.mode else None
    new_base_name = args.new_base_name
    standardize = args.standardize
    sanitize = args.sanitize
    no_timestamp = args.no_timestamp
    
    # Validate that --no-timestamp is only used with --sanitize
    if no_timestamp and not sanitize:
        print("Error: --no-timestamp can only be used with --sanitize")
        sys.exit(1)
    
    # Handle --sanitize without mode (sanitize existing JSON files)
    if sanitize and not mode:
        files = find_log_files(base_name, extension=".json")
        
        if not files:
            print(f"Error: No JSON files found matching pattern '{base_name}.*.*.json'")
            sys.exit(1)
        
        # Filter out files that are already sanitized or standardized
        files = [f for f in files if '.sanitized.' not in f and '.standard.' not in f]
        
        if not files:
            print(f"Error: No non-sanitized JSON files found matching pattern '{base_name}.*.*.json'")
            sys.exit(1)
        
        print(f"Found {len(files)} JSON file(s) to sanitize:")
        for f in files:
            print(f"  - {f}")
        print()
        
        success_count = 0
        for json_file in files:
            # Sanitize JSON file
            try:
                sanitize_json_file(json_file, json_file, no_timestamp)
                success_count += 1
            except Exception as e:
                print(f"Error sanitizing {json_file}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\nSuccessfully sanitized {success_count}/{len(files)} file(s)")
        return
    
    # Handle --standardize without mode (standardize existing JSON files)
    if standardize and not mode:
        files = find_log_files(base_name, extension=".json")
        
        if not files:
            print(f"Error: No JSON files found matching pattern '{base_name}.*.*.json'")
            sys.exit(1)
        
        # Filter out files that are already standardized (contain .standard. in filename)
        files = [f for f in files if '.standard.' not in f]
        
        if not files:
            print(f"Error: No non-standard JSON files found matching pattern '{base_name}.*.*.json'")
            sys.exit(1)
        
        print(f"Found {len(files)} JSON file(s) to standardize:")
        for f in files:
            print(f"  - {f}")
        print()
        
        success_count = 0
        for json_file in files:
            # Create standard filename: name.json -> name.standard.json
            if new_base_name:
                standard_file = json_file.replace(base_name, new_base_name, 1).replace('.json', '.standard.json')
            else:
                standard_file = json_file.replace('.json', '.standard.json')
            
            try:
                standardize_json_file(json_file, standard_file)
                success_count += 1
            except Exception as e:
                print(f"Error standardizing {json_file}: {e}")
        
        print(f"\nSuccessfully standardized {success_count}/{len(files)} file(s)")
        return
    
    # Require mode if not using --standardize alone
    if not mode:
        print("Error: mode (tobin/tojson) required when not using --standardize alone")
        parser.print_help()
        sys.exit(1)
    
    if mode == "tobin":
        # Find all JSON files matching the pattern
        files = find_log_files(base_name, extension=".json")
        
        if not files:
            print(f"Error: No JSON files found matching pattern '{base_name}.*.*.json'")
            sys.exit(1)
        
        # Filter out files that are already processed (standardized or sanitized)
        files = [f for f in files if '.standard.' not in f and '.sanitized.' not in f]
        
        if not files:
            print(f"Error: No non-converted JSON files found matching pattern '{base_name}.*.*.json'")
            sys.exit(1)
        
        print(f"Found {len(files)} JSON file(s) to convert:")
        for f in files:
            print(f"  - {f}")
        print()
        
        success_count = 0
        for json_file in files:
            if new_base_name:
                # Replace base name and remove .json extension
                bin_file = json_file.replace(base_name, new_base_name, 1)[:-5]
            else:
                # Remove .json extension for binary output
                bin_file = json_file[:-5]
            
            try:
                json_to_bin(json_file, bin_file)
                success_count += 1
            except Exception as e:
                print(f"Error converting {json_file}: {e}")
        
        print(f"\nSuccessfully converted {success_count}/{len(files)} file(s)")
    
    elif mode == "tojson":
        # Find all binary files matching the pattern
        files = find_log_files(base_name, extension=None)
        
        if not files:
            print(f"Error: No binary files found matching pattern '{base_name}.*.*'")
            sys.exit(1)
        
        print(f"Found {len(files)} binary file(s) to convert:")
        for f in files:
            print(f"  - {f}")
        print()
        
        success_count = 0
        for bin_file in files:
            if new_base_name:
                # Replace base name and add .json extension
                json_file = bin_file.replace(base_name, new_base_name, 1) + '.json'
            else:
                # Add .json extension for JSON output
                json_file = bin_file + '.json'
            
            try:
                bin_to_json(bin_file, json_file)
                success_count += 1
                
                # If --sanitize, sanitize the JSON file in-place
                if sanitize:
                    try:
                        sanitize_json_file(json_file, json_file, no_timestamp)
                    except Exception as e:
                        print(f"Error sanitizing {json_file}: {e}")
                
                # If --standardize, also generate standard JSON
                if standardize:
                    standard_file = json_file.replace('.json', '.standard.json')
                    try:
                        standardize_json_file(json_file, standard_file)
                    except Exception as e:
                        print(f"Error standardizing {json_file}: {e}")
                
            except Exception as e:
                print(f"Error converting {bin_file}: {e}")
        
        print(f"\nSuccessfully converted {success_count}/{len(files)} file(s)")
        if sanitize:
            print("Sanitized JSON files")
        if standardize:
            print("Generated standard JSON files with '.standard.json' extension")
    
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'tobin' or 'tojson'")
        sys.exit(1)

if __name__ == "__main__":
    main()

