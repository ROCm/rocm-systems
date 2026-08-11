# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

# example run
# python3 ./[rccl]/tools/scripts/npkit_trace_generator.py --npkit_dump_dir=[npkit_dump_dir] --npkit_event_header_path=[rccl]/src/include/npkit/npkit_event.h --output_dir=/home/akollias/dev/

import argparse
import os
import json

from queue import Queue

def parse_npkit_event_header(npkit_event_header_path):
    npkit_event_def = {'id_to_type': {}, 'type_to_id': {}}
    with open(npkit_event_header_path, 'r') as f:
        lines = [x.strip() for x in f.readlines() if len(x.strip()) != 0]
        line_idx = 0
        while line_idx < len(lines):
            if lines[line_idx].startswith('#define NPKIT_EVENT_'):
                fields = lines[line_idx].split()
                if len(fields) == 3:
                    event_type = fields[1]
                    event_id = int(fields[2], 0)
                    npkit_event_def['type_to_id'][event_type] = event_id
                    npkit_event_def['id_to_type'][event_id] = event_type
            line_idx += 1
    return npkit_event_def

def parse_gpu_clock_scale(gpu_clock_file_path):
    with open(gpu_clock_file_path, 'r') as f:
        freq_in_khz = f.read()
        return float(freq_in_khz) * 1e3 / 1e6

def parse_cpu_clock_scale(cpu_clock_den_file_path, cpu_clock_num_file_path):
    with open(cpu_clock_num_file_path, 'r') as f:
        num = float(f.read())
    with open(cpu_clock_den_file_path, 'r') as f:
        den = float(f.read())
    return den / num / 1e6

def parse_gpu_event(event_bytes):
    return {
        'id': int.from_bytes(event_bytes[0:1], byteorder='little', signed=False),
        'size': int.from_bytes(event_bytes[1:5], byteorder='little', signed=False),
        'rsvd': int.from_bytes(event_bytes[5:8], byteorder='little', signed=False),
        'timestamp': int.from_bytes(event_bytes[8:16], byteorder='little', signed=False)
    }

def parse_cpu_event(event_bytes):
    return {
        'id': int.from_bytes(event_bytes[0:1], byteorder='little', signed=False),
        'size': int.from_bytes(event_bytes[1:5], byteorder='little', signed=False),
        'slot': int.from_bytes(event_bytes[5:8], byteorder='little', signed=False),
        'timestamp': int.from_bytes(event_bytes[8:16], byteorder='little', signed=False)
    }

def parse_gpu_event_file_time(sync_dictionary, npkit_dump_dir, npkit_event_def, rank, buf_idx, gpu_clock_scale, cpu_clock_scale, dictionary_of_stats, warmup_runs):
    gpu_event_file_path = os.path.join(npkit_dump_dir, 'gpu_events_rank_%d_buf_%d' % (rank, buf_idx))
    stats_key = 'gpu_rank_%d' % (rank)
    channel_stats = {}
    raw_event_size = 16
    curr_cpu_base_time = None
    curr_gpu_base_time = None
    gpu_events = []
    event_type_to_seq = {}
    unfiltered_events = []
    start_event_id = 0
    with open(gpu_event_file_path, 'rb') as f:
        raw_content = f.read()
        raw_content_size = len(raw_content)
        raw_content_idx = 0
        if raw_content_size > 0 and stats_key not in dictionary_of_stats:
            dictionary_of_stats[stats_key] = {}
        warmup_raw_content_idx = 0
        while warmup_runs != 0 and warmup_raw_content_idx < raw_content_size: #warmup run cleanup
            parsed_gpu_event = parse_gpu_event(raw_content[warmup_raw_content_idx : warmup_raw_content_idx + raw_event_size])
            unfiltered_events.insert(0, parsed_gpu_event)
            if start_event_id == 0:
                decoded_id = npkit_event_def['id_to_type'][parsed_gpu_event['id']]
                if decoded_id == 'NPKIT_EVENT_TIME_SYNC_CPU' or decoded_id == 'NPKIT_EVENT_TIME_SYNC_GPU':
                    warmup_raw_content_idx += raw_event_size
                    continue
                else:
                    start_event_id = parsed_gpu_event['id']

            warmup_raw_content_idx += raw_event_size
            if parsed_gpu_event['id'] == (start_event_id + 1):
                warmup_runs -= 1
        raw_content_idx = warmup_raw_content_idx
        while raw_content_idx < raw_content_size:
            parsed_gpu_event = parse_gpu_event(raw_content[raw_content_idx : raw_content_idx + raw_event_size])
            if npkit_event_def['id_to_type'][parsed_gpu_event['id']] in ['NPKIT_EVENT_TIME_SYNC_CPU']:
                if ('cpu') in sync_dictionary:
                    sync_dictionary['cpu'].append(parsed_gpu_event)
                else:
                    sync_dictionary['cpu'] = [parsed_gpu_event]
            if npkit_event_def['id_to_type'][parsed_gpu_event['id']] in ['NPKIT_EVENT_TIME_SYNC_GPU']:
                if ('gpu') in sync_dictionary:
                    sync_dictionary['gpu'].append(parsed_gpu_event)
                else:
                    sync_dictionary['gpu'] = [parsed_gpu_event]
            raw_content_idx += raw_event_size
    return len(sync_dictionary)

def parse_gpu_event_file(rank_cpu_time_xcid, npkit_dump_dir, npkit_event_def, rank, buf_idx, gpu_clock_scale, cpu_clock_scale, dictionary_of_stats, warmup_runs=5):
    gpu_event_file_path = os.path.join(npkit_dump_dir, 'gpu_events_rank_%d_buf_%d' % (rank, buf_idx))
    stats_key = 'gpu_rank_%d' % (rank)
    channel_stats = {}
    raw_event_size = 16
    curr_cpu_base_time = None
    curr_gpu_base_time = None
    gpu_events = []
    event_type_to_seq = {}
    unfiltered_events = []
    start_event_id = 0
    with open(gpu_event_file_path, 'rb') as f:
        raw_content = f.read()
        raw_content_size = len(raw_content)
        raw_content_idx = 0
        if raw_content_size > 0 and stats_key not in dictionary_of_stats:
            dictionary_of_stats[stats_key] = {}
        warmup_raw_content_idx = 0
        while warmup_runs != 0 and warmup_raw_content_idx < raw_content_size: #warmup run cleanup
            parsed_gpu_event = parse_gpu_event(raw_content[warmup_raw_content_idx : warmup_raw_content_idx + raw_event_size])
            unfiltered_events.insert(0, parsed_gpu_event)
            if start_event_id == 0:
                decoded_id = npkit_event_def['id_to_type'][parsed_gpu_event['id']]
                if decoded_id == 'NPKIT_EVENT_TIME_SYNC_CPU' or decoded_id == 'NPKIT_EVENT_TIME_SYNC_GPU':
                    warmup_raw_content_idx += raw_event_size
                    continue
                else:
                    start_event_id = parsed_gpu_event['id']

            warmup_raw_content_idx += raw_event_size
            if parsed_gpu_event['id'] == (start_event_id + 1):
                warmup_runs -= 1
        raw_content_idx = warmup_raw_content_idx
        while raw_content_idx < raw_content_size:
            parsed_gpu_event = parse_gpu_event(raw_content[raw_content_idx : raw_content_idx + raw_event_size])
            unfiltered_events.insert(0, parsed_gpu_event)
            if npkit_event_def['id_to_type'][parsed_gpu_event['id']] == 'NPKIT_EVENT_TIME_SYNC_CPU':
                # curr_cpu_base_time = parsed_gpu_event['timestamp'] / cpu_clock_scale
                curr_cpu_base_time = rank_cpu_time_xcid['cpu'] / cpu_clock_scale
                curr_gpu_base_time = None
            elif npkit_event_def['id_to_type'][parsed_gpu_event['id']] == 'NPKIT_EVENT_TIME_SYNC_GPU':
                if curr_gpu_base_time is None:
                    # curr_gpu_base_time = parsed_gpu_event['timestamp'] / gpu_clock_scale
                    curr_gpu_base_time = rank_cpu_time_xcid['gpu'] / gpu_clock_scale
            else:
                if curr_gpu_base_time is None:
                    curr_gpu_base_time = parsed_gpu_event['timestamp'] / gpu_clock_scale
                event_type = npkit_event_def['id_to_type'][parsed_gpu_event['id']]
                phase = 'B' if event_type.endswith('_ENTRY') else 'E'
                gpu_events.append({
                    'ph': phase,
                    'ts': curr_cpu_base_time + parsed_gpu_event['timestamp'] / gpu_clock_scale - curr_gpu_base_time,
                    'pid': rank,
                    'tid': buf_idx + 1

                })
                if phase == 'B':
                    if event_type not in event_type_to_seq:
                        event_type_to_seq[event_type] = 0
                    gpu_events[-1].update({
                        'name': event_type,
                        'cat': 'GPU',
                        'args': {
                            'rank': rank,
                            'buf_idx': buf_idx,
                            'seq': event_type_to_seq[event_type],
                            'rsvd_0': parsed_gpu_event['rsvd'],
                            'size_0': parsed_gpu_event['size']
                        }
                    })
                    event_type_to_seq[event_type] += 1
                else:
                    gpu_events[-1]['args'] = {'size': parsed_gpu_event['size'], 'rsvd': parsed_gpu_event['rsvd']}
                    current_id = parsed_gpu_event['id']

                    for i in unfiltered_events:
                        if i['id'] == (current_id-1):
                            event_start_ts = curr_cpu_base_time + i['timestamp'] / gpu_clock_scale - curr_gpu_base_time
                            break
                    delta_time = max(0.001, gpu_events[-1]['ts'] - event_start_ts) # delta needs to take the last begin
                    bandwidth = gpu_events[-1]['args']['size'] / delta_time / 1e3
                    if (current_id,parsed_gpu_event['size']) in channel_stats:
                        temp_size = channel_stats[(current_id,parsed_gpu_event['size'])][1]+1
                        temp = channel_stats[(current_id,parsed_gpu_event['size'])][0] * (temp_size - 1 )/ (temp_size)
                        temp_delta = channel_stats[(current_id,parsed_gpu_event['size'])][2] * (temp_size - 1 )/ (temp_size)
                        channel_stats[(current_id,parsed_gpu_event['size'])][0] = bandwidth / (temp_size) + temp
                        channel_stats[(current_id,parsed_gpu_event['size'])][1] = temp_size
                        channel_stats[(current_id,parsed_gpu_event['size'])][2] = delta_time / (temp_size) + temp_delta
                    else:
                        channel_stats[(current_id,parsed_gpu_event['size'])] = [bandwidth, 1, delta_time]
                    gpu_events[-1]['args']['bw (GB/s)'] = bandwidth

            raw_content_idx += raw_event_size

    for key in channel_stats:
        if key in dictionary_of_stats[stats_key]:
            dictionary_of_stats[stats_key][key][0] += channel_stats[key][0]
            old_size = dictionary_of_stats[stats_key][key][1]
            new_size = channel_stats[key][1]
            dictionary_of_stats[stats_key][key][1] += channel_stats[key][1]
            cur_size = dictionary_of_stats[stats_key][key][1]
            new_avg_time = (dictionary_of_stats[stats_key][key][2]*(old_size)+(new_size)*channel_stats[key][2])/(cur_size)
            dictionary_of_stats[stats_key][key][2] = new_avg_time
        else:
            dictionary_of_stats[stats_key][key] = channel_stats[key]
    return gpu_events

def parse_cpu_event_file(npkit_dump_dir, npkit_event_def, rank, channel, cpu_clock_scale):
    cpu_event_file_path = os.path.join(npkit_dump_dir, 'cpu_events_rank_%d_channel_%d' % (rank, channel))
    raw_event_size = 16
    cpu_events = []
    event_type_to_seq = {}

    fiber_is_usable = []
    fiber_open_ts = []
    slot_to_fiber_id = {}
    channel_shift = 1000
    unfiltered_events = []
    start_event_id = 0
    with open(cpu_event_file_path, 'rb') as f:
        raw_content = f.read()
        raw_content_size = len(raw_content)
        raw_content_idx = 0
        parsed_cpu_event = parse_cpu_event(raw_content[raw_content_idx : raw_content_idx + raw_event_size])
        start_event_id = parsed_cpu_event['id'] # start event id

        while raw_content_idx < raw_content_size:
            parsed_cpu_event = parse_cpu_event(raw_content[raw_content_idx : raw_content_idx + raw_event_size])
            event_type = npkit_event_def['id_to_type'][parsed_cpu_event['id']]
            phase = 'B' if event_type.endswith('_ENTRY') else 'E'
            cpu_events.append({
                'ph': phase,
                'ts': parsed_cpu_event['timestamp'] / cpu_clock_scale,
                'pid': rank
            })
            slot = parsed_cpu_event['slot']
            if phase == 'B':
                # Open fiber event
                fiber_id = 0
                while fiber_id < len(fiber_is_usable):
                    if fiber_is_usable[fiber_id]:
                        break
                    fiber_id += 1
                if fiber_id == len(fiber_is_usable):
                    fiber_is_usable.append(True)
                    fiber_open_ts.append(0.0)
                slot_to_fiber_id[slot] = fiber_id
                fiber_open_ts[fiber_id] = cpu_events[-1]['ts']
                fiber_is_usable[fiber_id] = False

                if event_type not in event_type_to_seq:
                    event_type_to_seq[event_type] = 0
                cpu_events[-1].update({
                    'name': event_type,
                    'cat': 'CPU',
                    'args': {
                        'rank': rank,
                        'channel': channel,
                        'slot': parsed_cpu_event['slot'],
                        'seq': event_type_to_seq[event_type],
                        'size_0': parsed_cpu_event['size']
                    }
                })
                event_type_to_seq[event_type] += 1
            else:
                # Close fiber event
                fiber_id = slot_to_fiber_id[slot]
                slot_to_fiber_id.pop(slot)
                last_ts = fiber_open_ts[fiber_id]
                fiber_is_usable[fiber_id] = True

                delta_time = max(0.001, cpu_events[-1]['ts'] - last_ts)
                cpu_events[-1]['args'] = {'size': parsed_cpu_event['size']}
                cpu_events[-1]['args']['bw (GB/s)'] = \
                cpu_events[-1]['args']['size'] / delta_time / 1e3

            cpu_events[-1]['tid'] = fiber_id + (channel + 1) * channel_shift

            raw_content_idx += raw_event_size
    return cpu_events



def convert_npkit_dump_to_trace(npkit_dump_dir, output_dir, npkit_event_def, gpu_statistics, warmup_runs=0):
   
    # check if directory exists, if not create it
    if not os.path.exists(npkit_dump_dir):
        #print(f"Directory {npkit_dump_dir} does not exist. Creating it...")
        os.makedirs(npkit_dump_dir, exist_ok=True)
    else:
        #print(f"Directory {npkit_dump_dir} already exists.")
        pass

    files_in_dump_dir = next(os.walk(npkit_dump_dir))[2]
    gpu_event_files = [x for x in files_in_dump_dir if x.startswith('gpu_events_rank_')]
    cpu_event_files = [x for x in files_in_dump_dir if x.startswith('cpu_events_rank_')]

    ranks = list(set([int(x.split('_rank_')[1].split('_')[0]) for x in gpu_event_files]))
    buf_indices = list(set([int(x.split('_buf_')[1].split('_')[0]) for x in gpu_event_files]))
    channels = list(set([int(x.split('_channel_')[1].split('_')[0]) for x in cpu_event_files]))
    trace = {'traceEvents': []}
    dictionary_of_stats = {}
    for rank in ranks:
        cpu_clock_den_file_path = os.path.join(npkit_dump_dir, 'cpu_clock_period_den_rank_%d' % rank)
        cpu_clock_num_file_path = os.path.join(npkit_dump_dir, 'cpu_clock_period_num_rank_%d' % rank)
        cpu_clock_scale = parse_cpu_clock_scale(cpu_clock_den_file_path, cpu_clock_num_file_path)

        gpu_clock_file_path = os.path.join(npkit_dump_dir, 'gpu_clock_rate_rank_%d' % rank)
        gpu_clock_scale = parse_gpu_clock_scale(gpu_clock_file_path)

        sync_dictionary = {} # per rank
        avg_time = {}
        number_events=0
        for buf_idx in buf_indices: # get the avg time
            parse_gpu_event_file_time(sync_dictionary, npkit_dump_dir, npkit_event_def, rank, buf_idx, gpu_clock_scale, cpu_clock_scale, dictionary_of_stats, warmup_runs)

        for key in sync_dictionary:
            avg_time[key] = 0
            number_events=len(sync_dictionary[key])
            for event in sync_dictionary[key]:
                avg_time[key] = avg_time[key] + (event['timestamp']/number_events)

        for buf_idx in buf_indices:
            gpu_events = parse_gpu_event_file(avg_time, npkit_dump_dir, npkit_event_def, rank, buf_idx, gpu_clock_scale, cpu_clock_scale, dictionary_of_stats, warmup_runs)
            trace['traceEvents'].extend(gpu_events)


        for channel in channels:
            cpu_events = parse_cpu_event_file(npkit_dump_dir, npkit_event_def, rank, channel, cpu_clock_scale)
            trace['traceEvents'].extend(cpu_events)


    trace['traceEvents'].sort(key=lambda x : x['ts'])
    trace['displayTimeUnit'] = 'ns'
    os.makedirs(output_dir, exist_ok=True)
    if gpu_statistics == True:
        with open(os.path.join(output_dir, 'npkit_event_stats.txt'), 'w') as f:
            for key in dictionary_of_stats:
                f.write(key + "\n")
                for event,size in dictionary_of_stats[key]:
                    f.write(npkit_event_def['id_to_type'][event] + "\t"+ "size:" + str(size) + " time: " +str(dictionary_of_stats[key][event,size][2]) + " \t bw: " + str(dictionary_of_stats[key][event,size][0]) + "\n")
    else:
        with open(os.path.join(output_dir, 'npkit_event_trace.json'), 'w') as f:
            json.dump(trace, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npkit_dump_dir', type=str, required=True, help='NPKit dump directory.')
    parser.add_argument('--npkit_event_header_path', type=str, required=True, help='Path to npkit_event.h.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory.')
    parser.add_argument('--gpu_run_stats', type=bool, nargs='?', const=True, default=False, help="print stats instead.")
    parser.add_argument('--warmup_runs', type=int, required=False, default=0, help="amount of warmup_runs on rccl.")
    args = parser.parse_args()
    gpu_statistics = False
    if args.gpu_run_stats is not None:
        gpu_statistics = args.gpu_run_stats
    npkit_event_def = parse_npkit_event_header(args.npkit_event_header_path)
    convert_npkit_dump_to_trace(args.npkit_dump_dir, args.output_dir, npkit_event_def, gpu_statistics, args.warmup_runs)
