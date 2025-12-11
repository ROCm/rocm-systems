# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

# Having run npkit_trace_generator.py, use functions in this script (via import * from script, for example)
# to parse and dump data from the raw json trace files. Always run parse() first with correct json file and desired event name

import csv
import json
import sys
from operator import itemgetter

events = dict()

# collect all occurences of a certain event from trace file
def parse(file, event):
	events.clear()

	# load json as dict
	f = open(file, 'rb')
	raw_content = f.read()
	json_data = json.loads(raw_content.decode('utf-8'))
	trace = json_data['traceEvents']

	B = dict()

	for entry in trace:
		id_pair = (entry['pid'], entry['tid'])

		if entry['ph'] == 'B':
			if id_pair not in B:
				B[id_pair] = []
			B[id_pair].append(entry) #stack from end
		else:
			b = B[id_pair].pop() #pop from end
			if b['name'] == event:
				dur = entry['ts'] - b['ts']
				#adding to results
				if id_pair not in events:
					events[id_pair] = dict()
				events[id_pair][b['ts']] = (dur, entry['args']['bw (GB/s)'], entry['args']['size'], entry['ts'])
				# channel : {start time: [duration, bw, size, end time], ... : ..., ...} 

	return events

def size():
	return len(events)

# return top i longest events within a certain (process,thread) pair, where default is all processes/threads
def longest_events(i, process = None, thread = None):
	if process == None and thread != None:
		raise RuntimeError("makes no sense to compare a thread id across all processes")
	flatten_list = []

	for id_pair in events:
		if process == None or id_pair[0] == process:
			if thread == None or id_pair[1] == thread:
				for ts in events[id_pair]:
					dur = events[id_pair][ts][0]
					flatten_list.append( (id_pair, ts, dur) )

	return sorted(flatten_list,key=itemgetter(2))[-i:]

# calculate total bandwidth of a channel aggregated through all events
def aggregate(channel):
	us = 0
	byte = 0
	timeline = events[channel]
	for i in timeline:
		us += timeline[i][0]
		byte += timeline[i][2]

	return byte/us/1e3

# total throughput of all channels on a gpu (process) in every <interval> us
# tested on proxy channel events (e.g. NPKIT_EVENT_NET_TEST_ENTRY), think twice for other events
def thruput_series(gpu, interval = 100):
	early = sys.maxsize
	late = 0
	a = events

	# determine earliest and latest happening events
	for i in a:
		for j in a[i]:
			if j < early:
				early = j
			if j > late:
				late = j

	# round up for interval length
	late_r = late - (late  % -interval)
	early_r = early - (early % interval)
	early = int(early_r)
	late = int(late_r)

	# aggregate all bytes transferred in a given interval
	series = []
	for ts in range(early,late, interval):
		totalbyte = 0
		for i in a:
			if i[0] == gpu:
				for j in a[i]:
					start = j
					end = a[i][j][3]
					size = a[i][j][2] # total bytes transferred of this event
					duration =  a[i][j][0] # total duration of this event
					if start <= ts and end > ts:
						end = min(end, ts+interval)
						# assume constant bw across time for an event, we only add bytes proportional
						# to this event's presence in this interval over its total duration
						totalbyte += (size / 1e6)  * ( (end-start) / duration)
					elif start < (ts + interval) and end >= ts: #>= for 0 dur case
						start = max(start, ts)
						end = min(end, ts+interval)
						# sometimes there are 0 time events, probably a bug in npkit or trace generation
						if duration == 0:
							assert end-start == 0
							totalbyte += (size / 1e6)
							continue
						totalbyte += (size / 1e6)  * ( (end-start) / duration)
					if totalbyte < 0:
						print(i, j, start, end, ts)
						print(totalbyte, size, ( (end-start) / duration))
						raise RuntimeError("an error with time interval")
			
		series.append(totalbyte * 1000 / interval)
	return series

# export the bw of all events as csv, used for producing heatmap later
# only used and tested for CU level events like NPKIT_EVENT_ALL_REDUCE_RING_ENTRY
def export_csv(name):
	a = events
	matrix = []
	for i in a :
		l = [i[0],i[1]]
		for j in a[i]:
			l.append(a[i][j][1]) #bw
		matrix.append(l)

	file = open(name, 'w')
	csvwriter = csv.writer(file)
	for i in matrix:
		csvwriter.writerow(i)
	file.close()
