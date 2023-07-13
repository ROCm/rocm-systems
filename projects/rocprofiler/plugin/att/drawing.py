#!/usr/bin/env python3
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from copy import deepcopy
import json

COUNTERS_MAX_CAPTURES = 1<<12

class Readable:
    def __init__(self, jsonstring):
        self.jsonstr = json.dumps(jsonstring)
        self.seek = 0

    def read(self, length=0):
        if length<=0:
            return self.jsonstr
        else:
            if self.seek >= len(self):
                self.seek = 0
                return None
            response =  self.jsonstr[self.seek:self.seek+length]
            self.seek += length
            return bytes(response, 'utf-8')

    def __len__(self):
        return len(self.jsonstr)

class FileBytesIO:
    def __init__(self, iobytes):
        self.iobytes = deepcopy(iobytes)
        self.seek = 0

    def __len__(self):
        return self.iobytes.getbuffer().nbytes

    def read(self, length=0):
        if length<=0:
            return bytes(self.iobytes.getbuffer())
        else:
            if self.seek >= self.iobytes.getbuffer().nbytes:
                self.seek = 0
                return None
            response =  self.iobytes.getbuffer()[self.seek:self.seek+length]
            self.seek += length
            return bytes(response)

def get_delta_time(events):
    try:
        CUS = [[e.time for e in events if e.cu==k and e.bank==0] for k in range(16)]
        CUS = [np.asarray(c).astype(np.int64) for c in CUS if len(c) > 2]
        return np.min([np.min(abs(c[1:]-c[:-1])) for c in CUS])
    except:
        return 1

def draw_wave_metrics(selections, normalize, TIMELINES, EVENTS, EVENT_NAMES):
    plt.figure(figsize=(15,4))

    delta_step = 8
    quad_delta_time = max(delta_step,int(0.5+np.min([get_delta_time(events) for events in EVENTS])))
    maxtime = np.max([np.max([e.time for e in events]) for events in EVENTS])/quad_delta_time+1

    if maxtime*delta_step >= COUNTERS_MAX_CAPTURES:
        delta_step = 1
    while maxtime >= COUNTERS_MAX_CAPTURES:
        quad_delta_time *= 2
        maxtime /= 2

    maxtime = int(min(maxtime*delta_step, COUNTERS_MAX_CAPTURES))
    event_timeline = np.zeros((16, maxtime), dtype=np.int32)
    print('Delta:', quad_delta_time)
    print('Max_cycles:', maxtime*quad_delta_time*4//delta_step)

    cycles = 4*quad_delta_time//delta_step*np.arange(maxtime)
    kernel = len(EVENTS)*quad_delta_time

    for events in EVENTS:
        for e in range(len(events)-1):
            bk = events[e].bank*4
            start = events[e].time // (quad_delta_time//delta_step)
            end = start+delta_step
            event_timeline[bk:bk+4, start:end] += np.asarray(events[e].toTuple()[1:5])[:, None]
        start = events[-1].time
        event_timeline[bk:bk+4, start:start+delta_step] += \
            np.asarray(events[-1].toTuple()[1:5])[:, None]

    event_timeline = [np.convolve(e, [kernel for k in range(3)])[1:-1] for e in event_timeline]
    #event_timeline = [e/kernel for e in event_timeline]

    if normalize:
        event_timeline = [100*e/max(e.max(), 1E-5) for e in event_timeline]

    colors = ['blue', 'green', 'gray', 'red', 'orange', 'cyan', 'black', 'darkviolet',
                'yellow', 'darkred', 'pink', 'lime', 'gold', 'tan', 'aqua', 'olive']
    [plt.plot(cycles, e, '-', label=n, color=c)
        for e, n, c, sel in zip(event_timeline, EVENT_NAMES, colors, selections) if sel]

    plt.legend()
    if normalize:
        plt.ylabel('As % of maximum')
    else:
        plt.ylabel('Value')
    plt.xlabel('Cycle')
    plt.subplots_adjust(left=0.04, right=1, top=1, bottom=0.1)

    figure_bytes = BytesIO()
    plt.savefig(figure_bytes, dpi=150)
    return EVENT_NAMES, FileBytesIO(figure_bytes)


def draw_wave_states(selections, normalize, TIMELINES):
    plot_indices = [1, 2, 3, 4]
    STATES = [['Empty', 'Idle', 'Exec', 'Wait', 'Stall'][k] for k in plot_indices]
    colors = [['gray', 'orange', 'green', 'red', 'blue'][k] for k in plot_indices]

    plt.figure(figsize=(15,4))

    maxtime = max([np.max((TIMELINES[k]!=0)*np.arange(0,TIMELINES[k].size)) for k in plot_indices])
    maxtime = max(maxtime, 1)
    timelines = [deepcopy(TIMELINES[k][:maxtime]) for k in plot_indices]
    timelines = [np.pad(t, [0, maxtime-t.size]) for t in timelines]

    if normalize:
        timelines = np.array(timelines) / np.maximum(np.sum(timelines,0)*1E-2,1E-7)

    trim = max(maxtime//5000,1)
    cycles = np.arange(0, timelines[0].size//trim, 1)*trim
    timelines = [time[:trim*(time.size//trim)].reshape((-1, trim)).mean(-1) if len(time) > 0 else cycles*0 for time in timelines]
    kernsize = 21
    kernel = np.asarray([np.exp(-abs(10*k/kernsize)) for k in range(-kernsize//2,kernsize//2+1)])
    kernel /= np.sum(kernel)

    timelines = [np.convolve(time, kernel)[kernsize//2:-kernsize//2] for time in timelines if len(time) > 0]

    [plt.plot(cycles, t, label='State '+s, linewidth=1.1, color=c)
        for t, s, c, sel in zip(timelines, STATES, colors, selections) if sel]

    plt.legend()
    if normalize:
        plt.ylabel('Waves state %')
    else:
        plt.ylabel('Waves state total')
    plt.xlabel('Cycle')
    plt.ylim(-1)
    plt.xlim(-maxtime//200, maxtime+maxtime//200+1)
    plt.subplots_adjust(left=0.04, right=1, top=1, bottom=0.1)
    figure_bytes = BytesIO()
    plt.savefig(figure_bytes, dpi=150)
    return STATES, FileBytesIO(figure_bytes)


def draw_occupancy(selections, normalize, OCCUPANCY, shadernames):
    plt.figure(figsize=(15,4))
    names = []
    if len(OCCUPANCY) == 1: # If single SE, do occupancy per CU/WGP
        OCCUPANCY = [[u for u in OCCUPANCY[0] if u&0xFF==k] for k in range(16)]
        shadernames = ['CU'+str(k) for k in range(16) if len(OCCUPANCY[k]) > 0]
        OCCUPANCY = [occ for occ in OCCUPANCY if len(occ) > 0]

    maxtime = 1
    delta = 1
    for name, occ in zip(shadernames, OCCUPANCY):
        occ_values = [0]
        occ_times = [0]
        occ = [(int(u>>16), (u>>8)&0xFF, u&0xFF) for u in occ]
        current_occ = [0 for k in range(16)]

        for time, value, cu in occ:
            occ_times.append(time)
            occ_values.append(occ_values[-1] + value - current_occ[cu])
            current_occ[cu] = value
        try:
            names.append('SE'+name.split('.att')[0].split('_se')[-1])
        except:
            names.append(name)

        NUM_DOTS = 1500
        maxtime = np.max(occ_times)
        delta = max(1, maxtime//NUM_DOTS)
        chart = np.zeros((maxtime//delta+1), dtype=np.float32)
        norm_fact = np.zeros_like(chart)

        for i, t in enumerate(occ_times[:-1]):
            b = t//delta
            e = max(b+1,occ_times[i+1]//delta)
            chart[b:e] += occ_values[i]
            norm_fact[b:e] += 1

        chart /= np.maximum(norm_fact,1)
        if normalize:
            chart /= max(chart.max(),1E-6)

        plt.plot(np.arange(chart.size)*delta, chart, label=name, linewidth=1.1)

    plt.legend()
    if normalize:
        plt.ylabel('Occupancy %')
    else:
        plt.ylabel('Occupancy total')
    plt.xlabel('Cycle')
    plt.ylim(-1)
    plt.xlim(-maxtime//200, maxtime+maxtime//200+delta+1)
    plt.subplots_adjust(left=0.04, right=1, top=1, bottom=0.1)
    figure_bytes = BytesIO()
    plt.savefig(figure_bytes, dpi=150)
    return names, FileBytesIO(figure_bytes)


def GeneratePIC(drawinfo, selections=[True for k in range(16)], normalize=False):
    EVENTS = drawinfo['EVENTS']

    response = {}
    figures = {}

    states, figure = draw_occupancy(selections, normalize, drawinfo['OCCUPANCY'], drawinfo['ShaderNames'])
    response['occupancy.png'] = states
    figures['occupancy.png'] = figure

    states, figure = draw_wave_states(selections, normalize, drawinfo['TIMELINES'])
    response['timeline.png'] = states
    figures['timeline.png'] = figure

    if len(EVENTS) > 0 and np.sum([len(e) for e in EVENTS]) > 32:
        EVENT_NAMES, figure = draw_wave_metrics(selections, normalize, drawinfo['TIMELINES'], EVENTS, drawinfo['EVENT_NAMES'])
        response['counters.png'] = EVENT_NAMES
        figures['counters.png'] = figure

    return Readable(response), figures
