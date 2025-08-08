#!/usr/bin/env python3
import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from copy import deepcopy
import json

COUNTERS_MAX_CAPTURES = 1 << 12


class Readable:
    def __init__(self, jsonstring):
        self.jsonstr = json.dumps(jsonstring)
        self.seek = 0

    def read(self, length=0):
        if length <= 0:
            return self.jsonstr
        else:
            if self.seek >= len(self):
                self.seek = 0
                return None
            response = self.jsonstr[self.seek : self.seek + length]
            self.seek += length
            return bytes(response, "utf-8")

    def __len__(self):
        return len(self.jsonstr)


class FileBytesIO:
    def __init__(self, iobytes):
        self.iobytes = deepcopy(iobytes)
        self.seek = 0

    def __len__(self):
        return self.iobytes.getbuffer().nbytes

    def read(self, length=0):
        if length <= 0:
            return bytes(self.iobytes.getbuffer())
        else:
            if self.seek >= self.iobytes.getbuffer().nbytes:
                self.seek = 0
                return None
            response = self.iobytes.getbuffer()[self.seek : self.seek + length]
            self.seek += length
            return bytes(response)


def get_delta_time(events):
    try:
        CUS = [[e.time for e in events if e.cu == k and e.bank == 0] for k in range(16)]
        CUS = [np.asarray(c).astype(np.int64) for c in CUS if len(c) > 2]
        return np.min([np.min(abs(c[1:] - c[:-1])) for c in CUS])
    except:
        return 1


def draw_wave_metrics(selections, normalize, TIMELINES, EVENTS, EVENT_NAMES):
    plt.figure(figsize=(15, 4))

    delta_step = 8
    quad_delta_time = max(
        delta_step, int(0.5 + np.min([get_delta_time(events) for events in EVENTS]))
    )
    maxtime = (
        np.max([np.max([e.time for e in events]) for events in EVENTS]) / quad_delta_time
        + 1
    )

    if maxtime * delta_step >= COUNTERS_MAX_CAPTURES:
        delta_step = 1
    while maxtime >= COUNTERS_MAX_CAPTURES:
        quad_delta_time *= 2
        maxtime /= 2

    maxtime = int(min(maxtime * delta_step, COUNTERS_MAX_CAPTURES))
    event_timeline = np.zeros((16, maxtime), dtype=np.int32)
    print("Delta:", quad_delta_time)
    print("Max_cycles:", maxtime * quad_delta_time * 4 // delta_step)

    cycles = 4 * quad_delta_time // delta_step * np.arange(maxtime)
    kernel = len(EVENTS) * quad_delta_time

    for events in EVENTS:
        for e in range(len(events) - 1):
            bk = events[e].bank * 4
            start = events[e].time // (quad_delta_time // delta_step)
            end = start + delta_step
            event_timeline[bk : bk + 4, start:end] += np.asarray(
                events[e].toTuple()[1:5]
            )[:, None]
        start = events[-1].time
        event_timeline[bk : bk + 4, start : start + delta_step] += np.asarray(
            events[-1].toTuple()[1:5]
        )[:, None]

    event_timeline = [
        np.convolve(e, [kernel for k in range(3)])[1:-1] for e in event_timeline
    ]
    # event_timeline = [e/kernel for e in event_timeline]

    if normalize:
        event_timeline = [100 * e / max(e.max(), 1e-5) for e in event_timeline]

    colors = [
        "blue",
        "green",
        "gray",
        "red",
        "orange",
        "cyan",
        "black",
        "darkviolet",
        "yellow",
        "darkred",
        "pink",
        "lime",
        "gold",
        "tan",
        "aqua",
        "olive",
    ]
    [
        plt.plot(cycles, e, "-", label=n, color=c)
        for e, n, c, sel in zip(event_timeline, EVENT_NAMES, colors, selections)
        if sel
    ]

    plt.legend()
    if normalize:
        plt.ylabel("As % of maximum")
    else:
        plt.ylabel("Value")
    plt.xlabel("Cycle")
    plt.subplots_adjust(left=0.04, right=1, top=1, bottom=0.1)

    figure_bytes = BytesIO()
    plt.savefig(figure_bytes, dpi=150)
    return EVENT_NAMES, FileBytesIO(figure_bytes)


def draw_wave_states(selections, normalize, TIMELINES):
    plot_indices = [1, 2, 3, 4]
    STATES = [["Empty", "Idle", "Exec", "Wait", "Stall"][k] for k in plot_indices]
    colors = [["gray", "orange", "green", "red", "blue"][k] for k in plot_indices]

    plt.figure(figsize=(15, 4))


    maxtime = max([np.max((TIMELINES[k]!=0)*np.arange(0,TIMELINES[k].size)) for k in plot_indices])
    maxtime = max(maxtime, 1)
    timelines = [deepcopy(TIMELINES[k][:maxtime]) for k in plot_indices]
    timelines = [np.pad(t, [0, maxtime - t.size]) for t in timelines]

    if normalize:
        timelines = np.array(timelines) / np.maximum(np.sum(timelines, 0) * 1e-2, 1e-7)

    trim = max(maxtime // 5000, 1)
    cycles = np.arange(0, timelines[0].size // trim, 1) * trim
    timelines = [
        time[: trim * (time.size // trim)].reshape((-1, trim)).mean(-1)
        if len(time) > 0
        else cycles * 0
        for time in timelines
    ]
    kernsize = 15
    kernel = np.asarray([
        np.exp(-abs(10 * k / kernsize)) for k in range(-kernsize // 2, kernsize // 2 + 1)
    ])
    kernel /= np.sum(kernel)

    timelines = [
        np.convolve(time, kernel)[kernsize // 2 : -kernsize // 2]
        for time in timelines if len(time) > 0
    ]
    maxtime *= 16
    cycles *= 16
    [
        plt.plot(cycles, t, label="State " + s, linewidth=1.1, color=c)
        for t, s, c, sel in zip(timelines, STATES, colors, selections)
        if sel
    ]

    plt.legend()
    if normalize:
        plt.ylabel("Waves state %")
    else:
        plt.ylabel("Waves state total")
    plt.xlabel("Cycle")
    plt.ylim(-1)
    plt.xlim(-maxtime // 200, maxtime + maxtime // 200 + 1)
    plt.subplots_adjust(left=0.04, right=1, top=1, bottom=0.1)
    figure_bytes = BytesIO()
    plt.savefig(figure_bytes, dpi=150)
    return STATES, FileBytesIO(figure_bytes)


def draw_occupancy_per_dispatch(selections, normalize, OCCUPANCY, dispatchnames):
    plt.figure(figsize=(15, 4))
    maxtime = 1
    delta = 1

    for k in range(len(OCCUPANCY)):
        if len(OCCUPANCY[k]) <= 1:
            continue
        for ev in OCCUPANCY[k]:
            maxtime = max(maxtime, ev[0])

    NUM_DOTS = 1600 # number of points taken for graphing
    delta = max(1, maxtime // NUM_DOTS) # Spacing between data points. Waves will be averaged over this interval.
    # Holds occupancy data
    chart = np.zeros((len(dispatchnames), maxtime // delta + 2), dtype=np.float32)

    for occ in OCCUPANCY:
        if len(occ) <= 1:
            continue
        # Number of waves multiplied by number of events
        small_chart = np.zeros_like(chart)
        # Holds number of events in that time period, for averaging.
        norm_fact = np.zeros_like(chart)
        norm_fact += 1E-5

        # Holds last known state per dispatch
        current_time = [0 for k in range(len(dispatchnames))]
        # Holds occupancy per Dispatch
        total_value = [0 for k in range(len(dispatchnames))]

        for time, en, kid in occ:
            b = current_time[kid]
            e = max(b + 1, time // delta)
            small_chart[kid][b:e] += total_value[kid]
            norm_fact[kid][b:e] += 1

            # Enable = 1 means a new wave started, enable = 0 means a wave has ended on that kernel ID.
            total_value[kid] += 2*en - 1
            current_time[kid] = time // delta
        for small, norm, time, value in zip(small_chart, norm_fact, current_time, total_value):
            small[time] += value
            norm[time] += value

        chart += small_chart/norm_fact # small_chart / norm_fact is the mean number of waves a tthat time point

    for (id, name), occ in zip(dispatchnames.items(), chart):
        plt.plot(np.arange(occ.size) * delta * 8, occ, label=str(id)+'#'+name, linewidth=1.1)

    plt.legend()
    if normalize:
        plt.ylabel("Occupancy %")
    else:
        plt.ylabel("Occupancy total")
    plt.xlabel("Cycle")
    plt.ylim(-1)
    plt.xlim(-maxtime // 200, maxtime + maxtime // 200 + delta + 1)
    plt.subplots_adjust(left=0.04, right=1, top=1, bottom=0.1)
    figure_bytes = BytesIO()
    plt.savefig(figure_bytes, dpi=150)
    return dispatchnames, FileBytesIO(figure_bytes)


def draw_occupancy(selections, normalize, OCCUPANCY, shadernames, numdispatchid):
    plt.figure(figsize=(15, 4))
    names = []

    g_maxtime = 1
    g_delta = 1
    for name, occ in zip(shadernames, OCCUPANCY):
        if len(occ) <= 1:
            continue

        occ_values = [0]
        occ_times = [0]

        for time, en, _ in occ:
            occ_times.append(time)
            occ_values.append(occ_values[-1] + 2*en - 1) # If enable = 1, increment. Else, decrement occupancy.

        try:
            names.append('SE'+name.split('_se')[1].split('.att')[0])
        except:
            names.append(name)

        NUM_DOTS = 1500 # Number of points taken for graphing
        maxtime = occ_times[-1]+1
        delta = max(1, maxtime // NUM_DOTS)
        g_maxtime = max(g_maxtime, maxtime)
        g_delta = max(g_delta, delta)
        chart = np.zeros((maxtime // delta + 1), dtype=np.float32)
        norm_fact = np.zeros_like(chart)
        norm_fact += 1E-6

        for i in range(len(occ_times)-1):
            b = occ_times[i] // delta
            e = max(b + 1, occ_times[i + 1] // delta)
            chart[b:e] += occ_values[i]
            norm_fact[b:e] += 1

        chart /= norm_fact
        if normalize:
            chart /= max(chart.max(), 1e-6)

        plt.plot(np.arange(chart.size) * delta, chart, label=names[-1], linewidth=1.1)

    plt.legend()
    if normalize:
        plt.ylabel("Occupancy %")
    else:
        plt.ylabel("Occupancy total")
    plt.xlabel("Cycle")
    plt.ylim(-1)
    plt.xlim(-g_maxtime // 200, g_maxtime + g_maxtime // 200 + g_delta + 1)
    plt.subplots_adjust(left=0.04, right=1, top=1, bottom=0.1)
    figure_bytes = BytesIO()
    plt.savefig(figure_bytes, dpi=150)
    return names, FileBytesIO(figure_bytes)


def getocc(u):
    # Parser struct occupancy_info_t
    # Bits 23:63 Time= Time divided by 8
    # Bit 18 = Enable (Wave start if 1, Wave end if 0)
    # Bits 0:11 is the kernel ID running on that wave
    return 8*int(u>>23), (u>>18) & 1, u&0xFFF


def GeneratePIC(drawinfo, selections=[True for k in range(16)], normalize=False):
    EVENTS = drawinfo["EVENTS"]

    response = {}
    figures = {}

    OCCUPANCY = drawinfo["OCCUPANCY"]
    # Transforms returned data into a array of events with each event being a tuple (time, enable, kernel ID)
    OCCUPANCY = [[getocc(u) for u in OCCUPANCY[k]] for k in range(len(OCCUPANCY))]

    states, figure = draw_occupancy(selections, normalize, OCCUPANCY, drawinfo["ShaderNames"], len(drawinfo["DispatchNames"]))
    response["occupancy.png"] = states
    figures["occupancy.png"] = figure

    states, figure = draw_occupancy_per_dispatch(selections, normalize, OCCUPANCY, drawinfo["DispatchNames"])
    response["dispatches.png"] = states
    figures["dispatches.png"] = figure

    states, figure = draw_wave_states(selections, normalize, drawinfo["TIMELINES"])
    response["timeline.png"] = states
    figures["timeline.png"] = figure

    if len(EVENTS) > 0 and np.sum([len(e) for e in EVENTS]) > 32:
        EVENT_NAMES, figure = draw_wave_metrics(
            selections, normalize, drawinfo["TIMELINES"], EVENTS, drawinfo["EVENT_NAMES"]
        )
        response["counters.png"] = EVENT_NAMES
        figures["counters.png"] = figure

    return Readable(response), figures
