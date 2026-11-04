#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Measure HSA registry query latency while CPU copy or fill owns the lock."""

import argparse
import ctypes
import statistics
import sys
import threading
import time

SUCCESS = 0
SYSTEM_INFO_VERSION_MAJOR = 0
AGENT_INFO_DEVICE = 17
DEVICE_TYPE_CPU = 0
POOL_INFO_ALLOC_ALLOWED = 5
AGENT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint64, ctypes.c_void_p)
POOL_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint64, ctypes.c_void_p)


def status(call, *args):
    result = call(*args)
    if result != SUCCESS:
        raise RuntimeError(f"{call.__name__} returned 0x{result:x}")


def configure(library):
    library.hsa_init.restype = ctypes.c_uint32
    library.hsa_shut_down.restype = ctypes.c_uint32
    library.hsa_system_get_info.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    library.hsa_system_get_info.restype = ctypes.c_uint32
    library.hsa_iterate_agents.argtypes = [AGENT_CALLBACK, ctypes.c_void_p]
    library.hsa_iterate_agents.restype = ctypes.c_uint32
    library.hsa_agent_get_info.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p]
    library.hsa_agent_get_info.restype = ctypes.c_uint32
    library.hsa_amd_agent_iterate_memory_pools.argtypes = [
        ctypes.c_uint64, POOL_CALLBACK, ctypes.c_void_p
    ]
    library.hsa_amd_agent_iterate_memory_pools.restype = ctypes.c_uint32
    library.hsa_amd_memory_pool_get_info.argtypes = [
        ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p
    ]
    library.hsa_amd_memory_pool_get_info.restype = ctypes.c_uint32
    library.hsa_amd_memory_pool_allocate.argtypes = [
        ctypes.c_uint64, ctypes.c_size_t, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.hsa_amd_memory_pool_allocate.restype = ctypes.c_uint32
    library.hsa_amd_memory_pool_free.argtypes = [ctypes.c_void_p]
    library.hsa_amd_memory_pool_free.restype = ctypes.c_uint32
    library.hsa_memory_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.hsa_memory_copy.restype = ctypes.c_uint32
    library.hsa_amd_memory_fill.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t
    ]
    library.hsa_amd_memory_fill.restype = ctypes.c_uint32


def probe(library):
    major = ctypes.c_uint16()
    started = time.perf_counter_ns()
    status(library.hsa_system_get_info, SYSTEM_INFO_VERSION_MAJOR, ctypes.byref(major))
    assert major.value == 1
    return time.perf_counter_ns() - started


def host_pool(library, size):
    agents = []
    agent_callback = AGENT_CALLBACK(lambda agent, _: agents.append(agent) or SUCCESS)
    status(library.hsa_iterate_agents, agent_callback, None)
    for agent in agents:
        device_type = ctypes.c_uint32()
        status(library.hsa_agent_get_info, agent, AGENT_INFO_DEVICE,
               ctypes.byref(device_type))
        if device_type.value != DEVICE_TYPE_CPU:
            continue
        pools = []
        pool_callback = POOL_CALLBACK(lambda pool, _: pools.append(pool) or SUCCESS)
        status(library.hsa_amd_agent_iterate_memory_pools, agent, pool_callback, None)
        for pool in pools:
            allowed = ctypes.c_bool()
            status(library.hsa_amd_memory_pool_get_info, pool, POOL_INFO_ALLOC_ALLOWED,
                   ctypes.byref(allowed))
            if not allowed.value:
                continue
            pointer = ctypes.c_void_p()
            if library.hsa_amd_memory_pool_allocate(pool, size, 0,
                                                    ctypes.byref(pointer)) == SUCCESS:
                return pointer
    raise RuntimeError("no CPU pool could allocate the fill buffer")


def summary(name, phase, samples, operations, size, elapsed_ns):
    ordered = sorted(samples)
    def percentile(fraction):
        return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))] / 1000
    print(f"{name},{phase},{operations},{size},{len(samples)},"
          f"{statistics.median(ordered) / 1000:.2f},{percentile(0.95):.2f},"
          f"{percentile(0.99):.2f},{ordered[-1] / 1000:.2f},"
          f"{elapsed_ns / 1e6:.2f}")


def run_trial(library, name, operation, operations, size, baseline_probes):
    baseline_start = time.perf_counter_ns()
    baseline = [probe(library) for _ in range(baseline_probes)]
    summary(name, "baseline", baseline, 0, size,
            time.perf_counter_ns() - baseline_start)

    start = threading.Event()
    done = threading.Event()
    errors = []
    def work():
        start.wait()
        try:
            for _ in range(operations):
                operation()
        except Exception as error:  # Keep the main thread able to join and report.
            errors.append(error)
        finally:
            done.set()

    worker = threading.Thread(target=work, name=f"hsa-{name}")
    worker.start()
    samples = []
    started = time.perf_counter_ns()
    start.set()
    try:
        while not done.is_set():
            samples.append(probe(library))
    finally:
        worker.join()
    if errors:
        raise errors[0]
    if not samples:
        raise RuntimeError("worker finished before a concurrent query was sampled")
    summary(name, "concurrent", samples, operations, size,
            time.perf_counter_ns() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="target/debug/librocddi_runtime.so")
    parser.add_argument("--mib", type=int, default=64)
    parser.add_argument("--operations", type=int, default=64)
    parser.add_argument("--baseline-probes", type=int, default=1000)
    args = parser.parse_args()
    if args.mib <= 0 or args.operations <= 0 or args.baseline_probes <= 0:
        parser.error("--mib, --operations, and --baseline-probes must be positive")
    size = args.mib * 1024 * 1024
    library = ctypes.CDLL(args.library)
    configure(library)
    result = library.hsa_init()
    if result != SUCCESS:
        print(f"SKIP: hsa_init returned 0x{result:x}; use a qualified GPU host", file=sys.stderr)
        return 77
    fill_pointer = None
    try:
        source = ctypes.create_string_buffer(size)
        destination = ctypes.create_string_buffer(size)
        source_address = ctypes.addressof(source)
        destination_address = ctypes.addressof(destination)
        print("scenario,phase,operations,bytes,query_samples,query_p50_us,"
              "query_p95_us,query_p99_us,query_max_us,elapsed_ms")
        run_trial(library, "copy", lambda: status(library.hsa_memory_copy,
                  destination_address, source_address, size),
                  args.operations, size, args.baseline_probes)
        fill_pointer = host_pool(library, size)
        run_trial(library, "fill", lambda: status(library.hsa_amd_memory_fill,
                  fill_pointer, 0x5a5a5a5a, size // 4),
                  args.operations, size, args.baseline_probes)
    finally:
        if fill_pointer is not None:
            status(library.hsa_amd_memory_pool_free, fill_pointer)
        status(library.hsa_shut_down)
    return 0


if __name__ == "__main__":
    sys.exit(main())
