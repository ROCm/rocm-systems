"""Representative profiler record shared by report and CI validation tests."""


def make_record(
    rank=0, size=1048576, coll="AllReduce", sn=None, n_ranks=2, exec_us=100.0
):
    return {
        "header": {"rank": rank, "n_ranks": n_ranks},
        "coll_perf": {
            "coll": coll,
            "coll_sn": size + rank if sn is None else sn,
            "coll_msg_size_bytes": size,
            "coll_algo": "Ring",
            "coll_proto": "Simple",
            "coll_n_channels": 4,
            "coll_exec_time_us": exec_us,
            "coll_algobw_gbs": 1.0,
            "coll_busbw_gbs": 1.5,
            "coll_timing_source": "cpu_wallclock",
            "decomposition": {
                "enqueue_to_kernel_us": 1,
                "gpu_kernel_avg_us": 2,
                "gpu_kernel_min_us": 2,
                "gpu_kernel_max_us": 2,
                "proxy_gpu_wait_us": 3,
                "proxy_network_us": 4,
                "proxy_peer_wait_us": 5,
                "proxy_flush_us": 6,
                "proxy_gpu_recv_wait_us": 7,
                "n_proxy_ops": 8,
                "n_send_ops": 4,
                "n_recv_ops": 4,
            },
            "event_trace_ts": {"kernel_events": [{"channel_id": 0, "duration_us": 2}]},
        },
    }
