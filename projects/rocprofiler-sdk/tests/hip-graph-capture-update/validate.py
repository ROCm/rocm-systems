import pytest

LAUNCH_APIS = {"hipLaunchKernel", "hipExtLaunchKernel", "hipLaunchKernelExC"}


def capture_calls(data):
    records = data["buffer_records"]
    kinds = next(k for k in records["names"] if k["kind"] == "HIP_RUNTIME_API")
    ops = {i for i, name in enumerate(kinds["operations"]) if name in LAUNCH_APIS}
    return sorted(
        {
            r["correlation_id"]["internal"]
            for r in records["hip_api_traces"]
            if r["operation"] in ops
        }
    )


@pytest.mark.parametrize(
    "mode", range(3), ids=["set-params", "disable-node", "exec-update"]
)
def test_mutation_invalidates_capture_plan(input_data, mode):
    data = input_data["rocprofiler-sdk-json-tool"]
    calls = capture_calls(data)
    assert len(calls) == 6
    by_exec = {}
    for r in data["buffer_records"]["kernel_dispatch"]:
        info = r["dispatch_info"]
        eid = info["graph_exec_id"]["handle"]
        if eid:
            by_exec.setdefault(eid, {}).setdefault(
                r["correlation_id"]["internal"], []
            ).append(info)
    execs = sorted(by_exec)
    assert len(execs) == 6
    original, fresh = [by_exec[e] for e in execs[2 * mode : 2 * mode + 2]]
    assert len(original) == 9 and len(fresh) == 3
    expected = calls[2 * mode : 2 * mode + 2]
    for j, cid in enumerate(sorted(original)):
        infos = sorted(original[cid], key=lambda r: r["dispatch_id"])
        got = [r["capture_correlation_id"] for r in infos]
        if j < 6:
            assert got == expected, (mode, j, got, expected)
        else:
            assert len(got) == (1 if mode == 1 else 2)
            assert not any(got), f"mode {mode}: stale capture plan after mutation: {got}"
    for infos in fresh.values():
        got = [
            r["capture_correlation_id"]
            for r in sorted(infos, key=lambda r: r["dispatch_id"])
        ]
        assert got == expected, (mode, "fresh exec", got, expected)
