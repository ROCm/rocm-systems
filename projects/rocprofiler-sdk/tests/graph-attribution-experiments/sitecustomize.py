import os
import threading
import time
from pathlib import Path

def setup():
    root = Path(os.environ["GRAPH_TRACE_DIR"])
    pid = os.getpid()
    os.environ["ROCPROFILER_TOOL_OUTPUT_FILE"] = str(root / f"trace-{pid}.json")
    import torch
    import ctypes
    tool = ctypes.CDLL(str(Path(os.environ["SDK_BUILD"]) / "lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so"))
    tool.json_tool_pause_until_graph.restype = None
    tool.json_tool_pause_until_graph()
    original_capture_begin = torch.cuda.CUDAGraph.capture_begin
    started = False
    def capture_begin(self, *args, **kwargs):
        nonlocal started
        if not started:
            assert tool.json_tool_start() == 0
            started = True
        return original_capture_begin(self, *args, **kwargs)
    torch.cuda.CUDAGraph.capture_begin = capture_begin
    def finish():
        while not (root / "finalize").exists():
            time.sleep(1)
        import ctypes
        tool = ctypes.CDLL(str(Path(os.environ["SDK_BUILD"]) / "lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so"))
        tool.json_tool_finalize_idle_worker.restype = None
        tool.json_tool_finalize_idle_worker()
        (root / f"finalized-{pid}").write_text("done\n")
    threading.Thread(target=finish, daemon=True).start()
if os.environ.get("GRAPH_TRACE_ENABLE") == "1":
    setup()
