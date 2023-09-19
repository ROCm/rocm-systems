import os.path
from pathlib import Path
from unittest.mock import patch
import pytest
from importlib.machinery import SourceFileLoader

omniperf = SourceFileLoader("omniperf", "src/omniperf").load_module()
workload_1 = os.path.realpath("workload")
app = ["./sample/vcopy", "1048576", "256"]


def test_path():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "--path",
                workload_1,
                "--",
            ] + app
        ):
            omniperf.main()
            
    #assert successful run
    assert e.value.code == 0

def test_kernel():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "--path",
                workload_1,
                "--kernel",
                "kernel_name"
                "--",
            ] + app
        ):
            omniperf.main()
            
    #assert successful run
    assert e.value.code == 0
    
def test_kernel_summaries():
    with pytest.raises(SystemExit) as e:
        with patch(
            "sys.argv",
            [
                "omniperf",
                "profile",
                "-n",
                "app",
                "--path",
                workload_1,
                "--kernel-summaries",
                "--",
            ] + app
        ):
            omniperf.main()
            
    #assert successful run
    assert e.value.code == 0