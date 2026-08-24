# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

import os
import subprocess
import pytest


@pytest.mark.ext_tuner
@pytest.mark.allreduce
def test_model_demo_tuner_runs(paths):
    """The model_demo tuner loads and tunes without taking the job down.

    The plugin registers as a v4 tuner, so getCollInfo must accept the cost table. A
    signature from an older API makes RCCL pass the algorithm count where the plugin
    expects a pointer, and the rank dies writing through it on the first collective;
    a clean exit is the regression check here.
    """

    plugin_so = os.path.join(paths.RCCL_INSTALL_DIR, "plugins", "tuner", "model_demo", "librccl-tuner.so")
    if not os.path.exists(plugin_so):
        pytest.skip(f"model_demo tuner is not built at {plugin_so}")

    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_TUNER_PLUGIN": plugin_so,
        # DDA returns before the tuner is consulted, and its threshold covers this whole
        # size range on gfx942, so leaving it on means getCollInfo is never called and the
        # ABI under test is never exercised.
        "RCCL_DDA_ENABLE": "0",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,TUNING",
    })

    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "8",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        f"{paths.RCCL_TESTS_DIR}/build/all_reduce_perf",
        "-b", "1M",
        "-e", "8M",
        "-f", "2",
        "-g", "1",
    ]

    log_dir = os.path.join(paths.LOGDIR, "model_demo_ext_tuner_test_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "model_demo.log")
    with open(log_file, "w") as logfile:
        result = subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

    assert result.returncode == 0, \
        f"Run with the model_demo tuner should exit cleanly, got {result.returncode}, see {log_file}"

    with open(log_file, 'r') as f:
        log_content = f.read()

    for crash in ("Segmentation fault", "signal 11", "core dumped", "Address not mapped"):
        assert crash not in log_content, \
            f"Run with the model_demo tuner reported '{crash}', see {log_file}"

    # This line is written once the v4 symbol has been resolved and bound, so unlike the
    # environment-variable message it proves the plugin loaded at the ABI under test.
    assert "TUNER/Plugin: Using Example (v4)" in log_content, \
        f"model_demo should have loaded and bound as a v4 tuner, see {log_file}"

    # A tuned run still has to produce results, so a plugin that silently disables
    # tuning does not pass as success.
    assert "Avg bus bandwidth" in log_content, \
        f"AllReduce should have completed and reported bandwidth, see {log_file}"
