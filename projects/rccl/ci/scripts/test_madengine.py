#!/usr/bin/env python3
"""Run MADEngine AI workloads against CI-built RCCL and track performance.

This script handles:
  1. Installing madengine from source into the CI Python environment
  2. Building a Docker overlay image with the CI-built RCCL
  3. Generating a manifest.json per A/B phase
  4. Running both phases back to back inside one SLURM allocation: the stock
     base image (baseline) and the same image with the CI RCCL laid over it
     (candidate)
  5. Scoring the candidate against the baseline measured on the same nodes
  6. Appending both absolute values and their ratio to a JSONL datastore

Usage from GitHub Actions (on ruby-linux-slurm-scale-runner):
  python3 rocm-systems/projects/rccl/ci/scripts/test_madengine.py \
      --artifact-dir ./build \
      --candidate-sha "$CANDIDATE_SHA" \
      --artifact-provenance "$WORK_DIR/candidate-artifact-run.json" \
      --candidate-tuner-config rocm-systems/projects/rccl/tuner/rccl_tuner_gfx950.csv \
      --workload llama-3.1-70b-training \
      --cluster ruby \
      --nodes 2 \
      --results-dir /apps/rccl-ci/madengine/perf \
      --work-dir /apps/rccl-ci/madengine/workdir/${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import filecmp
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from rccl_ci_utils import (
    find_rccl_library,
    send_email_report,
    send_teams_webhook,
    set_github_output,
    write_github_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PERF_DATASTORE = "madengine_results.jsonl"

# madengine's perf_entry_super.json uses short metric names that differ
# from our canonical workload-config keys.  Map each config key to the
# set of madengine metric names we accept as a match.
#
# Score the ``_avg`` rows (Megatron's running mean over measured iterations),
# matching parse_live_log_metrics(); the plain rows are one noisy sample.
_METRIC_ALIASES: dict[str, set[str]] = {
    "tokens_per_second_per_gpu": {"tok_per_s_per_gpu_avg"},
}
_TFLOPS_METRICS = {"TFLOPS_per_gpu_avg"}

# Applied to the candidate/baseline ratio of one same-node A/B pair.
REGRESSION_THRESHOLD_TRAINING = 0.02  # 2%
REGRESSION_THRESHOLD_INFERENCE = 0.05  # 5%

BASELINE = "baseline"
CANDIDATE = "candidate"

# Its own directory because it is also the nodes' build context: WORK_DIR has
# live sbatch logs, and tarring a file that is still growing fails the build.
OVERLAY_CTX = "overlay_ctx"
OVERLAY_DOCKERFILE = "Dockerfile.rccl-overlay"
# The maxtext family needs a second image: its A/B baseline cannot be the stock
# registry tag, which carries no Primus. See stage_maxtext_baseline().
MAXTEXT_BASELINE_DOCKERFILE = "Dockerfile.maxtext-baseline"
# Likewise for sglang-disagg, whose baseline is MAD's Dockerfile.
SGLANG_BASELINE_DOCKERFILE = "Dockerfile.sglang-baseline"
SGLANG_MAD_DOCKERFILE = "docker/sglang_disagg_inference_full_overlay.ubuntu.amd.Dockerfile"
# Families whose baseline is built here rather than pulled as the stock tag.
LOCAL_BASELINE_DOCKERFILES = {
    "maxtext": MAXTEXT_BASELINE_DOCKERFILE,
    "sglang-disagg": SGLANG_BASELINE_DOCKERFILE,
}

# Must not predate ROCm/madengine#213: the A/B runs madengine inside salloc.
MADENGINE_REPO = "https://github.com/ROCm/madengine.git"
MADENGINE_REF = "00ee3e80c0a308599633caf74e5d91a72164a2fe"  # develop, has #190 and #213
MAD_REPO = "https://github.com/ROCm/MAD.git"
MAD_REF = "36339bd825f9c3772e35a54adfeba9635d6765da"  # mad-rccl, has MAD#279, #280 and #284
MAD_BRANCH = "mad-rccl"

# Primus is not vendored in MAD; tools/fetch_primus.sh checks it out into
# scripts/Primus, and the MaxText images COPY that tree in. Pinned rather than
# left to the script's default so a default change cannot move the image under
# the benchmark.
#
# v26.7.0 and not jax-maxtext-v26.8: both ship primus-cli and the MI355X 405B
# config, but AMD-AGI/Primus#999 retired examples/run_pretrain.sh and v26.8 no
# longer carries it.  We gate on primus-cli, which is in both (and ROCm/MAD#280
# moved MAD's own overlay to the same check), so this pin is the only thing
# tying us to v26.7.0.
PRIMUS_URL = "https://github.com/AMD-AGI/Primus"
PRIMUS_REF = "v26.7.0"

WORKLOAD_CONFIGS = {
    "llama-3.1-70b-training": {
        "type": "training",
        "model_repo": "primus_pyt_megatron_lm_train_llama-3.1-70b",
        "model_repo_aliases": [
            "primus_pyt_megatron_lm_train_llama-3.1-70b_overlay",
            "primus_pyt_megatron_lm_train_llama-3.1-70b_scaleout",
        ],
        "base_image": "rocm/primus:v26.4",
        "gpu_target": "gfx950",
        "metric_key": "tokens_per_second_per_gpu",
        "multiple_results": "perf_primus-megatron-Llama-3.1-70B.csv",
        "precisions": ["BF16", "FP8"],  # what MAD's run.sh trains on gfx950
        "reference_values": {
            "2N": 1685,
            "4N": 1600,
            "16N": 1685,
            "32N": 1485,
            "44N": 1432,
        },
        "slurm_partition": "meta64",
        "gpus_per_node": 8,
        "time_limit": "04:00:00",  # one allocation, two runs, plus image staging
        "docker_mounts": {"/dev/infiniband": "/dev/infiniband"},
        "docker_run_options": "--privileged --group-add render --shm-size 64G "
            "--device=/dev/infiniband --cap-add IPC_LOCK "
            "--ulimit memlock=-1 -v /sys:/sys:ro -v /run/udev:/run/udev:ro",
    },
    # The same model and image with the parallel layout swapped from FSDP2 to
    # TP1 x PP4 x DP4, so pipeline point-to-point traffic is what RCCL carries.
    # Its own key because its numbers are a different baseline. Megatron
    # rejects FSDP2 together with PP > 1, hence the three overrides together
    # (ROCm/MAD#284).
    "llama-3.1-70b-pp4-training": {
        "type": "training",
        "model_repo": "primus_pyt_megatron_lm_train_llama-3.1-70b",
        "base_image": "rocm/primus:v26.4",
        "gpu_target": "gfx950",
        "metric_key": "tokens_per_second_per_gpu",
        "multiple_results": "perf_primus-megatron-Llama-3.1-70B.csv",
        "precisions": ["BF16", "FP8"],
        "env_vars": {
            "PRIMUS_EXTRA_ARGS": "--pipeline_model_parallel_size 4 "
                "--use_torch_fsdp2 false --use_distributed_optimizer true",
        },
        # PP4 on 16 GPUs is DP4; another scale is another layout.
        "required_nodes": 2,
        "slurm_partition": "meta64",
        "gpus_per_node": 8,
        "time_limit": "04:00:00",  # one allocation, two runs, plus image staging
        "docker_mounts": {"/dev/infiniband": "/dev/infiniband"},
        "docker_run_options": "--privileged --group-add render --shm-size 64G "
            "--device=/dev/infiniband --cap-add IPC_LOCK "
            "--ulimit memlock=-1 -v /sys:/sys:ro -v /run/udev:/run/udev:ro",
    },
    "gpt-oss-120b-training": {
        "type": "training",
        "model_repo": "primus_pyt_megatron_lm_train_gpt-oss-120b",
        "base_image": "rocm/primus:v26.4",
        "gpu_target": "gfx950",
        "metric_key": "tokens_per_second_per_gpu",
        "multiple_results": "perf_primus-megatron-GPT-OSS-120B.csv",
        "precisions": ["BF16", "FP8"],  # what MAD's run.sh trains on gfx950
        # The shipped config profiles iterations 6-7, which at 2 nodes cost
        # 3.2x (BF16) and 3.7x (FP8) a steady iteration and leave the next two
        # elevated. Primus averages from iteration 3, so 15 iterations give 13
        # measured ones at ~65 s (BF16) and ~116 s (FP8) each.
        "env_vars": {
            "PRIMUS_TRAIN_ITERS": "15",
            "PRIMUS_DISABLE_PROFILE": "1",
        },
        "slurm_partition": "meta64",
        "gpus_per_node": 8,
        "time_limit": "04:00:00",  # one allocation, two runs, plus image staging
        "docker_mounts": {"/dev/infiniband": "/dev/infiniband"},
        "docker_run_options": "--privileged --group-add render --shm-size 64G "
            "--device=/dev/infiniband --cap-add IPC_LOCK "
            "--ulimit memlock=-1 -v /sys:/sys:ro -v /run/udev:/run/udev:ro",
    },
    # The first JAX workload here, and the first one whose base image is not
    # rocm/primus. That drives the "family" dispatch in install_madengine(),
    # stage_rccl_overlay() and generate_manifest(): a different MAD
    # script directory, a Primus checkout baked into both A/B images, and a
    # launcher that takes a Primus config rather than a model repo name.
    #
    # Settings that look arbitrary are each a failure mode found during the MAD
    # enabling (ROCm/MAD#228):
    #   XLA_GPU_AUTOTUNE_LEVEL=0  Primus defaults to 4, which at 32 ranks
    #       presents as a dead hang -- no steps, no NCCL output, one thread
    #       spinning in libhsa-runtime64 -- rather than a slow start. The
    #       MAD-native XLA_AUTOTUNE_LEVEL is a silent no-op here.
    #   per_device_batch_size=1   the shipped config OOMs at 241.25 GiB for
    #       jit_train_step at 4 nodes; that is activation memory and does not
    #       move with the parallelism overrides.
    #   nofile=1048576            a 32-device clique exhausts the default and
    #       stalls silently at "Initialize clique".
    "llama-3.1-405b-maxtext": {
        "type": "training",
        "family": "maxtext",
        # MAD's models.json calls this "jax-maxtext/maxtext_MI355X_...", but the
        # directory prefix must NOT travel into the manifest: madengine builds
        # its sbatch script path as
        #   <output_dir>/madengine_{model_info[name]}.sh
        # (deployment/slurm.py), so a slash in the name points the write at a
        # subdirectory that was never created and the whole deployment dies
        # with "Failed to generate script: No such file or directory" before
        # anything is submitted. Verified against the pinned madengine and
        # v2.2.1 -- both carry that line. The name is only an identifier here;
        # `scripts` is what actually locates run.sh.
        "model_repo": "maxtext_MI355X_llama3.1_405B-fp8-pretrain",
        "base_image": "rocm/jax-training:maxtext-v26.6",
        "gpu_target": "gfx950",
        "metric_key": "tokens_per_second_per_gpu",
        # extract_maxtext_perf.py emits tok_per_s_per_gpu / TFLOPS_per_gpu /
        # seconds_per_step, and madengine copies those names through unchanged.
        # The global alias accepts only `tok_per_s_per_gpu_avg`, which this
        # extractor never writes -- its `tok_per_s_per_gpu` IS the average, of
        # the last 10 steps. Accepted here and not in _METRIC_ALIASES because
        # Megatron emits both names with different meanings, and widening the
        # global table would score Llama-3.1-70B on its final iteration again.
        "metric_aliases": {"tok_per_s_per_gpu"},
        "tflops_aliases": {"TFLOPS_per_gpu"},
        "multiple_results": "primus_perf_output.csv",
        "training_precision": "fp8",
        "primus_config": "examples/maxtext/configs/MI355X/llama3.1_405B-fp8-pretrain.yaml",
        "model_args_extra": "per_device_batch_size=1",
        "tags": ["maxtext", "jax", "MI355X", "llama3.1_405B-fp8-pretrain",
                 "fp8", "training"],
        "env_vars": {
            # INFO, not the WARN every other workload inherits, and it is
            # verify_rccl_replacement() that needs it rather than anyone
            # reading the log. RCCL prints its version banner from
            #   if ((levelMask & (1u << NCCL_LOG_INFO)) ||
            #       ((levelMask & (1u << NCCL_LOG_VERSION)) && myrank == 0))
            # (src/init.cc, the showVersion call_once), so at WARN only global
            # rank 0 prints it. MaxText runs one communicator across all 32
            # GPUs, so rank 0 sits on exactly one node and the other three
            # produce no banner -- while the check requires one per node and
            # would fail a run that trained perfectly. Megatron escapes this
            # because PyTorch builds several process groups, so some rank 0
            # lands on every node; that is luck, not design, and it is why the
            # setting is scoped to this workload.
            #
            # SUBSYS keeps the volume down: INIT and NET are what the banner
            # and the "NET/IB ... GDRDMA" confirmation live in. Both match
            # ROCm/MAD#228's validated template.
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,NET",
            "XLA_GPU_AUTOTUNE_LEVEL": "0",
            # In MAD#228's validated template. The Megatron workloads are green
            # without it on the same cluster, so this is "explicitly no plugin"
            # rather than a known fix.
            "NCCL_NET_PLUGIN": "none",
        },
        "slurm_partition": "meta64",
        "gpus_per_node": 8,
        # Enforced in main() before the allocation is taken: 4 is the scale
        # MAD#228 validated, and per_device_batch_size=1 already sits at the
        # memory ceiling there.
        "required_nodes": 4,
        # MAD#228's template asks for 12:00:00 for a single run. 05:00:00 for
        # the A/B pair instead: a 4-node exclusive request with a 12 h limit
        # backfills badly on a shared partition, the measured 4N FP8 run is
        # ~17 min of training (MAD#228 reports 17.40 s/step) plus prep, and
        # 300 min stays inside the step's own 330 so the script still reports.
        "time_limit": "05:00:00",
        "docker_mounts": {"/dev/infiniband": "/dev/infiniband"},
        "docker_run_options": "--privileged --group-add render --shm-size 64G "
            "--device=/dev/infiniband --cap-add IPC_LOCK "
            "--ulimit memlock=-1 --ulimit nofile=1048576 "
            "-v /sys:/sys:ro -v /run/udev:/run/udev:ro",
    },
    # The first inference workload: DeepSeek-R1 served disaggregated, two
    # prefill and two decode nodes, KV cache moved by nixl. MoE all-to-all is
    # left to RCCL (MOE_A2A_BACKEND=none) rather than MoRI, so expert traffic
    # is what the candidate library carries.
    #
    # Both A/B images are MAD's sglang Dockerfile; see
    # stage_sglang_baseline().
    "deepseek-r1-sglang-disagg": {
        "type": "inference",
        "family": "sglang-disagg",
        # madengine names each result row <this>_2p2d_isl<I>_osl<O>_con<C>.
        "model_repo": "sglang-disagg-deepseek-r1",
        # A ROCm SDK (TheRock) base, not a rocm7xx one: the candidate librccl
        # is kpack-split, and a pre-kpack HIP runtime segfaults on it in
        # ncclInitKernelsForDevice. No newer than sglang 0.5.18: 0.5.19 split
        # --cuda-graph-bs into -decode/-prefill, and argparse rejects the
        # plain flag MAD's models.yaml passes as ambiguous.
        "base_image": "rocm/sgl-dev:v0.5.18-rocm10-mi35x-20260904",
        "gpu_target": "gfx950",
        "metric_key": "output_token_throughput_tok_s",
        "metric_unit": "output tok/s of the whole 2P+2D deployment",
        "multiple_results": "perf_sglang-disagg-DeepSeek-R1.csv",
        # The sweep points that are scored, in place of precisions; the rest of
        # the sweep is run and ignored. Only the highest concurrency: the lower
        # ones are latency-bound and say little about the collectives.
        "precisions": ["isl1024_osl1024_con64", "isl8192_osl1024_con64"],
        "tags": ["pyt", "sglang", "disagg", "inference", "deepseek-r1"],
        "model_args": "--model-name DeepSeek-R1 --model-path /models/DeepSeek-R1-0528 "
            "--run-mori 1 --dp-mode 1 --xp 2 --yd 2 --gpus-per-node 8 "
            "--kv-transfer-backend nixl",
        "disagg": {"prefill_nodes": 2, "decode_nodes": 2},
        "env_vars": {
            "MODEL_NAME": "DeepSeek-R1",
            "MODEL_PATH": "/models/DeepSeek-R1-0528",
            "RUN_MORI": "1",
            "DP_MODE": "1",
            "MOE_A2A_BACKEND": "none",
            "xP": "2",
            "yD": "2",
            "GPUS_PER_NODE": "8",
            "KV_TRANSFER_BACKEND": "nixl",
            # INFO for the same reason as the MaxText entry: at WARN only a
            # communicator's rank 0 prints the version banner, and a server
            # spanning two nodes has it on one of them.
            "NCCL_DEBUG": "INFO",
            # TUNING records the effective CSV source. The candidate gate
            # requires its SHA-pinned gfx950 table and rejects a stale stock CSV.
            "NCCL_DEBUG_SUBSYS": "INIT,NET,TUNING",
            # Thor2 settings the sglang stack runs with, over the cluster's.
            "NCCL_IB_QPS_PER_CONNECTION": "1",
            "NCCL_IB_USE_INLINE": "0",
            "NCCL_IB_MERGE_NICS": "0",
            "NCCL_IB_SPLIT_DATA_ON_QPS": "0",
            "NCCL_IB_ADAPTIVE_ROUTING": "0",
            "NCCL_GDR_FLUSH_DISABLE": "1",
            "NCCL_DMABUF_ENABLE": "0",
            "NCCL_IB_ROCE_VERSION_NUM": "2",
            "IB_DEVICES": "bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,"
                "bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7",
            "MORI_RDMA_DEVICES": "bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,"
                "bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7",
            "MORI_SOCKET_IFNAME": "fenic0",
            "MORI_IB_GID_INDEX": "3",
            "MORI_JIT_CACHE_DIR": "/tmp/mori_jit",
            "MORI_SHMEM_HEAP_SIZE": "17179869184",
            "RDMAV_DRIVERS": "bnxt_re",
            "IBV_DRIVERS": "bnxt_re",
            "GPU_MAX_HW_QUEUES": "2",
            "HSA_ENABLE_SDMA": "0",
            "HSA_FORCE_FINE_GRAIN_PCIE": "1",
            # The common OMP/TORCH_NCCL settings reach only the deployment
            # env, not the container, and without OMP_NUM_THREADS each of the
            # eight server processes per node sizes its pool to every core.
            "OMP_NUM_THREADS": "8",
            "TORCH_NCCL_HIGH_PRIORITY": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            # Part of meta64 has less VRAM per GPU, and sglang refuses a TP
            # group spanning both kinds ("The memory capacity is unbalanced").
            # Off, it warns and sizes the KV pool by the smaller GPU; both A/B
            # phases share the nodes, so the ratio is unaffected.
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
            # A hang otherwise holds four nodes for two hours before failing.
            "BARRIER_TIMEOUT_S": "1800",
            "IP_SYNC_TIMEOUT": "1800",
            "SMOKE": "0",
            # A point is only 2 x concurrency requests, and one pass at
            # isl1024 con64 moved 6.8% between two A/A phases on the same
            # nodes. The parser keeps each point's best iteration, so three
            # passes of just the gated points cost about what the full sweep
            # did.
            "BENCHMARK_ITR": "3",
            "BENCHMARK_COMBINATIONS": "1024/1024 8192/1024",
            "BENCHMARK_CONCURRENCY_LEVELS": "64",
            "BENCHMARK_PRECHECK": "0",
            "BENCHMARK_FAIL_FAST": "1",
        },
        # The image builds its own rdma-core 63 for Thor2; the host's 61
        # mounted over it would put nixl and MoRI on a stack never run here.
        "mount_host_ib_libs": False,
        "required_nodes": 4,
        "slurm_partition": "meta64",
        "gpus_per_node": 8,
        # One phase took 31 min by hand with the image cached; each phase here
        # also builds, saves and loads it.
        "time_limit": "03:00:00",
        "docker_mounts": {
            "/dev/infiniband": "/dev/infiniband",
            "/sys/class/infiniband": "/sys/class/infiniband",
        },
        # --init: the launcher polls its servers with `kill -0`, which succeeds
        # on a zombie, and without a reaping PID 1 every node exits 1 after a
        # complete sweep.
        "docker_run_options": "--init --privileged --group-add render --shm-size 64G "
            "--device=/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 "
            "--ulimit nofile=1048576:1048576 "
            "-v /sys:/sys:ro -v /run/udev:/run/udev:ro -e SLURM_JOB_ID",
    },
}

CLUSTER_CONFIGS = {
    "ruby": {
        "gpu_target": "gfx950",
        "slurm_partition": "meta64",
        "slurm_qos": "",
        "slurm_no_gres": True,  # Ruby SLURM has no GPU GRES configured
        "mount_host_ib_libs": True,
        "nccl_env": {
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_HCA": "bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re6:1,bnxt_re7:1",
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_IB_TC": "104",
            "NCCL_IB_QPS_PER_CONNECTION": "4",
            "NCCL_SOCKET_IFNAME": "fenic0",
            "NCCL_DEBUG": "WARN",
        },
        "results_base": "/apps/rccl-ci/madengine/perf",
        # Mounted read-only at /models for the inference workloads.
        "models_dir": "/apps/data/models",
    },
}


def check_mad_knobs(work_dir: Path, workload_config: dict) -> None:
    """Fail if the MAD pin's Megatron script never reads a PRIMUS_* knob we set.

    An unknown knob is simply ignored there, and the run measures the shipped
    config under this workload's name -- the PP=4 key would score FSDP.
    """
    if workload_config.get("family", "megatron") != "megatron":
        return
    knobs = [k for k in workload_config.get("env_vars", {}) if k.startswith("PRIMUS_")]
    if not knobs:
        return
    script = (work_dir / "scripts" / "primus_megatron-lm"
              / "primus_megatron-lm_benchmark_report.sh")
    text = script.read_text(errors="replace") if script.is_file() else ""
    missing = [k for k in knobs if k not in text]
    if missing:
        raise RuntimeError(
            f"MAD pin {MAD_REF[:12]} does not read {', '.join(missing)} "
            f"({script}) -- the run would ignore it"
        )


def fetch_primus(mad_dir: Path) -> Path:
    """Check Primus out into MAD's scripts/Primus and return the path.

    Both MaxText images COPY this tree, so it has to exist before either build.
    MAD ships tools/fetch_primus.sh for exactly this and it is idempotent, so
    prefer it; the fallback only matters if the MAD pin predates the script.
    """
    primus_dir = mad_dir / "scripts" / "Primus"
    fetch_script = mad_dir / "tools" / "fetch_primus.sh"

    env = os.environ.copy()
    env["PRIMUS_URL"] = PRIMUS_URL
    env["PRIMUS_REF"] = PRIMUS_REF

    if fetch_script.is_file():
        log.info("Checking Primus out at %s via tools/fetch_primus.sh...", PRIMUS_REF)
        subprocess.run(["bash", str(fetch_script)], check=True,
                       cwd=str(mad_dir), env=env)
    else:
        log.info("No tools/fetch_primus.sh in this MAD pin; cloning Primus directly")
        primus_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth=1", "--branch", PRIMUS_REF,
             PRIMUS_URL, str(primus_dir)],
            check=True,
        )

    # Checked here rather than four layers into a docker build, where it
    # surfaces as a bare "no such file".
    if not (primus_dir / "primus-cli").is_file():
        raise RuntimeError(
            f"Primus checkout at {primus_dir} has no primus-cli -- "
            f"PRIMUS_REF={PRIMUS_REF} is wrong or the checkout failed"
        )
    config_rel = "examples/maxtext/configs/MI355X"
    if not (primus_dir / config_rel).is_dir():
        raise RuntimeError(
            f"Primus checkout at {primus_dir} has no {config_rel} -- "
            f"PRIMUS_REF={PRIMUS_REF} does not carry the MI355X MaxText configs"
        )
    log.info("Primus ready at %s (%s)", primus_dir, PRIMUS_REF)
    return primus_dir


def install_madengine(work_dir: Path, family: str = "megatron") -> Path:
    """Clone and install madengine into the current Python environment."""
    madengine_dir = work_dir / "madengine"
    mad_dir = work_dir / "MAD"

    if not madengine_dir.exists():
        log.info("Cloning madengine at %s...", MADENGINE_REF[:12])
        subprocess.run(
            ["git", "clone", "--depth=1", MADENGINE_REPO, str(madengine_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(madengine_dir), "fetch", "--depth=1", "origin", MADENGINE_REF],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(madengine_dir), "checkout", MADENGINE_REF],
            check=True,
        )

    if not mad_dir.exists():
        log.info("Cloning MAD (%s) at %s...", MAD_BRANCH, MAD_REF[:12])
        subprocess.run(
            ["git", "clone", "--depth=1", "--branch", MAD_BRANCH, MAD_REPO, str(mad_dir)],
            check=True,
        )
        # Verify the clone landed on the expected SHA. The --branch clone
        # gives us the branch tip; fetch+checkout is only needed if the
        # pinned ref differs from the tip (e.g. after the branch moves).
        head = subprocess.run(
            ["git", "-C", str(mad_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if not head.startswith(MAD_REF[:12]):
            log.info("MAD HEAD %s != pinned %s, fetching...", head[:12], MAD_REF[:12])
            subprocess.run(
                ["git", "-C", str(mad_dir), "fetch", "--depth=1", "origin", MAD_REF],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(mad_dir), "checkout", MAD_REF],
                check=True,
            )
        else:
            log.info("MAD HEAD matches pinned ref: %s", head[:12])

    log.info("Installing madengine...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(madengine_dir)],
        check=True,
    )

    log.info("madengine installed to: %s", madengine_dir)

    if family == "maxtext":
        scripts_src = mad_dir / "scripts" / "jax-maxtext"
        scripts_name = "jax-maxtext"
    elif family == "sglang-disagg":
        scripts_src = mad_dir / "scripts" / "sglang_disagg"
        scripts_name = "sglang_disagg"
    else:
        scripts_src = mad_dir / "scripts" / "primus_megatron-lm"
        if not scripts_src.is_dir():
            scripts_src = mad_dir / "scripts" / "primus" / "megatron-lm"
        scripts_name = "primus_megatron-lm"

    scripts_dst = work_dir / "scripts" / scripts_name
    if scripts_src.is_dir() and not scripts_dst.exists():
        scripts_dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-r", str(scripts_src), str(scripts_dst)], check=True)
        log.info("Copied MAD %s scripts to %s", scripts_name, scripts_dst)
    elif not scripts_src.is_dir():
        # Fatal for maxtext and sglang: the manifest points `scripts` at
        # run.sh, and a missing one surfaces as a container-side "no such
        # file" on every node instead of here. The megatron path keeps its
        # historical warning.
        if family == "maxtext":
            raise RuntimeError(
                f"MAD jax-maxtext scripts not found at {scripts_src} -- "
                f"MAD pin {MAD_REF[:12]} predates ROCm/MAD#228"
            )
        if family == "sglang-disagg":
            raise RuntimeError(
                f"MAD sglang_disagg scripts not found at {scripts_src} -- "
                f"MAD pin {MAD_REF[:12]} does not carry them"
            )
        log.error("MAD primus scripts not found — expected at %s", scripts_src)

    if family == "maxtext":
        fetch_primus(mad_dir)
    if family == "sglang-disagg" and not (mad_dir / SGLANG_MAD_DOCKERFILE).is_file():
        raise RuntimeError(
            f"MAD pin {MAD_REF[:12]} has no {SGLANG_MAD_DOCKERFILE}"
        )

    result = subprocess.run(
        ["madengine", "--version"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        log.info("madengine version: %s", result.stdout.strip())
    else:
        log.warning("madengine --version failed, but install may still be usable")

    return madengine_dir


def get_rccl_commit(rccl_lib: Path | None = None) -> str:
    """Derive a unique identifier for the RCCL build.

    Checks, in order: exact CANDIDATE_SHA, RCCL_COMMIT_HASH, GITHUB_RUN_ID,
    sha256 of the librccl.so binary. Does NOT fall back to ``git rev-parse HEAD``
    because in CI the checkout is TheRock (not RCCL), which would produce
    a constant tag and cause stale cache hits on persistent runners.
    """
    commit = os.environ.get("CANDIDATE_SHA", "")
    if commit:
        return commit[:12]

    commit = os.environ.get("RCCL_COMMIT_HASH", "")
    if commit:
        return commit[:12]

    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if run_id:
        return f"run{run_id}"

    if rccl_lib and rccl_lib.exists():
        h = hashlib.sha256(rccl_lib.read_bytes()).hexdigest()
        return h[:12]

    return "unknown"


def _rccl_uses_kpack(rccl_lib: Path) -> bool:
    """Check if librccl.so uses kpack (GPU kernels in separate .kpack files).

    TheRock builds with kpack produce a small .so (~4MB) with a
    .rocm_kpack_ref section and an empty (NOBITS) .hip_fatbin section.
    These libraries crash on base images whose HIP runtime pre-dates
    kpack support.
    """
    try:
        result = subprocess.run(
            ["readelf", "-S", str(rccl_lib)],
            capture_output=True, text=True, timeout=10,
        )
        return ".rocm_kpack_ref" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rccl_kpack_files(rccl_lib: Path) -> list[Path]:
    """The kpack archives belonging to *rccl_lib*, in stable name order."""
    files = list(rccl_lib.parent.rglob("*.kpack"))
    if not files:
        files = list(rccl_lib.parent.parent.rglob("rccl*.kpack"))
    return sorted(set(p.resolve() for p in files), key=lambda p: p.name)


def rccl_asset_identity(rccl_lib: Path, tuner_config: Path) -> str:
    """Short identity of every version-coupled candidate overlay asset."""
    digest = hashlib.sha256()
    for path in [rccl_lib.resolve(), *rccl_kpack_files(rccl_lib), tuner_config.resolve()]:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def get_rccl_fingerprint(rccl_lib: Path) -> dict:
    """Extract identifying information from a librccl.so artifact.

    Returns hashes, ``version`` (e.g. ``2.30.4-HEAD:e711c9e``), and ``size``
    that can be compared against runtime output.
    """
    fp: dict = {"md5": "", "sha256": "", "version": "", "size": 0}
    if not rccl_lib or not rccl_lib.exists():
        return fp
    resolved = rccl_lib.resolve()
    fp["size"] = resolved.stat().st_size
    fp["md5"] = hashlib.md5(resolved.read_bytes()).hexdigest()
    fp["sha256"] = sha256_file(resolved)
    try:
        result = subprocess.run(
            ["strings", str(resolved)],
            capture_output=True, text=True, timeout=10,
        )
        # The banner is assembled at run time: init.cc formats VERSION_STRING
        # and rcclGitHash together, while git_version.cmake emits the hash as its
        # own `<branch>:<7 hex><+ if dirty>` literal. So the two halves live in
        # the binary separately and have to be recombined here -- without the
        # hash the fingerprint cannot tell this build from the image's bundled
        # copy of the same version.
        semver = ""
        git_ref = ""
        for line in result.stdout.splitlines():
            if not semver:
                m = re.match(r"^(\d+\.\d+\.\d+)$", line)
                if m:
                    semver = m.group(1)
            if not git_ref:
                m = re.match(r"^([\w./-]+:[0-9a-fA-F]{7}\+?)$", line)
                if m:
                    git_ref = m.group(1)
            if semver and git_ref:
                break
        if semver and git_ref:
            fp["version"] = f"{semver}-{git_ref}"
        elif semver:
            fp["version"] = semver
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return fp


def _split_rccl_version(version: str) -> tuple[str, str, bool]:
    """Split into semver, commit and dirty flag.

    `2.30.7`, `2.30.7-HEAD:e711c9e`, `2.30.7-develop:1b64803+`. The trailing `+`
    is kept apart from the hash: a build with uncommitted changes is not the
    same library as a clean one at that commit, and hashes may be abbreviated
    to different lengths.
    """
    m = re.match(r"(\d+\.\d+\.\d+)(?:-\S*?:([0-9a-fA-F]{6,})(\+?))?", version)
    if not m:
        return version, "", False
    return m.group(1), (m.group(2) or "").lower(), m.group(3) == "+"


CANDIDATE_TUNER_PATH = "/opt/rccl-candidate/tuner/rccl_tuner_gfx950.csv"
FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
CANDIDATE_PROVENANCE = "candidate-provenance.json"


def candidate_provenance(
    candidate_sha: str,
    artifact_provenance_path: Path,
    tuner_config: Path,
    rccl_lib: Path,
    rccl_fingerprint: dict,
) -> dict:
    """Validate all pre-run candidate assets and return their manifest."""
    candidate_sha = candidate_sha.strip().lower()
    if not FULL_COMMIT_SHA.fullmatch(candidate_sha):
        raise RuntimeError(
            f"candidate SHA must be a full 40-character commit, got {candidate_sha!r}"
        )
    try:
        artifact = json.loads(artifact_provenance_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot read candidate artifact provenance {artifact_provenance_path}: {exc}"
        ) from exc
    for key in ("candidate_sha", "artifact_source_sha"):
        actual = str(artifact.get(key, "")).lower()
        if actual != candidate_sha:
            raise RuntimeError(
                f"candidate artifact provenance mismatch: {key}={actual!r}, "
                f"expected {candidate_sha}"
            )

    _, lib_commit, dirty = _split_rccl_version(rccl_fingerprint.get("version", ""))
    if dirty or not lib_commit or not candidate_sha.startswith(lib_commit):
        raise RuntimeError(
            "candidate librccl version does not match the exact source SHA: "
            f"version={rccl_fingerprint.get('version')!r}, source={candidate_sha}"
        )
    if not tuner_config.is_file():
        raise RuntimeError(f"candidate tuner source is missing: {tuner_config}")

    uses_kpack = _rccl_uses_kpack(rccl_lib)
    kpack_files = rccl_kpack_files(rccl_lib)
    if uses_kpack and not kpack_files:
        raise RuntimeError(
            f"{rccl_lib} references kpack kernels but its exact artifact has no .kpack files"
        )

    return {
        "schema_version": 1,
        "candidate_sha": candidate_sha,
        "overlay_asset_id": rccl_asset_identity(rccl_lib, tuner_config),
        "artifact": artifact,
        "librccl": {
            "path": os.fspath(rccl_lib),
            **rccl_fingerprint,
        },
        "kpack": [
            {"path": os.fspath(path), "sha256": sha256_file(path), "size": path.stat().st_size}
            for path in kpack_files
        ],
        "tuner": {
            "policy": "remove stock disk overrides and install the exact candidate CSV",
            "source": "candidate source tree at the verified candidate SHA",
            "source_path": os.fspath(tuner_config),
            "source_sha256": sha256_file(tuner_config),
            "expected_runtime_source": CANDIDATE_TUNER_PATH,
        },
    }


def write_candidate_provenance(work_dir: Path, manifest: dict) -> None:
    """Write the live manifest and keep the repro copy in sync when present."""
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (work_dir / CANDIDATE_PROVENANCE).write_text(text)
    repro = work_dir / REPRO_DIR / CANDIDATE_PROVENANCE
    if repro.parent.is_dir():
        repro.write_text(text)


# RCCL prints this on EVERY call of a deprecated API name (src/misc/api_trace.cc),
# at WARN, so NCCL_DEBUG=WARN keeps it. GPT-OSS-120B's MoE all-to-all makes it
# 97% of each node log, which madengine keeps three copies of.
_LOG_SPAM_RE = re.compile(rb"NCCL WARN Please note that (\w+) is deprecated")


def squash_log_spam(run_dir: Path) -> int:
    """Drop the repeats of _LOG_SPAM_RE lines from one phase's logs, in place.

    The first occurrence of each stays where it was, and one line at the end
    of the file says how many repeats went. Files with no repeats are left
    untouched. Returns the number of bytes saved.
    """
    log_dir = run_dir / "slurm_output"
    if not log_dir.is_dir():
        return 0
    saved = 0
    for path in sorted(
        p for p in log_dir.rglob("*")
        if p.suffix in (".out", ".err", ".log") and p.is_file()
    ):
        dropped: dict[bytes, int] = {}
        tmp = path.with_name(path.name + ".squash")
        try:
            before = path.stat().st_size
            with path.open("rb") as src, tmp.open("wb") as dst:
                for line in src:
                    m = _LOG_SPAM_RE.search(line)
                    if m:
                        api = m.group(1)
                        if api in dropped:
                            dropped[api] += 1
                            continue
                        dropped[api] = 0
                    dst.write(line)
                for api, count in dropped.items():
                    if count:
                        dst.write(
                            b"[rccl-ci] dropped %d repeats of 'NCCL WARN Please note "
                            b"that %s is deprecated'\n" % (count, api)
                        )
            if not any(dropped.values()):
                tmp.unlink()
                continue
            os.replace(tmp, path)
            saved += before - path.stat().st_size
        except OSError as exc:
            log.warning("Could not squash %s: %s", path, exc)
            tmp.unlink(missing_ok=True)
    return saved


def rccl_banner_logs(work_dir: Path) -> list[Path]:
    """One log per node of a phase, the one its RCCL version banner lands in.

    madengine's per-node logs, unless the phase has sglang server logs: that
    launcher sends the servers' output, and with it the banner, to /run_logs
    (one prefill or decode log per node) instead.
    """
    log_dir = work_dir / "slurm_output"
    if not log_dir.is_dir():
        return []
    server_logs = sorted(
        p for p in log_dir.glob("run_logs/*/*_NODE*.log")
        if p.name.startswith(("prefill_", "decode_"))
    )
    return server_logs or sorted(log_dir.glob("*node_*.out"))


def runtime_rccl_versions(work_dir: Path) -> list[str]:
    """Distinct ``RCCL version :`` banners across one phase's node logs."""
    versions: list[str] = []
    for log_file in rccl_banner_logs(work_dir):
        for v in re.findall(r"RCCL version\s*:\s*(\S+)", log_file.read_text(errors="replace")):
            if v not in versions:
                versions.append(v)
    return versions


def runtime_tuner_sources(work_dir: Path) -> list[str]:
    """Distinct CSV tuner sources reported by RCCL in this phase."""
    sources: list[str] = []
    pattern = re.compile(r"TUNER/CsvTuner: Loaded \d+ tuning configurations from (\S+)")
    for log_file in rccl_banner_logs(work_dir):
        for source in pattern.findall(log_file.read_text(errors="replace")):
            if source not in sources:
                sources.append(source)
    return sources


def verify_candidate_tuner(
    work_dir: Path,
    nodes: int,
    expected_source: str = CANDIDATE_TUNER_PATH,
) -> tuple[bool, str]:
    """Require every candidate node to report only the SHA-pinned candidate CSV."""
    node_logs = rccl_banner_logs(work_dir)
    if not node_logs:
        return False, "No node logs found — cannot verify tuner source"
    by_node: dict[int, list[str]] = {}
    pattern = re.compile(r"TUNER/CsvTuner: Loaded \d+ tuning configurations from (\S+)")
    for log_file in node_logs:
        match = re.search(r"_(?:node_|NODE)(\d+)\.(?:out|log)$", log_file.name)
        if not match:
            continue
        node = int(match.group(1))
        sources = pattern.findall(log_file.read_text(errors="replace"))
        by_node.setdefault(node, []).extend(sources)

    missing = sorted(node for node in range(nodes) if not by_node.get(node))
    if missing:
        return False, (
            f"no loaded CSV tuner source in node log(s) {missing}; "
            "NCCL_DEBUG_SUBSYS must include TUNING"
        )
    wrong = {
        node: sorted(set(sources))
        for node, sources in by_node.items()
        if any(source != expected_source for source in sources)
    }
    if wrong:
        return False, (
            f"candidate tuner source mismatch: expected {expected_source}, got {wrong}; "
            "a stock disk CSV may be shadowing the candidate"
        )
    return True, f"tuner verified on {nodes} node(s): source={expected_source}"


def verify_rccl_replacement(
    work_dir: Path,
    expected: dict,
    nodes: int,
    slurm_job_id: str = "",
) -> tuple[bool, str]:
    """Verify that every node used the CI-built RCCL, not its bundled copy.

    Requires a banner log (rccl_banner_logs) from each of the *nodes* nodes;
    a library that reached only some of them fails.

    Returns (ok, message).  Generic across frameworks — any RCCL-based
    application prints the version string during init.
    """
    if not expected.get("version"):
        return True, "No RCCL version to verify"

    node_logs = rccl_banner_logs(work_dir)
    if not node_logs:
        return False, "No node logs found — cannot verify RCCL"
    logged = {
        int(m.group(1))
        for log_file in node_logs
        if (m := re.search(r"_(?:node_|NODE)(\d+)\.(?:out|log)$", log_file.name))
    }
    missing = sorted(set(range(nodes)) - logged)
    if missing:
        return False, (
            f"no log from node(s) {', '.join(map(str, missing))} of {nodes} — "
            f"cannot verify RCCL there"
        )

    verified_nodes: list[str] = []
    for log_file in node_logs:
        node_label = log_file.name
        log_text = log_file.read_text(errors="replace")

        rccl_versions = re.findall(r"RCCL version\s*:\s*(.+)", log_text)
        if not rccl_versions:
            return False, (
                f"{node_label}: no 'RCCL version :' found — "
                f"RCCL may not have initialized on this node"
            )

        # The container spells the build out further than the artifact does:
        # `2.30.7-develop:1b64803+` against a bare `2.30.7`. The commit is what
        # tells the CI build from the image's bundled copy, so it decides
        # whenever the artifact carries one; against a bare fingerprint the two
        # are indistinguishable and this degrades to a version check.
        runtime_version = rccl_versions[0].strip()
        exp_semver, exp_commit, exp_dirty = _split_rccl_version(expected["version"])
        run_semver, run_commit, run_dirty = _split_rccl_version(runtime_version)
        same_commit = (
            exp_commit
            and run_commit
            and (run_commit.startswith(exp_commit) or exp_commit.startswith(run_commit))
            and exp_dirty == run_dirty
        )
        if run_semver != exp_semver or (exp_commit and not same_commit):
            return False, (
                f"{node_label}: RCCL version mismatch: "
                f"expected '{expected['version']}' (from artifact), "
                f"got '{runtime_version}' (from container)"
            )
        verified_nodes.append(node_label)

    return True, (
        f"RCCL verified on {len(verified_nodes)} node(s): "
        f"version={expected['version']}, "
        f"artifact_md5={expected.get('md5', 'N/A')}"
    )


def _soname_bridge(dirs: str, sonames: str) -> str:
    """Shell that links each missing soname in *dirs* onto the copy already there."""
    if not sonames:
        return ""
    return (
        f'    for d in {dirs}; do \\\n'
        f'        for so in {sonames}; do \\\n'
        '            [ -e "$d/$so" ] && continue; \\\n'
        '            base=${so%%.so.*}.so; \\\n'
        '            have=$(for f in "$d/$base".*; do \\\n'
        '                [ -e "$f" ] && [ "$f" != "$d/$so" ] && echo "$f"; \\\n'
        '            done | sort -V | tail -1); \\\n'
        '            [ -n "$have" ] && ln -sfn "$(basename "$have")" "$d/$so"; \\\n'
        '        done; \\\n'
        '    done; \\\n'
    )


def _needed_sonames(lib: Path) -> list[str]:
    """DT_NEEDED entries of *lib* that are safe to bridge across ROCm versions.

    Only `libamd_smi`: RCCL touches a narrow part of it, and it is the one that
    actually skews. Bridging a core runtime like `libamdhip64` would trade a
    clear load failure for an unresolved symbol somewhere deeper.
    """
    try:
        result = subprocess.run(
            ["readelf", "-d", str(lib)], capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log.warning("readelf unusable (%s); not reconciling dependencies", exc)
        return []
    if result.returncode != 0:
        log.warning("readelf failed on %s; not reconciling dependencies", lib)
        return []
    sonames = re.findall(r"\(NEEDED\).*\[([^\]]+)\]", result.stdout)
    return [s for s in sonames if s.startswith("libamd_smi.")]


def _soname_bridge_discovered(sonames: str) -> str:
    """Shell that bridges each missing soname beside every copy of that library.

    The directories are discovered from the library being bridged, not from
    librccl: every directory that already holds some version of it, which is
    exactly the set any RPATH or RUNPATH pointing at that library can reach.

    _soname_bridge() links only inside the directories that received a librccl,
    which is the wrong set. A librccl resolves its dependencies through an
    RPATH that reaches directories holding no librccl at all -- in the Primus
    image, ``_rocm_sdk_libraries/lib/librccl.so.1`` reaches
    ``_rocm_sdk_core/lib``, which carries libamd_smi and nothing else of ours.
    A soname missing there is not bridged, and the CDLL of librccl in
    ``rocm_sdk.preload_libraries()`` fails at import. The build gate does not
    catch it either: ``ldd`` on that copy reports nothing either way.
    """
    if not sonames:
        return ""
    return (
        f'    for so in {sonames}; do \\\n'
        '      base=${so%%.so.*}.so; \\\n'
        '      for d in $(find / -xdev -name "${base}.*" -printf "%h\\n" \\\n'
        '                 2>/dev/null | sort -u); do \\\n'
        '        [ -e "$d/$so" ] && continue; \\\n'
        '        have=$(for f in "$d/$base".*; do \\\n'
        '          [ -e "$f" ] && [ "$f" != "$d/$so" ] && echo "$f"; \\\n'
        '        done | sort -V | tail -1); \\\n'
        '        if [ -n "$have" ]; then \\\n'
        '          ln -sfn "$(basename "$have")" "$d/$so"; \\\n'
        '          echo "  bridged: $d/$so -> $(basename "$have")"; \\\n'
        '        fi; \\\n'
        '      done; \\\n'
        '    done; \\\n'
    )


def _librccl_swap_snippet(bridge: str = "", kpack: bool = False) -> str:
    """A RUN layer installing /tmp/rccl_ci/librccl.so over every librccl on disk.

    Ported from ROCm/MAD#228's primus_maxtext_rccl_overlay Dockerfile. Unlike
    the hardcoded SDK-venv branch below it discovers its targets, which is what
    makes it work on a base that is not the Primus image.

    Two properties are load-bearing and easy to lose in a rewrite:

    * **Each destination keeps its own RUNPATH**, including an empty one. A
      librccl inside the ROCm pip wheels resolves libamd_smi through a RUNPATH
      reaching a sibling wheel; the artifact carries a different one. Dropping
      the bytes in verbatim leaves a library that cannot resolve its own
      dependencies and init dies before step 0. Restoring an *empty* RUNPATH
      matters for the same reason in reverse: that destination resolves through
      the loader's search path, and leaving ours behind would silently change
      where its dependencies come from.

      The tag is kept too: patchelf writes DT_RUNPATH unless told otherwise,
      and DT_RPATH, which the SDK wheels' librccl carries, outranks
      LD_LIBRARY_PATH while DT_RUNPATH does not. In the rocm10 sgl-dev image,
      LD_LIBRARY_PATH names _rocm_sdk_devel/lib, which holds its own copy of
      libamd_smi.so.27. As a RUNPATH the swapped library loads that copy next
      to the _rocm_sdk_core one amdsmi uses. The two interpose on each other,
      amdsmi then reports no GPUs, and so does torch.cuda.device_count().
    * **The gate compares unresolved sonames by LIBRARY, not by exact name**,
      per target, before and after. Counts alone are not enough -- dropping one
      missing soname while introducing another leaves the count unchanged. An
      absolute "must resolve everything" test would fail a good build instead:
      the pristine _rocm_sdk_devel copy already reports unresolved deps because
      it is only ever dlopen'd where they are already loaded. The rule is that
      the swap must not make linkage worse than it found it.

      Comparing exact names is too strict. Where the artifact needs
      libamd_smi.so.27 and the base ships .26, a target whose RUNPATH is a row
      of empty entries never searches its own directory, so the bridge's
      symlink cannot help it: .26 was unresolved before the swap and .27 is
      unresolved after -- the same library, equally absent, under a different
      version. Linkage is no worse, yet a set comparison reads it as new
      breakage. So a newly unresolved soname is tolerated only when some
      version of the same library was already unresolved on that target; a
      library that resolved before and does not now still fails.

    *bridge* runs after the copy and before the post-swap measurement, so a
    soname the artifact needs and the base predates (libamd_smi .27 against
    .26) is reconciled rather than tripping the gate it would otherwise fail.
    """
    # A librccl swap is one version-coupled unit: binary, device archives and
    # tuner policy. Remove archives left by the stock image before optionally
    # installing the candidate's exact set.
    kpack_block = (
        '    for d in $(cut -f2 /opt/RCCL_PRESWAP.txt | xargs -r -n1 dirname \\\n'
        '               | xargs -r -n1 dirname | sort -u); do \\\n'
        '      rm -f "$d/.kpack"/rccl*.kpack; \\\n'
        '    done; \\\n'
    )
    if kpack:
        # Generalises what the SDK-venv branch does for two known directories:
        # every directory that received a library gets the archives too, since
        # we do not know which copy the runtime will load.
        #
        # ONE LEVEL ABOVE the library, not beside it. TheRock's packer puts the
        # archives at the artifact root and embeds `../.kpack` in a library that
        # sits under `lib/`, so a copy at `<x>/lib/librccl.so.1` resolves them
        # at `<x>/.kpack`. The two hardcoded paths elsewhere in this file agree
        # -- both name `_rocm_sdk_libraries/.kpack` for a library in
        # `_rocm_sdk_libraries/lib/` -- as does the artifact-side search, which
        # looks in `rccl_lib.parent.parent`. Writing `$d/.kpack` instead leaves
        # the device code where nothing looks for it, and nothing fails until
        # the library is actually loaded.
        kpack_block += (
            '    ls /tmp/rccl_ci/.kpack/*.kpack >/dev/null 2>&1 || \\\n'
            '      { echo "GATE FAIL: candidate librccl requires kpack but none was staged"; exit 1; }; \\\n'
            '    sha256sum /tmp/rccl_ci/.kpack/*.kpack > /opt/RCCL_KPACK_SHA256S.txt; \\\n'
            '    for d in $(cut -f2 /opt/RCCL_PRESWAP.txt | xargs -r -n1 dirname \\\n'
            '               | xargs -r -n1 dirname | sort -u); do \\\n'
            '      mkdir -p "$d/.kpack"; \\\n'
            '      for src in /tmp/rccl_ci/.kpack/*.kpack; do \\\n'
            '        cp "$src" "$d/.kpack/"; cmp -s "$src" "$d/.kpack/$(basename "$src")" || \\\n'
            '          { echo "GATE FAIL: kpack copy mismatch in $d/.kpack"; exit 1; }; \\\n'
            '      done; \\\n'
            '    done; \\\n'
            '    echo "kpack staged atomically into: $(cut -f2 /opt/RCCL_PRESWAP.txt \\\n'
            '          | xargs -r -n1 dirname | xargs -r -n1 dirname | sort -u | tr "\\\\n" " ")"; \\\n'
        )
    tuner_block = (
        '    : > /opt/RCCL_TUNER_REMOVED.txt; \\\n'
        '    for t in $(cut -f2 /opt/RCCL_PRESWAP.txt); do \\\n'
        '      d=$(dirname "$t"); \\\n'
        '      for td in "$d/tuner" "$d/../share/rccl/tuner"; do \\\n'
        '        [ -d "$td" ] || continue; \\\n'
        '        find "$td" -maxdepth 1 -name "rccl_tuner*.csv" \\\n'
        '          -print -delete >> /opt/RCCL_TUNER_REMOVED.txt; \\\n'
        '      done; \\\n'
        '    done; \\\n'
        '    for td in /opt/rocm*/share/rccl/tuner; do \\\n'
        '      [ -d "$td" ] || continue; \\\n'
        '      find "$td" -maxdepth 1 -name "rccl_tuner*.csv" \\\n'
        '        -print -delete >> /opt/RCCL_TUNER_REMOVED.txt; \\\n'
        '    done; \\\n'
        f'    install -D -m 0444 /tmp/rccl_ci/tuner/rccl_tuner_gfx950.csv '
        f'{CANDIDATE_TUNER_PATH}; \\\n'
        f'    sha256sum {CANDIDATE_TUNER_PATH} > /opt/RCCL_TUNER_SHA256.txt; \\\n'
        '    echo "tuner policy: removed $(sort -u /opt/RCCL_TUNER_REMOVED.txt | wc -l) '
        f'disk override(s); installed {CANDIDATE_TUNER_PATH}"; \\\n'
    )
    return f"""\
RUN set -e; \\
    SRC=/tmp/rccl_ci/librccl.so; \\
    [ -s "$SRC" ] || {{ echo "GATE FAIL: no librccl staged at $SRC"; exit 1; }}; \\
    canon_src="$(readlink -f "$SRC")"; \\
    find / -xdev -type f -name 'librccl.so*' -not -path '/tmp/rccl_ci/*' \\
      2>/dev/null > /opt/RCCL_TARGETS.txt || true; \\
    [ -s /opt/RCCL_TARGETS.txt ] || {{ echo "GATE FAIL: no librccl found in base image"; exit 1; }}; \\
    : > /opt/RCCL_PRESWAP.txt; \\
    mkdir -p /tmp/rcclswap; n=0; \\
    while read -r t; do \\
      n=$((n+1)); \\
      ldd "$t" 2>/dev/null | awk '/not found/ {{print $1}}' | sort -u > "/tmp/rcclswap/pre.$n"; \\
      printf '%s\\t%s\\n' "$n" "$t" >> /opt/RCCL_PRESWAP.txt; \\
    done < /opt/RCCL_TARGETS.txt; \\
    while read -r t; do \\
      [ "$(readlink -f "$t")" = "$canon_src" ] && continue; \\
      rp="$(patchelf --print-rpath "$t" 2>/dev/null || true)"; \\
      if readelf -d "$t" 2>/dev/null | grep -qF '(RPATH)'; then tag=--force-rpath; else tag=; fi; \\
      echo "  overwrite: $t (rpath: ${{rp:-none}}${{tag:+, kept as DT_RPATH}})"; \\
      cp -fL --remove-destination "$SRC" "$t"; \\
      if [ -n "$rp" ]; then patchelf $tag --set-rpath "$rp" "$t" || \\
        {{ echo "GATE FAIL: could not restore RUNPATH on $t"; exit 1; }}; \\
      else patchelf --remove-rpath "$t" || \\
        {{ echo "GATE FAIL: could not clear RUNPATH on $t"; exit 1; }}; fi; \\
    done < /opt/RCCL_TARGETS.txt; \\
{kpack_block}{tuner_block}{bridge}    ldconfig || true; \\
    while IFS="$(printf '\\t')" read -r n t; do \\
      if ldd "$t" > /tmp/rcclswap/ldd.$n 2>/dev/null; then lddok=yes; else lddok=NO; fi; \\
      awk '/not found/ {{print $1}}' /tmp/rcclswap/ldd.$n | sort -u > "/tmp/rcclswap/post.$n"; \\
      sed 's/\\.so\\..*/.so/' "/tmp/rcclswap/pre.$n" | sort -u > "/tmp/rcclswap/prefam.$n"; \\
      echo "  unresolved sonames before=$(wc -l < "/tmp/rcclswap/pre.$n") after=$(wc -l < "/tmp/rcclswap/post.$n") ldd_ok=$lddok  $t"; \\
      if [ "$lddok" = NO ]; then \\
        echo "    NOTE: ldd could not read this copy, so the counts above prove nothing"; \\
      fi; \\
      regressed=""; \\
      for s in $(comm -13 "/tmp/rcclswap/pre.$n" "/tmp/rcclswap/post.$n"); do \\
        fam="$(printf '%s' "$s" | sed 's/\\.so\\..*/.so/')"; \\
        if grep -qxF "$fam" "/tmp/rcclswap/prefam.$n"; then \\
          echo "    tolerated: $s (${{fam}}.* was already unresolved here)"; \\
        else regressed="$regressed $s"; fi; \\
      done; \\
      if [ -n "$regressed" ]; then \\
        echo "GATE FAIL: the swap left a library unresolved that resolved before on $t:"; \\
        printf '    %s\\n' $regressed; \\
        echo "  The candidate was installed without the RUNPATH this location needs;"; \\
        echo "  loading it would fail at import time, before any step runs."; exit 1; \\
      fi; \\
    done < /opt/RCCL_PRESWAP.txt; \\
    rm -rf /tmp/rcclswap /tmp/rccl_ci; \\
    echo "SWAP GATE PASS: $(wc -l < /opt/RCCL_TARGETS.txt) librccl replaced, no new unresolved sonames"
"""


# Both MaxText images share everything except the RCCL swap, so the baseline
# is this prefix alone and the candidate is this plus the swap layer.
MAXTEXT_PRIMUS_PAYLOAD = """\
# patchelf carries each replaced library's own RUNPATH across the swap;
# binutils/ldd back the soname gate. Installed in the baseline too, so the two
# images differ by the swap and nothing else.
RUN apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=5 update \\
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
      binutils libatomic1 patchelf \\
 && rm -rf /var/lib/apt/lists/*

# The base may ship /workspace/Primus as a git clone, and COPY cannot replace
# a .git directory with a plain tree.
RUN rm -rf /workspace/Primus
COPY Primus/ /workspace/Primus/

# Gated on primus-cli, NOT examples/run_pretrain.sh as MAD#228's dockerfile
# does: AMD-AGI/Primus#999 retired that file and scripts/jax-maxtext/run.sh
# drives `primus-cli direct` instead.
RUN test -f /workspace/Primus/primus-cli
RUN test -f /workspace/Primus/requirements-jax.txt

# Prove the base really carries MaxText, so a wrong base fails the build
# instead of step 0 of a 4-node training run.
RUN test -f /workspace/maxtext/pyproject.toml \\
 || (echo "ERROR: no MaxText at /workspace/maxtext -- wrong base image" >&2 && exit 1)

# Installed here, not per run: run.sh sets PRIMUS_SKIP_PIP=1 so a launch stays
# off the network.
RUN pip3 install --no-cache-dir -r /workspace/Primus/requirements-jax.txt

ENV WORKSPACE_DIR=/workspace
# The Primus repo root, not /workspace: run.sh resolves examples/ against it.
ENV PRIMUS_ROOT=/workspace/Primus
# Pin the base's tree; Primus would otherwise default to
# $PRIMUS_ROOT/third_party/maxtext, which this image does not ship.
ENV MAXTEXT_PATH=/workspace/maxtext
WORKDIR /workspace

# Same label as MAD's primus_maxtext, so madengine treats this image the same.
LABEL mad.launcher=primus
"""


def _stage_primus_context(ctx_dir: Path, primus_dir: Path) -> None:
    """Copy the Primus tree into *ctx_dir* so both image builds can COPY it.

    Copied rather than referenced in place because the manifest advertises the
    Dockerfile to the compute nodes so they can rebuild the image themselves,
    and their build context is this directory. .git is excluded: it is most of
    the tree's size and nothing reads it.
    """
    ctx_primus = ctx_dir / "Primus"
    if ctx_primus.exists():
        return
    log.info("Staging Primus into the build context...")
    ctx_primus.mkdir(parents=True)
    src = subprocess.Popen(
        ["tar", "-C", str(primus_dir), "--exclude=.git", "-cf", "-", "."],
        stdout=subprocess.PIPE,
    )
    subprocess.run(["tar", "-C", str(ctx_primus), "-xf", "-"],
                   stdin=src.stdout, check=True)
    src.stdout.close()
    if src.wait() != 0:
        raise RuntimeError(f"failed to stage Primus from {primus_dir}")


# What `docker pull` accepts, so the digest is the one a pull by tag would
# record in RepoDigests: the index for a multi-arch tag, else the manifest.
_MANIFEST_MEDIA_TYPES = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))


def registry_digest(image: str, timeout: int = 60) -> str:
    """*image* pinned by digest as its registry serves it now, or "" on failure.

    Asks the registry for the manifest digest and downloads no layer: the
    runner host's disk is shared by every runner on it and has no room for a
    base image (the sgl-dev one is 27 GB compressed), and nothing there would
    use it.
    """
    if "@sha256:" in image:
        return image
    name, _, tag = image.rpartition(":")
    if not name or "/" in tag:
        name, tag = image, "latest"
    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        host, path = first, rest
    else:
        host, path = "registry-1.docker.io", name if rest else f"library/{name}"
    url = f"https://{host}/v2/{path}/manifests/{tag}"
    headers = {"Accept": _MANIFEST_MEDIA_TYPES}
    try:
        try:
            req = urllib.request.Request(url, headers=headers, method="HEAD")
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            challenge = exc.headers.get("WWW-Authenticate", "")
            if exc.code != 401 or not challenge.startswith("Bearer "):
                raise
            params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
            realm = params.pop("realm")
            params.setdefault("scope", f"repository:{path}:pull")
            token_url = f"{realm}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(token_url, timeout=timeout) as tok:
                body = json.load(tok)
            headers["Authorization"] = (
                f"Bearer {body.get('token') or body.get('access_token')}"
            )
            req = urllib.request.Request(url, headers=headers, method="HEAD")
            resp = urllib.request.urlopen(req, timeout=timeout)
        with resp:
            digest = resp.headers.get("Docker-Content-Digest", "")
    except (OSError, ValueError, KeyError) as exc:
        log.warning("Could not get the registry digest of %s: %s", image, exc)
        return ""
    return f"{name}@{digest}" if digest.startswith("sha256:") else ""


def resolve_base_digest(base_image: str) -> str:
    """*base_image* pinned by digest, for both A/B Dockerfiles.

    Both A/B images must come from the same base, and a tag cannot promise
    that: each phase builds its own image on the compute nodes, and
    `rocm/jax-training:maxtext-v26.6` is mutable. A digest is immutable, so
    resolving it once and writing it into both Dockerfiles closes it.

    Fails closed when the registry cannot tell: silently using a mutable tag
    would make the candidate image irreproducible and could split the A/B pair.
    """
    digest = registry_digest(base_image)
    if not digest:
        raise RuntimeError(
            f"could not resolve immutable registry digest for {base_image}; "
            "refusing to build either A/B image from a mutable tag"
        )
    log.info("Base pinned for both A/B images: %s", digest)
    return digest


def base_identity(from_ref: str) -> str:
    """Short, stable token for the exact base an A/B image was built from.

    Both MaxText tags carry it so the tag changes whenever the base does.
    Without it the run id alone identified the images, and a workflow *re-run*
    keeps GITHUB_RUN_ID: on the second attempt resolve_base_digest() could pick
    up a moved tag, the baseline would rebuild on it, and the candidate's
    `docker image inspect` would still hit the first attempt's image and skip
    its build. The pair would then straddle two base images while every
    Dockerfile prefix still matched. Compute nodes match on the tag too, so the
    same staleness reaches them.
    """
    return hashlib.sha256(from_ref.encode()).hexdigest()[:10]


def stage_maxtext_baseline(
    base_image: str,
    work_dir: Path,
    primus_dir: Path,
    run_token: str,
    from_ref: str,
) -> str:
    """Stage the A/B baseline for the MaxText family: Primus, stock RCCL.

    The Megatron workloads use the stock registry image as their baseline, but
    that cannot work here: ``rocm/jax-training:maxtext-*`` ships MaxText and no
    Primus, so ``scripts/jax-maxtext/run.sh`` would not find ``primus-cli`` and
    the baseline phase would die before step 0.

    So the baseline is the base image plus the Primus payload and nothing else.
    That makes the pair *tighter* than the Megatron A/B rather than looser: the
    two images differ by the librccl swap alone, which is exactly the variable
    the ratio is supposed to isolate.

    Only the Dockerfile and its context are written here; node 0 of the
    allocation builds the image and the other nodes load it.
    """
    ctx_dir = work_dir / OVERLAY_CTX
    ctx_dir.mkdir(exist_ok=True)
    _stage_primus_context(ctx_dir, primus_dir)

    dockerfile = ctx_dir / MAXTEXT_BASELINE_DOCKERFILE
    dockerfile.write_text(f"FROM {from_ref}\n\n{MAXTEXT_PRIMUS_PAYLOAD}")

    # A per-run tag, with the same token the candidate uses: a node holding an
    # image under a reused tag would skip the build and could pair a fresh
    # candidate with a stale baseline, breaking the one invariant this pair
    # exists to hold -- that the two differ by the RCCL swap alone.
    tag = f"{base_image}-primus-{PRIMUS_REF}-{run_token}-{base_identity(from_ref)}"
    log.info("MaxText baseline (Primus, stock RCCL), built on the nodes: %s", tag)
    return tag


# MAD's Dockerfile ends with `dpkg -r --force-all` on the distro rdma-core,
# which leaves apt unusable, so the swap's patchelf has to come from pip.
# Pinned to the version the MaxText swap gets from apt: the 0.19.1 the
# rocm10 sgl-dev venv ships refuses to set a RUNPATH on the kpack-split
# librccl ("section content is not naturally aligned").
SGLANG_SWAP_TOOLS = "RUN pip install --no-cache-dir patchelf==0.18.0.0\n"


def sglang_recipe(mad_dir: Path, from_ref: str) -> str:
    """MAD's sglang Dockerfile on *from_ref*, plus what the RCCL swap needs.

    The base becomes a literal FROM rather than the ARG default, so no build
    argument can move it off the digest both A/B images were pinned to.
    """
    text = (mad_dir / SGLANG_MAD_DOCKERFILE).read_text()
    pinned, count = re.subn(
        r"(?m)^FROM \$\{?BASE_DOCKER\}?\s*$", f"FROM {from_ref}", text
    )
    if count != 1:
        raise RuntimeError(
            f"{SGLANG_MAD_DOCKERFILE} at MAD {MAD_REF[:12]} has {count} "
            f"`FROM $BASE_DOCKER` lines, expected 1"
        )
    return f"{pinned}\n{SGLANG_SWAP_TOOLS}"


def stage_sglang_baseline(
    base_image: str,
    work_dir: Path,
    run_token: str,
    from_ref: str,
) -> str:
    """Stage the A/B baseline for sglang-disagg: MAD's image, stock RCCL.

    The base alone is not what MAD runs: its Dockerfile adds rdma-core 63,
    without which queue-pair creation fails on Thor2. Built on the nodes and
    tagged like the MaxText baseline, for the same reasons.
    """
    ctx_dir = work_dir / OVERLAY_CTX
    ctx_dir.mkdir(exist_ok=True)
    dockerfile = ctx_dir / SGLANG_BASELINE_DOCKERFILE
    dockerfile.write_text(sglang_recipe(work_dir / "MAD", from_ref))

    tag = f"{base_image}-mad-{MAD_REF[:8]}-{run_token}-{base_identity(from_ref)}"
    log.info("sglang baseline (MAD recipe, stock RCCL), built on the nodes: %s", tag)
    return tag


def stage_rccl_overlay(
    rccl_lib: Path,
    tuner_config: Path,
    base_image: str,
    gpu_target: str,
    work_dir: Path,
    family: str = "megatron",
    primus_dir: Path | None = None,
    from_ref: str = "",
) -> str:
    """Stage the candidate image (base + CI-built RCCL) under ``overlay_ctx``.

    Returns the tag. Nothing is built here; the allocation's nodes build it.

    *family* selects the recipe. ``megatron`` keeps the two layout-specific
    branches that target the Primus image; ``maxtext`` layers the same Primus
    payload as the baseline and then a discovery-based swap, because its base
    has neither Primus nor the SDK venv paths those branches write to.
    """
    rccl_commit = get_rccl_commit(rccl_lib)
    asset_id = rccl_asset_identity(rccl_lib, tuner_config)
    tag = f"{base_image}-rccl-{gpu_target}-{rccl_commit}-{asset_id}"
    if family in ("maxtext", "sglang-disagg") and from_ref:
        # Same identity the baseline carries, for the same reason: the run id
        # alone survives a re-run, so without this the candidate could hit the
        # previous attempt's image and skip its build while the baseline was
        # rebuilt on a base that had moved.
        tag = f"{tag}-{base_identity(from_ref)}"

    ctx_dir = work_dir / OVERLAY_CTX
    ctx_dir.mkdir(exist_ok=True)
    dockerfile = ctx_dir / OVERLAY_DOCKERFILE
    rccl_lib_dir = rccl_lib.parent
    uses_kpack = _rccl_uses_kpack(rccl_lib)

    staging_dir = ctx_dir / "rccl_libs"
    staging_dir.mkdir(exist_ok=True)
    for so_file in rccl_lib_dir.glob("librccl*"):
        dest = staging_dir / so_file.name
        if not dest.exists():
            subprocess.run(["cp", "-L", str(so_file), str(dest)], check=True)
    tuner_staging = staging_dir / "tuner" / tuner_config.name
    tuner_staging.parent.mkdir(exist_ok=True)
    shutil.copy2(tuner_config, tuner_staging)

    # The CI RCCL is built against a newer ROCm than the base image, so it can
    # name a soname the image predates -- `libamd_smi.so.27` against its `.26` --
    # and then fail to load at all. Reconciled in the image by symlink, the way
    # reconcile_soname_versions() does it for the bind-mount path; copying the
    # artifact's own copy instead would drag in the rocm_sysdeps that
    # quarantine_rocm_sysdeps() exists to keep out of the loader path.
    rccl_needed = " ".join(_needed_sonames(rccl_lib))
    if rccl_needed:
        log.info("RCCL needs: %s", rccl_needed)
    bridge_sdk = _soname_bridge('"$SDK_LIB" "$SDK_DEV"', rccl_needed)
    bridge_dep = _soname_bridge('"$DEP_DIR"', rccl_needed)

    # Staged for every family: the maxtext swap places them next to each
    # target it discovers rather than into two known directories.
    has_kpack_files = False
    if uses_kpack:
        kpack_files = rccl_kpack_files(rccl_lib)
        has_kpack_files = len(kpack_files) > 0
        if has_kpack_files:
            kpack_staging = staging_dir / ".kpack"
            kpack_staging.mkdir(exist_ok=True)
            for kp in kpack_files:
                dest = kpack_staging / kp.name
                if not dest.exists():
                    subprocess.run(["cp", "-L", str(kp), str(dest)], check=True)
            log.info("Found %d kpack file(s): %s",
                     len(kpack_files), [f.name for f in kpack_files])
        else:
            raise RuntimeError(
                f"{rccl_lib} has kpack references but its artifact contains no .kpack files"
            )

    if family == "maxtext":
        if primus_dir is None or not (primus_dir / "primus-cli").is_file():
            raise RuntimeError(
                f"maxtext overlay needs a Primus checkout with primus-cli; got {primus_dir!r}"
            )
        _stage_primus_context(ctx_dir, primus_dir)
        # Bridge beside every copy of the library itself, not beside librccl --
        # the directory an RPATH reaches for it is usually neither.
        swap = _librccl_swap_snippet(
            bridge=_soname_bridge_discovered(rccl_needed),
            kpack=has_kpack_files,
        )
        # The payload is byte-identical to the baseline's, so those layers come
        # straight out of the cache and only the swap actually builds here.
        # Staged AFTER it so anything those layers drop on disk -- a wheel
        # vendoring its own librccl, say -- is caught by the sweep too.
        # The same pinned digest the baseline used. A tag here would let the
        # two images build from different versions of a mutable base and the
        # ratio would then be measuring more than RCCL.
        dockerfile.write_text(
            f"FROM {from_ref or base_image}\n\n{MAXTEXT_PRIMUS_PAYLOAD}\n"
            f"COPY rccl_libs/ /tmp/rccl_ci/\n{swap}\n"
            f"ENV NCCL_TUNER_CONFIG_FILE={CANDIDATE_TUNER_PATH}\n"
            "ENV NCCL_DEBUG=WARN\n"
        )
    elif family == "sglang-disagg":
        # The baseline's recipe plus the swap, as for MaxText: the sgl-dev
        # base does not have the Primus image's layout, so the targets are
        # discovered rather than assumed.
        swap = _librccl_swap_snippet(
            bridge=_soname_bridge_discovered(rccl_needed),
            kpack=has_kpack_files,
        )
        dockerfile.write_text(
            f"{sglang_recipe(work_dir / 'MAD', from_ref or base_image)}\n"
            f"COPY rccl_libs/ /tmp/rccl_ci/\n{swap}\n"
            f"ENV NCCL_TUNER_CONFIG_FILE={CANDIDATE_TUNER_PATH}\n"
            "ENV NCCL_DEBUG=WARN\n"
        )
    elif uses_kpack:
        log.info(
            "CI-built librccl.so uses kpack (%.1f MB .so). "
            "Building overlay with SDK venv layout for %s.",
            rccl_lib.stat().st_size / 1e6,
            base_image,
        )
        dockerfile.write_text(f"""\
FROM {base_image}
COPY rccl_libs/ /tmp/rccl_ci/
RUN set -e; \\
    SDK_LIB="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"; \\
    SDK_DEV="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib"; \\
    SDK_KPACK_LIB="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/.kpack"; \\
    SDK_KPACK_DEV="/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/.kpack"; \\
    cp /tmp/rccl_ci/librccl.so "$SDK_LIB/librccl.so.1"; \\
    cp /tmp/rccl_ci/librccl.so "$SDK_LIB/librccl.so.1.0" 2>/dev/null || true; \\
    cp /tmp/rccl_ci/librccl.so "$SDK_DEV/librccl.so.1"; \\
    cp /tmp/rccl_ci/librccl.so "$SDK_DEV/librccl.so.1.0"; \\
    ls /tmp/rccl_ci/.kpack/*.kpack >/dev/null 2>&1 || \\
      {{ echo "GATE FAIL: candidate librccl requires kpack but none was staged"; exit 1; }}; \\
    for d in "$SDK_KPACK_LIB" "$SDK_KPACK_DEV"; do \\
      mkdir -p "$d"; rm -f "$d"/rccl*.kpack; \\
      cp /tmp/rccl_ci/.kpack/*.kpack "$d/"; \\
      for src in /tmp/rccl_ci/.kpack/*.kpack; do \\
        cmp -s "$src" "$d/$(basename "$src")" || \\
          {{ echo "GATE FAIL: kpack copy mismatch in $d"; exit 1; }}; \\
      done; \\
    done; \\
    sha256sum /tmp/rccl_ci/.kpack/*.kpack > /opt/RCCL_KPACK_SHA256S.txt; \\
    find "$SDK_LIB/tuner" "$SDK_DEV/tuner" /opt/rocm*/share/rccl/tuner \\
      -maxdepth 1 -name 'rccl_tuner*.csv' -print -delete \\
      2>/dev/null > /opt/RCCL_TUNER_REMOVED.txt || true; \\
    install -D -m 0444 /tmp/rccl_ci/tuner/rccl_tuner_gfx950.csv {CANDIDATE_TUNER_PATH}; \\
    sha256sum {CANDIDATE_TUNER_PATH} > /opt/RCCL_TUNER_SHA256.txt; \\
    echo "tuner policy: disk overrides removed; installed {CANDIDATE_TUNER_PATH}"; \\
{bridge_sdk}    rm -rf /tmp/rccl_ci
ENV NCCL_TUNER_CONFIG_FILE={CANDIDATE_TUNER_PATH}
ENV NCCL_DEBUG=WARN
""")
    else:
        log.info(
            "CI-built librccl.so has embedded GPU kernels (%.1f MB)",
            rccl_lib.stat().st_size / 1e6,
        )
        dockerfile.write_text(f"""\
FROM {base_image}
COPY rccl_libs/ /tmp/rccl_ci/
RUN set -e; \\
    RCCL_REAL=$(readlink -f /opt/rocm/lib/librccl.so 2>/dev/null || \\
                find /opt/rocm*/lib -name 'librccl.so.*.*' -not -type l 2>/dev/null | head -1); \\
    cp /tmp/rccl_ci/librccl.so "$RCCL_REAL"; \\
    DEP_DIR=$(dirname "$RCCL_REAL"); \\
    find "$DEP_DIR/tuner" "$DEP_DIR/../share/rccl/tuner" /opt/rocm*/share/rccl/tuner \\
      -maxdepth 1 -name 'rccl_tuner*.csv' -print -delete \\
      2>/dev/null > /opt/RCCL_TUNER_REMOVED.txt || true; \\
    install -D -m 0444 /tmp/rccl_ci/tuner/rccl_tuner_gfx950.csv {CANDIDATE_TUNER_PATH}; \\
    sha256sum {CANDIDATE_TUNER_PATH} > /opt/RCCL_TUNER_SHA256.txt; \\
    echo "tuner policy: disk overrides removed; installed {CANDIDATE_TUNER_PATH}"; \\
{bridge_dep}    rm -rf /tmp/rccl_ci
ENV NCCL_TUNER_CONFIG_FILE={CANDIDATE_TUNER_PATH}
ENV NCCL_DEBUG=WARN
""")

    log.info("Candidate (CI RCCL), built on the nodes: %s", tag)
    return tag


def slurm_time_seconds(limit: str) -> int:
    """Seconds in a SLURM ``HH:MM:SS`` time limit."""
    hours, minutes, seconds = (int(part) for part in limit.split(":"))
    return (hours * 60 + minutes) * 60 + seconds


def generate_manifest(
    workload_name: str,
    workload_config: dict,
    cluster_config: dict,
    overlay_image: str,
    nodes: int,
    work_dir: Path,
    rccl_lib: Path | None = None,
    run_dir: Path | None = None,
    pull_only: bool = False,
    dockerfile_name: str = OVERLAY_DOCKERFILE,
) -> Path:
    """Generate a madengine manifest.json for the workload.

    Structure follows the reference template from the mad-rccl branch:
    deployment config under ``deployment_config``, env vars inside both
    ``context.docker_env_vars`` and ``deployment_config.env_vars``, mounts
    in ``context.docker_mounts``.

    *run_dir*: madengine's working directory for this phase; gets the manifest.
    *pull_only*: a stock registry image, pulled per node rather than built.
    """
    gpus_per_node = workload_config["gpus_per_node"]

    nccl_env = dict(cluster_config.get("nccl_env", {}))
    if nodes == 1:
        nccl_env.pop("NCCL_NET", None)
        ifname = nccl_env.get("NCCL_SOCKET_IFNAME", "")
        if "," in ifname:
            nccl_env["NCCL_SOCKET_IFNAME"] = ifname.split(",")[0]

    # Last over everything below, socket interfaces included.
    workload_env = workload_config.get("env_vars", {})
    socket_ifname = workload_env.get(
        "NCCL_SOCKET_IFNAME", nccl_env.get("NCCL_SOCKET_IFNAME", "")
    )

    # HF_TOKEN is passed via MAD_SECRETS_HFTOKEN in the process environment
    # (set in the workflow). Do NOT write it into the manifest — the manifest
    # is uploaded as a CI artifact and would leak the credential.

    model_repo = workload_config["model_repo"]
    family = workload_config.get("family", "megatron")
    if family == "maxtext":
        scripts_dir = work_dir / "scripts" / "jax-maxtext"
    elif family == "sglang-disagg":
        scripts_dir = work_dir / "scripts" / "sglang_disagg"
    else:
        scripts_dir = work_dir / "scripts" / "primus_megatron-lm"
        if not scripts_dir.is_dir():
            scripts_dir = work_dir / "scripts" / "primus" / "megatron-lm"

    # Megatron's run.sh selects the model by repo name; MaxText's takes the
    # Primus YAML plus any `key=value` overrides, which it strips from the
    # --config_path position and forwards to primus-cli.
    if family == "maxtext":
        model_args = f"--config_path {workload_config['primus_config']}"
        extra_args = workload_config.get("model_args_extra", "")
        if extra_args:
            model_args += f" {extra_args}"
    elif family == "sglang-disagg":
        model_args = workload_config["model_args"]
    else:
        model_args = f"--model_repo {model_repo}"

    image_key = "overlay"
    gpu_indices = ",".join(str(i) for i in range(gpus_per_node))
    render_ds = [128 + i for i in range(gpus_per_node)]

    docker_env_vars = {
        **nccl_env,
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_DISABLE": "0",
        "NCCL_TIMEOUT": "900",
        "IBV_SHOW_WARNINGS": "1",
        **workload_env,
    }
    if socket_ifname and "GLOO_SOCKET_IFNAME" not in workload_env:
        docker_env_vars["GLOO_SOCKET_IFNAME"] = socket_ifname

    docker_mounts = dict(workload_config.get("docker_mounts", {}))
    docker_run_opts = workload_config.get("docker_run_options", "")

    # Mount the host's rdma-core stack into the container.
    # The host's libibverbs has a compiled-in provider search path of
    # /usr/lib64/libibverbs/ so the providers must appear there.
    # The library itself replaces the container's copy so the linker
    # picks it up from the standard search path.
    if cluster_config.get("mount_host_ib_libs") and workload_config.get(
        "mount_host_ib_libs", True
    ):
        docker_run_opts += (
            " -v /usr/lib64/libibverbs.so.1"
            ":/usr/lib/x86_64-linux-gnu/libibverbs.so.1:ro"
            " -v /usr/lib64/libibverbs:/usr/lib64/libibverbs:ro"
            " -v /usr/lib64/libibumad.so.3"
            ":/usr/lib/x86_64-linux-gnu/libibumad.so.3:ro"
        )

    if family == "sglang-disagg":
        # The launcher writes its server logs, and the RCCL banner the
        # candidate is verified by, to /run_logs/$SLURM_JOB_ID. Both phases
        # share one job id, so each gets its own directory. It must be shared
        # storage and world-writable: Docker runs as root, which a
        # root-squashed NFS maps to nobody.
        run_logs = (run_dir or work_dir) / "slurm_output" / "run_logs"
        run_logs.mkdir(parents=True, exist_ok=True)
        run_logs.chmod(0o777)
        docker_run_opts += f" -v {run_logs}:/run_logs"
        # Read-only: run.sh downloads the weights into MODEL_PATH when they
        # look incomplete, and that must fail rather than fill shared storage.
        docker_run_opts += f" -v {cluster_config['models_dir']}:/models:ro"

    # When --skip-overlay-build is used the base image still has its
    # bundled RCCL.  Bind-mount the CI-built librccl.so (and kpack
    # files if present) over the container's copies so we actually
    # test the artifact, not the image default.
    if rccl_lib is not None:
        # These are the Primus image's SDK venv paths. On the MaxText base they
        # do not exist, and `docker run -v` CREATES a missing source as an empty
        # root-owned directory rather than refusing -- so the run would look
        # healthy while measuring the image's own RCCL. Refuse instead.
        if family in ("maxtext", "sglang-disagg"):
            raise RuntimeError(
                f"--skip-overlay-build is not supported for the {family} family: "
                "the bind-mount targets are Primus-image paths absent from "
                + workload_config["base_image"] + ", and Docker would silently "
                "mount empty directories over nothing. Build the overlay."
            )
        host_so = str(rccl_lib.resolve())
        sdk_lib = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
        sdk_dev = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib"
        docker_run_opts += (
            f" -v {host_so}:{sdk_lib}/librccl.so.1:ro"
            f" -v {host_so}:{sdk_lib}/librccl.so.1.0:ro"
            f" -v {host_so}:{sdk_dev}/librccl.so.1:ro"
            f" -v {host_so}:{sdk_dev}/librccl.so.1.0:ro"
        )
        kpack_dir = rccl_lib.resolve().parent.parent / ".kpack"
        if not kpack_dir.is_dir():
            kpack_dir = rccl_lib.resolve().parent / ".kpack"
        if kpack_dir.is_dir():
            sdk_kpack = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/.kpack"
            for kp in kpack_dir.glob("rccl*.kpack"):
                docker_run_opts += f" -v {kp}:{sdk_kpack}/{kp.name}:ro"
            log.info("Bind-mounting RCCL kpack from %s", kpack_dir)
        log.info("Bind-mounting CI-built RCCL: %s", host_so)

    slurm_config = {
        "partition": cluster_config.get("slurm_partition", workload_config["slurm_partition"]),
        "qos": cluster_config.get("slurm_qos", ""),
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "time": workload_config["time_limit"],
        "output_dir": "./slurm_output",
        "exclusive": True,
        "enable_node_check": False,
        "network_interface": socket_ifname,
        "skip_gpus_directive": cluster_config.get("slurm_no_gres", False),
    }

    overlay_dockerfile = work_dir / OVERLAY_CTX / dockerfile_name

    if pull_only:
        image_entry = {
            "docker_image": overlay_image,
            "local_image": False,
            "registry_image": overlay_image,
            "registry": None,
            "base_docker": overlay_image,
            "build_status": "SKIPPED",
            "build_duration": 0,
            "gpu_vendor": "AMD",
        }
    else:
        image_entry = {
            "docker_image": overlay_image,
            "local_image": True,
            "registry_image": None,
            "registry": None,
            "base_docker": workload_config["base_image"],
            "build_status": "SKIPPED",
            "build_duration": 0,
            "gpu_vendor": "AMD",
            # Lets a compute node build the image it can neither find nor
            # pull; the context is this file's own directory, overlay_ctx.
            # Advertised only when it exists: --skip-overlay-build writes no
            # Dockerfile, and a dead path would turn madengine's fallback
            # into a guaranteed failure.
            **(
                {"dockerfile": str(overlay_dockerfile)}
                if overlay_dockerfile.is_file()
                else {}
            ),
        }

    manifest = {
        "built_images": {image_key: image_entry},
        "built_models": {
            image_key: {
                "name": model_repo,
                "tags": workload_config.get("tags", ["pyt", "pretrain", "training"]),
                "dockerfile": "N/A (overlay image)",
                "scripts": f"scripts/{scripts_dir.name}/run.sh",
                "n_gpus": "-1",
                "owner": "",
                "training_precision": workload_config.get("training_precision", ""),
                "multiple_results": workload_config.get("multiple_results", ""),
                "args": model_args,
                "additional_docker_run_options": docker_run_opts,
                "data": "",
                "cred": "",
                # None falls back to madengine's 2h default, which would stop a
                # phase early; the allocation's own limit is the real cap.
                "timeout": slurm_time_seconds(workload_config["time_limit"]),
            },
        },
        "context": {
            "gpu_vendor": "AMD",
            "guest_os": "UBUNTU",
            "docker_gpus": gpu_indices,
            "gpu_renderDs": render_ds,
            "docker_env_vars": docker_env_vars,
            "docker_mounts": docker_mounts,
            "docker_build_arg": {},
        },
        "deployment_config": {
            "target": "slurm",
            "slurm": slurm_config,
            "distributed": {
                "launcher": "sglang-disagg" if family == "sglang-disagg" else "primus",
                "backend": "nccl",
                "port": 29500,
                "nnodes": nodes,
                # 8 even for JAX. Do not "fix" this to 1 on the theory that
                # JAX runs one process per node: MAD#228's template is from
                # validated 4N runs and uses 8.
                "nproc_per_node": gpus_per_node,
                # MaxText needs the launcher told which Primus config and
                # backend to drive; the Megatron path carries neither.
                **(
                    {
                        "primus": {
                            "config_path": workload_config["primus_config"],
                            "backend": "MaxText",
                        }
                    }
                    if family == "maxtext"
                    else {}
                ),
                **(
                    {"sglang_disagg": workload_config["disagg"]}
                    if family == "sglang-disagg"
                    else {}
                ),
            },
            "env_vars": {
                **docker_env_vars,
                "OMP_NUM_THREADS": "8",
                # PyTorch/MIOpen knobs: inert under JAX, and leaving them out
                # of the MaxText manifest avoids implying they do something.
                **(
                    {}
                    if family == "maxtext"
                    else {
                        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                        "TORCH_NCCL_HIGH_PRIORITY": "1",
                        "MIOPEN_FIND_MODE": "1",
                    }
                ),
            },
            "debug": False,
            "docker_gpus": gpu_indices,
        },
    }

    manifest_path = (run_dir or work_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written to: %s", manifest_path)
    return manifest_path


def prepare_phase_dir(work_dir: Path, phase: str) -> Path:
    """Create one phase's madengine working directory, with its own scripts copy.

    madengine writes its outputs under fixed names relative to its cwd.

    Recreated from scratch: on a reused --work-dir an earlier run's exit_code,
    perf files and node logs would otherwise still be there, and an allocation
    that never ran a phase would be scored on them as if it had. Fails if the
    old directory cannot be removed rather than run on top of it.
    """
    run_dir = work_dir / "ab" / phase
    if run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except OSError as exc:
            raise RuntimeError(
                f"{run_dir} holds an earlier run's results and cannot be removed "
                f"({exc}); use a fresh --work-dir"
            ) from exc
    run_dir.mkdir(parents=True)

    scripts_src = work_dir / "scripts"
    scripts_dst = run_dir / "scripts"
    if scripts_src.is_dir() and not scripts_dst.exists():
        subprocess.run(["cp", "-r", str(scripts_src), str(scripts_dst)], check=True)

    return run_dir


# Uploaded as its own artifact so a night's result can be rerun by hand: the
# phase manifests alone name a local-only image whose build context lives in
# WORK_DIR, which the cleanup step deletes.
REPRO_DIR = "repro"
REPRO_MANIFESTS = {
    CANDIDATE: "manifest.repro.json",
    BASELINE: "manifest.baseline.repro.json",
}
# TheRock's bucket for runs of a ROCm/* repo that are not from a fork.
THEROCK_ARTIFACT_BUCKET = "therock-ci-artifacts-external"


def repro_manifest(manifest: dict, work_dir: Path) -> dict:
    """*manifest* with its build paths made relative to the bundle root.

    The CI manifest names the Dockerfile by an absolute path under WORK_DIR and
    gives no build context, so madengine falls back to ``./docker`` whenever the
    invocation directory has one -- which a MAD ``MODEL_DIR`` checkout does, and
    the build then fails on ``COPY rccl_libs/``. In the bundle both point into its
    own ``overlay_ctx/``; madengine resolves them against the directory it runs
    from, and on every node against the node-local copy of that directory.
    """
    out = copy.deepcopy(manifest)
    ctx = work_dir / OVERLAY_CTX
    for key, image in out.get("built_images", {}).items():
        dockerfile = image.get("dockerfile")
        if not dockerfile:
            continue
        try:
            rel = Path(dockerfile).relative_to(ctx)
        except ValueError:
            continue
        image["dockerfile"] = f"{OVERLAY_CTX}/{rel.as_posix()}"
        model = out.get("built_models", {}).get(key)
        if model is not None:
            model["dockercontext"] = OVERLAY_CTX
    # The CI phase directory is gone by then; a rerun salloc's from its copy.
    for model in out.get("built_models", {}).values():
        if "additional_docker_run_options" in model:
            model["additional_docker_run_options"] = re.sub(
                r"-v \S+/slurm_output/run_logs:/run_logs",
                "-v ${SLURM_SUBMIT_DIR}/slurm_output/run_logs:/run_logs",
                model["additional_docker_run_options"],
            )
    return out


def copy_overlay_ctx(src: Path, dst: Path) -> None:
    """Copy the build context without the duplicate librccl names.

    stage_rccl_overlay() stages every ``librccl*`` the artifact carries,
    but each recipe reads only ``rccl_libs/librccl.so`` and ``rccl_libs/.kpack``,
    and the other names are byte-identical copies of the same library.
    """
    lib = src / "rccl_libs" / "librccl.so"

    def ignore(dirpath: str, names: list[str]) -> list[str]:
        if Path(dirpath) != lib.parent or not lib.is_file():
            return []
        return [
            n for n in names
            if n.startswith("librccl") and n != lib.name
            and (lib.parent / n).is_file()
            and filecmp.cmp(lib, lib.parent / n, shallow=False)
        ]

    shutil.copytree(src, dst, symlinks=True, ignore=ignore)


def repro_env(
    workload_name: str,
    workload_config: dict,
    cluster_config: dict,
    nodes: int,
    rccl_fingerprint: dict,
    images: dict[str, str],
    base_digest: str,
    env: dict,
) -> dict[str, str]:
    """Provenance of this run, plus the cluster setting the manifest cannot carry."""
    repo = env.get("GITHUB_REPOSITORY", "")
    run_id = env.get("GITHUB_RUN_ID", "")
    artifact_run_id = env.get("ARTIFACT_RUN_ID", "") or run_id
    s3_prefix = (
        f"{repo.replace('/', '-')}/{artifact_run_id}-linux/"
        if repo and artifact_run_id else ""
    )
    family = workload_config.get("family", "megatron")
    values = {
        "WORKLOAD": workload_name,
        # REPRO_*, not SLURM_*: srun reads SLURM_PARTITION and friends as input.
        "REPRO_NODES": str(nodes),
        "CI_RUN_URL": (
            f"{env.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/actions/runs/{run_id}"
            if repo and run_id else ""
        ),
        "CI_RUN_ID": run_id,
        "CI_RUN_ATTEMPT": env.get("GITHUB_RUN_ATTEMPT", ""),
        "CI_SHA": env.get("GITHUB_SHA", ""),
        "CANDIDATE_SHA": env.get("CANDIDATE_SHA", ""),
        "THEROCK_ARTIFACT_RUN_ID": artifact_run_id,
        # Provenance only: the library itself is in overlay_ctx/rccl_libs.
        "RCCL_ARTIFACTS_S3": f"s3://{THEROCK_ARTIFACT_BUCKET}/{s3_prefix}" if s3_prefix else "",
        "RCCL_ARTIFACTS_URL": (
            f"https://{THEROCK_ARTIFACT_BUCKET}.s3.amazonaws.com/{s3_prefix}" if s3_prefix else ""
        ),
        "RCCL_MD5": rccl_fingerprint.get("md5", ""),
        "RCCL_SHA256": rccl_fingerprint.get("sha256", ""),
        "RCCL_VERSION": rccl_fingerprint.get("version", ""),
        "CANDIDATE_IMAGE": images.get(CANDIDATE, ""),
        "BASELINE_IMAGE": images.get(BASELINE, ""),
        "BASE_IMAGE": workload_config["base_image"],
        "BASE_DIGEST": base_digest,
        "MADENGINE_REPO": MADENGINE_REPO,
        "MADENGINE_REF": MADENGINE_REF,
        "MAD_REPO": MAD_REPO,
        "MAD_BRANCH": MAD_BRANCH,
        "MAD_REF": MAD_REF,
        "PRIMUS_REF": PRIMUS_REF if family == "maxtext" else "",
        "REPRO_PARTITION": cluster_config.get(
            "slurm_partition", workload_config["slurm_partition"]
        ),
        "REPRO_TIME_LIMIT": workload_config["time_limit"],
        # ruby's per-job /tmp is invisible to the Docker daemon; the workflow
        # sets this, and a run by hand has to as well.
        "SLURM_TMPDIR": env.get("SLURM_TMPDIR", ""),
    }
    return {k: v for k, v in values.items() if v}


REPRO_README = """\
# Reproduce `{WORKLOAD}` from CI run {CI_RUN_ID}

{CI_RUN_URL}

Everything the run needs is in this directory: the image's build context with
the CI-built RCCL (`overlay_ctx/`), the MAD model scripts (`scripts/`), and one
manifest per A/B phase with paths relative to this directory. `repro.env` holds
the refs, image digests and cluster settings. The run's own numbers are in the
`madengine-results-*` artifact of the same CI run.

1. Unpack this directory on storage that every compute node sees, `cd` into it,
   and `source repro.env`.
2. In a Python 3.10+ environment the nodes can also see (conda on shared storage):
   `pip install "git+$MADENGINE_REPO@$MADENGINE_REF"`
3. `export MAD_DOCKER_BUILDS=<shared dir>` and `chmod 0777` it: node 0 saves the
   image there for the others through `sudo docker`, which a root-squashed NFS
   maps to `nobody`.
4. `salloc -p "$REPRO_PARTITION" -N "$REPRO_NODES" --exclusive -t "$REPRO_TIME_LIMIT"`, then inside it:
   `madengine run --manifest-file manifest.repro.json --live-output`
   madengine runs in place in the allocation; node 0 builds the image from
   `overlay_ctx/` and the other nodes load it.

- The baseline phase is `manifest.baseline.repro.json`. Run it from a second copy
  of this directory: each phase writes `perf.csv` and `slurm_output/` to its cwd.
- An sglang-disagg run also needs `mkdir -m 0777 -p slurm_output/run_logs` in
  each copy before `salloc`, and the weights under `/models` as the manifest
  mounts them.
- To shorten a Megatron run, add `"PRIMUS_TRAIN_ITERS": "15"` to both
  `context.docker_env_vars` and `deployment_config.env_vars`.
- `BASE_DIGEST` is what `BASE_IMAGE` pointed to when CI ran. MaxText and
  sglang-disagg images build from that digest; Megatron ones from `BASE_IMAGE`
  by tag, so if the base has moved since, the image differs.
"""


def write_repro_bundle(
    work_dir: Path,
    phase_dirs: dict[str, Path],
    workload_name: str,
    workload_config: dict,
    cluster_config: dict,
    nodes: int,
    rccl_fingerprint: dict,
    images: dict[str, str],
    base_digest: str,
    env: dict | None = None,
) -> Path:
    """Write ``WORK_DIR/repro``: what a rerun by hand needs once WORK_DIR is gone.

    The CI manifests are left exactly as the run uses them; the bundle carries
    rewritten copies.
    """
    bundle = work_dir / REPRO_DIR
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    ctx = work_dir / OVERLAY_CTX
    if ctx.is_dir():
        copy_overlay_ctx(ctx, bundle / OVERLAY_CTX)
    if (work_dir / "scripts").is_dir():
        shutil.copytree(work_dir / "scripts", bundle / "scripts", symlinks=True)
    for name in (CANDIDATE_PROVENANCE, "candidate-artifact-run.json"):
        if (work_dir / name).is_file():
            shutil.copy2(work_dir / name, bundle / name)

    for phase, run_dir in phase_dirs.items():
        manifest = json.loads((run_dir / "manifest.json").read_text())
        (bundle / REPRO_MANIFESTS[phase]).write_text(
            json.dumps(repro_manifest(manifest, work_dir), indent=2) + "\n"
        )

    values = repro_env(
        workload_name, workload_config, cluster_config, nodes,
        rccl_fingerprint, images, base_digest,
        os.environ if env is None else env,
    )
    (bundle / "repro.env").write_text(
        "# Source before running; see REPRO.md.\n"
        + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in values.items())
    )
    readme_values = {"CI_RUN_URL": "", "CI_RUN_ID": "(local run)", **values}
    (bundle / "REPRO.md").write_text(REPRO_README.format(**readme_values))
    log.info("Repro bundle written to: %s", bundle)
    return bundle


# The sglang launcher reaches every node at the address DNS gives its name and
# binds its barrier to the node's own one, so a stale record leaves rank 0
# waiting out BARRIER_TIMEOUT_S for a node that is up and never answers.
PREFLIGHT_DNS = """\
#!/bin/bash
h=$(hostname -s)
ip=$(getent ahostsv4 "$h" | awk 'NR==1 {print $1}')
if [ -n "$ip" ] && hostname -I | tr ' ' '\\n' | grep -qxF "$ip"; then
    echo "  ok   $h -> $ip"
else
    echo "  BAD  $h -> ${ip:-unresolved}, not an address of this node"
    exit 1
fi
"""

# sglang's nixl backend registers every KV-cache buffer with every RDMA NIC,
# and each bnxt_re registration needs physically contiguous host memory (an
# order-9 allocation). On a node whose RAM is nearly all page cache those fail
# -- "bnxt_re3: Failed to register user MR - rc = -12" in dmesg -- and the
# server that hits one dies at startup with NIXL_ERR_BACKEND. The nodes are
# held exclusively, so drop the clean page cache and compact before each phase.
# Best effort: without passwordless sudo the phase runs as before.
RELEASE_HOST_MEMORY = """\
#!/bin/bash
if sudo -n sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'; then
    echo "  ok   $(hostname -s) free=$(free -g | awk '/^Mem:/ {print $4}')G"
else
    echo "  skip $(hostname -s): no passwordless sudo"
fi
exit 0
"""

# Run on every node after both phases. Container output is root-owned, so the
# workspace is removed from inside a local image (never pulled); $1 is kept.
CLEANUP_IMAGES = """\
#!/bin/bash
helper="$1"; shift
if [ -n "${SLURM_JOB_ID}" ]; then
    for ws in "${SLURM_TMPDIR:-/tmp}"/madengine_job_"${SLURM_JOB_ID}"_* \\
              /tmp/madengine_job_"${SLURM_JOB_ID}"_*; do
        [ -d "$ws" ] || continue
        rm -rf "$ws" 2>/dev/null || sudo -n rm -rf "$ws" 2>/dev/null
        for image in "$helper" "$@"; do
            [ -e "$ws" ] || break
            [ -n "$image" ] || continue
            args=(run --rm --pull never -u 0 -v "$(dirname "$ws"):/w"
                  --entrypoint rm "$image" -rf "/w/$(basename "$ws")")
            docker "${args[@]}" >/dev/null 2>&1 \\
                || sudo -n docker "${args[@]}" >/dev/null 2>&1
        done
        if [ -e "$ws" ]; then
            echo "  could not remove $(hostname -s) $ws"
        else
            echo "  removed $(hostname -s) $ws"
        fi
    done
fi
for image in "$@"; do
    if docker image rm -f "$image" >/dev/null 2>&1 \\
        || sudo -n docker image rm -f "$image" >/dev/null 2>&1; then
        echo "  removed $(hostname -s) $image"
    fi
done
exit 0
"""

CANDIDATE_IMAGE_ID = """\
#!/bin/bash
image="$1"
out="$2"
expected_tuner_sha="$3"
host=$(hostname -s)
id=$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null \\
  || sudo -n docker image inspect --format '{{.Id}}' "$image" 2>/dev/null) || {
  echo "GATE FAIL: cannot inspect candidate image $image on $host"
  exit 1
}
case "$id" in
  sha256:[0-9a-f][0-9a-f]*) ;;
  *) echo "GATE FAIL: invalid candidate image id '$id' on $host"; exit 1 ;;
esac
tuner_sha=$(docker run --rm --pull never --entrypoint sha256sum "$image" \\
  /opt/rccl-candidate/tuner/rccl_tuner_gfx950.csv 2>/dev/null \\
  || sudo -n docker run --rm --pull never --entrypoint sha256sum "$image" \\
     /opt/rccl-candidate/tuner/rccl_tuner_gfx950.csv 2>/dev/null) || {
  echo "GATE FAIL: cannot hash candidate tuner config on $host"
  exit 1
}
tuner_sha=${tuner_sha%% *}
if [ "$tuner_sha" != "$expected_tuner_sha" ]; then
  echo "GATE FAIL: candidate tuner hash mismatch on $host: $tuner_sha != $expected_tuner_sha"
  exit 1
fi
printf '%s\\t%s\\t%s\\n' "$host" "$id" "$tuner_sha" > "$out/$host"
echo "  $host candidate image $id tuner_sha256=$tuner_sha"
"""


def verify_image_identities(
    identity_dir: Path,
    nodes: int,
    expected_tuner_sha: str,
) -> tuple[bool, str, str]:
    """Require one identical immutable candidate image ID from every node."""
    rows = []
    for path in sorted(identity_dir.glob("*")) if identity_dir.is_dir() else []:
        try:
            host, image_id, tuner_sha = path.read_text().strip().split("\t", 2)
        except (OSError, ValueError):
            return False, f"unreadable candidate image identity {path}", ""
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            return False, f"invalid candidate image identity from {host}: {image_id!r}", ""
        if tuner_sha != expected_tuner_sha:
            return False, f"candidate tuner hash mismatch from {host}: {tuner_sha!r}", ""
        rows.append((host, image_id))
    if len(rows) != nodes:
        return False, f"candidate image identity found on {len(rows)}/{nodes} node(s)", ""
    ids = {image_id for _, image_id in rows}
    if len(ids) != 1:
        return False, f"candidate image mismatch across nodes: {rows}", ""
    image_id = ids.pop()
    return True, f"candidate image verified on {nodes} node(s): {image_id}", image_id


def _raise_on_term(signum, frame):  # noqa: ARG001 - signal handler signature
    """Make SIGTERM unwind the stack the way SIGINT already does."""
    raise KeyboardInterrupt(f"signal {signum}")


def run_ab_in_one_allocation(
    phases: list[tuple[str, Path]],
    work_dir: Path,
    nodes: int,
    workload_name: str,
    workload_config: dict,
    cluster_config: dict,
    timeout_minutes: int,
    built_images: list[str] | None = None,
    candidate_image: str = "",
    candidate_tuner_sha256: str = "",
) -> dict:
    """Run every phase back to back inside one SLURM allocation, on the same nodes.

    *built_images* are removed from every node afterwards (not on a timeout).

    Returns ``{"exit_codes": {phase: int}, "job_id": str, "nodelist": str}``.
    """
    alloc_info = work_dir / "allocation.txt"
    alloc_info.unlink(missing_ok=True)

    script_lines = [
        "#!/bin/bash",
        "# Generated by test_madengine.py. No `set -e`: a failed baseline must",
        "# still let the candidate run.",
        'echo "Allocation: job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"',
        'printf "%s\\n%s\\n" "${SLURM_JOB_ID}" "${SLURM_JOB_NODELIST}" > '
        + shlex.quote(str(alloc_info)),
        "",
        # Every node must be able to reach the Docker daemon before either
        # phase starts. Not every node in this partition has it up, and the
        # failure is otherwise unrecognisable: the nodes that do have it start
        # their container and sit in the launcher's rendezvous until it times
        # out, so the log blames the distributed runtime --
        #
        #   F client.h:80] Terminating process because the JAX distributed
        #   service detected fatal errors ... DEADLINE_EXCEEDED
        #
        # -- on the healthy nodes, while the real message ("dial unix
        # /var/run/docker.sock: connect: no such file or directory") sits in a
        # different node's log. The wider the job, the likelier it is to catch
        # at least one such node.
        #
        # Try to start it, then check. Reported per node and failed fast, so
        # the allocation is released instead of burning its time limit.
        'echo "===== preflight: docker daemon on every allocated node ====="',
        'srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 '
        '--overlap bash -c \''
        'sudo -n systemctl start docker >/dev/null 2>&1; '
        'if docker info >/dev/null 2>&1 || sudo -n docker info >/dev/null 2>&1; then '
        'echo "  ok   $(hostname -s)"; '
        'else echo "  DOWN $(hostname -s)"; exit 1; fi\' || {',
        '  echo "::error::Docker is unreachable on at least one allocated '
        'node (see DOWN above). Releasing the allocation rather than letting '
        'the launcher time out."',
        "  exit 3",
        "}",
    ]
    if workload_config.get("family") == "sglang-disagg":
        dns_check = work_dir / "ab" / "preflight_dns.sh"
        dns_check.parent.mkdir(parents=True, exist_ok=True)
        dns_check.write_text(PREFLIGHT_DNS)
        script_lines += [
            'echo "===== preflight: every node resolves to an address it owns ====="',
            'srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 '
            f"--overlap bash {shlex.quote(str(dns_check))} || {{",
            '  echo "::error::At least one allocated node resolves to an address '
            'it does not have (see BAD above); the sglang barrier would wait for '
            'it until timeout. Exclude it via MADENGINE_SLURM_EXCLUDE."',
            "  exit 3",
            "}",
        ]
        release_memory = work_dir / "ab" / "release_host_memory.sh"
        release_memory.write_text(RELEASE_HOST_MEMORY)
    image_id_dir = work_dir / "ab" / "candidate-image-ids"
    image_id_dir.mkdir(parents=True, exist_ok=True)
    image_id_script = work_dir / "ab" / "candidate_image_id.sh"
    image_id_script.write_text(CANDIDATE_IMAGE_ID)
    for phase, run_dir in phases:
        script_lines += ["", f'echo "===== phase: {phase} ====="']
        if workload_config.get("family") == "sglang-disagg":
            script_lines.append(
                'srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 '
                f"--overlap bash {shlex.quote(str(release_memory))}"
            )
        script_lines += [
            f"cd {shlex.quote(str(run_dir))} || exit 1",
            "madengine run -m manifest.json -o perf.csv --live-output --verbose",
            "echo $? > " + shlex.quote(str(run_dir / "exit_code")),
        ]
        if phase == CANDIDATE and candidate_image:
            script_lines += [
                'echo "===== verify: immutable candidate image on every node ====="',
                'srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 '
                f"--overlap bash {shlex.quote(str(image_id_script))} "
                f"{shlex.quote(candidate_image)} {shlex.quote(str(image_id_dir))} "
                f"{shlex.quote(candidate_tuner_sha256)} "
                "|| echo 1 > " + shlex.quote(str(image_id_dir / "FAILED")),
            ]
    cleanup = work_dir / "ab" / "cleanup_images.sh"
    cleanup.parent.mkdir(parents=True, exist_ok=True)
    cleanup.write_text(CLEANUP_IMAGES)
    script_lines += [
        "",
        'echo "===== cleanup: this run\'s workspaces and images on every node ====="',
        'srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 '
        f"--overlap bash {shlex.quote(str(cleanup))} "
        + " ".join(
            shlex.quote(i) for i in [workload_config["base_image"], *(built_images or [])]
        )
        + " || true",
    ]

    script_path = work_dir / "ab" / "run_phases.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(script_lines) + "\n")

    # Unique per run, because cancelling by name is the only way to reach a
    # request that is still queued: it has no job id in allocation.txt yet,
    # and SIGKILL, which salloc cannot handle, leaves it queued to start later
    # with nothing attached (InactiveLimit is 0 on ruby). Recorded before
    # salloc starts so the workflow's release step can read it too.
    run_tag = (
        f"{os.environ['GITHUB_RUN_ID']}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
        if os.environ.get("GITHUB_RUN_ID") else f"local-{os.getpid()}"
    )
    job_name = f"rccl-ab-{workload_name}-{run_tag}"
    (work_dir / "allocation.name").write_text(job_name + "\n")

    cmd = [
        "salloc",
        "--nodes", str(nodes),
        "--ntasks-per-node", "1",
        "--exclusive",
        "--partition",
        cluster_config.get("slurm_partition", workload_config["slurm_partition"]),
        "--time", workload_config["time_limit"],
        "--job-name", job_name,
    ]
    qos = cluster_config.get("slurm_qos", "")
    if qos:
        cmd += ["--qos", qos]
    # Set from a repository variable by the workflow: the nodes to avoid are
    # the cluster's business and their names do not belong in this file.
    exclude = os.environ.get("MADENGINE_SLURM_EXCLUDE", "").strip()
    if exclude:
        if not re.fullmatch(r"[A-Za-z0-9_.,\[\]-]+", exclude):
            raise RuntimeError(
                f"MADENGINE_SLURM_EXCLUDE is not a SLURM hostlist: {exclude!r}"
            )
        cmd += ["--exclude", exclude]
    cmd += ["bash", str(script_path)]

    log.info("Running: %s", " ".join(cmd))
    log.info("Phases: %s", ", ".join(p for p, _ in phases))
    log.info("Timeout: %d minutes (includes time spent queueing)", timeout_minutes)

    # Pre-warm the bytecode cache: madengine runs `madengine --version` itself,
    # and a cold import of its dependencies off NFS is slow.
    try:
        subprocess.run(["madengine", "--version"], capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        log.warning("madengine --version pre-warm timed out (non-fatal)")
    except Exception:
        pass

    env = os.environ.copy()
    docker_builds_dir = work_dir / "docker_builds"
    docker_builds_dir.mkdir(exist_ok=True)
    env["MAD_DOCKER_BUILDS"] = str(docker_builds_dir)

    # Own process group: killing salloc alone leaves run_phases.sh and its
    # madengine/srun children running on the allocation.
    proc = subprocess.Popen(cmd, cwd=work_dir, env=env, start_new_session=True)

    def release_allocation(reason: str) -> None:
        """Give the nodes back: cancel the job, then kill the whole group.

        The group is in a session of its own, so nothing the caller is killed
        by reaches it. Whatever ends this function has to do it explicitly or
        the allocation keeps the nodes until its own --time expires -- four
        exclusive ones for five hours, in the 405B case.
        """
        log.error("Releasing the allocation: %s", reason)
        try:
            # By name, not by the id in allocation.txt: the name also reaches
            # a request that is still queued, which has no id there yet.
            try:
                subprocess.run(["scancel", "--me", "--name", job_name], timeout=60)
            except (OSError, subprocess.SubprocessError) as exc:
                log.warning("scancel --name %s failed: %s", job_name, exc)
        finally:
            # A second signal during scancel raises out of this function; the
            # group must still be killed and reaped.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    # Actions cancels a run with SIGINT and escalates to SIGTERM. SIGINT
    # already arrives as KeyboardInterrupt, but SIGTERM ends the interpreter
    # without unwinding, so neither `except` nor `finally` would run: turn it
    # into the same exception. SIGKILL, which follows, cannot be caught at
    # all -- the allocation's own --time is the only backstop left for it.
    previous_term = signal.signal(signal.SIGTERM, _raise_on_term)
    try:
        proc.wait(timeout=timeout_minutes * 60)
    except subprocess.TimeoutExpired:
        release_allocation(f"timed out after {timeout_minutes} minutes")
    except BaseException:
        release_allocation("the driver was interrupted")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)

    exit_codes = {}
    for phase, run_dir in phases:
        marker = run_dir / "exit_code"
        try:
            exit_codes[phase] = int(marker.read_text().strip())
        except (OSError, ValueError):
            # No marker means the phase never ran: salloc never got the nodes,
            # or the timeout above cut it short.
            log.error("Phase %s produced no exit code — it did not run", phase)
            exit_codes[phase] = 124
        else:
            log.info("Phase %s exit code: %d", phase, exit_codes[phase])

    job_id, granted_nodes = "", ""
    try:
        job_id, granted_nodes = alloc_info.read_text().splitlines()[:2]
    except (OSError, ValueError):
        log.warning("Could not read allocation details from %s", alloc_info)
    else:
        log.info("Allocation: job=%s nodes=%s", job_id, granted_nodes)

    if candidate_image:
        image_ok, image_msg, image_id = verify_image_identities(
            image_id_dir, nodes, candidate_tuner_sha256
        )
    else:
        image_ok, image_msg, image_id = True, "candidate image verification not requested", ""
    log.info("Candidate image verification: %s", image_msg)
    return {
        "exit_codes": exit_codes,
        "job_id": job_id,
        "nodelist": granted_nodes,
        "candidate_image_ok": image_ok,
        "candidate_image_message": image_msg,
        "candidate_image_id": image_id,
    }


def parse_perf_results(work_dir: Path) -> list[dict]:
    """Parse madengine performance results.

    Prefers ``perf_entry_super.json`` (31 fixed columns, per-precision rows
    with ``multi_results``).  Falls back to ``perf.csv`` (variable-width,
    long-format).  Returns a list of result dicts — one per row.
    """
    super_json = work_dir / "perf_entry_super.json"
    if super_json.exists():
        try:
            entries = json.loads(super_json.read_text())
            if entries:
                log.info("Parsed %d result(s) from perf_entry_super.json", len(entries))
                for e in entries:
                    log.info("  model=%s perf=%s metric=%s status=%s precision=%s",
                             e.get("model"), e.get("performance"),
                             e.get("metric"), e.get("status"),
                             e.get("training_precision"))
                return entries
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("Could not parse %s: %s", super_json, exc)

    csv_path = work_dir / "perf.csv"
    if csv_path.exists():
        rows = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                log.info("perf.csv row: %s", dict(row))
                rows.append(dict(row))
        if rows:
            log.info("Parsed %d row(s) from perf.csv", len(rows))
            return rows

    log.warning("No perf results found in %s", work_dir)
    return []


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ITER_RE = re.compile(
    r"iteration\s+(?P<iter>\d+)/\s*(?P<total>\d+)"
    r".*throughput per GPU \(TFLOP/s/GPU\):\s*[\d.]+/(?P<tflops_avg>[\d.]+)"
    r".*tokens per GPU \(tokens/s/GPU\):\s*[\d.]+/(?P<tps_avg>[\d.]+)"
)
_RUN_HEADER_RE = re.compile(r"Running:\s+(.+)\s+-\s+(\w+)\s+-\s+(\w+)\s*$")


def parse_live_log_metrics(work_dir: Path) -> list[dict]:
    """Parse madengine live logs for training metrics.

    Detects multiple runs within a single log (e.g. BF16 then FP8) by
    watching for "Running: Model - Precision - Mode" header lines.

    Returns a list of run dicts with keys:
      model, precision, mode, iter, total, tflops_avg,
      tokens_per_second_per_gpu, completed, log_file
    """
    logs = sorted(work_dir.glob("*.run.live.log"))
    if not logs:
        return []

    runs = []
    for log_path in logs:
        current = None
        with open(log_path) as f:
            for raw_line in f:
                line = _ANSI_RE.sub("", raw_line)

                hdr = _RUN_HEADER_RE.search(line)
                if hdr:
                    if current:
                        current.setdefault("iter", 0)
                        current.setdefault("total", 0)
                        current["completed"] = (
                            current["iter"] > 0
                            and current["iter"] == current["total"]
                        )
                        runs.append(current)
                    current = {
                        "model": hdr.group(1).strip(),
                        "precision": hdr.group(2),
                        "mode": hdr.group(3),
                        "log_file": str(log_path),
                    }
                    continue

                m = _ITER_RE.search(line)
                if m:
                    if current is None:
                        current = {"log_file": str(log_path)}
                    current.update({
                        "iter": int(m.group("iter")),
                        "total": int(m.group("total")),
                        "tflops_avg": float(m.group("tflops_avg")),
                        "tokens_per_second_per_gpu": float(m.group("tps_avg")),
                    })

        if current:
            current.setdefault("iter", 0)
            current.setdefault("total", 0)
            current["completed"] = (
                current["iter"] > 0
                and current["iter"] == current["total"]
            )
            runs.append(current)

    return runs


_SWEEP_POINT_RE = re.compile(r"_(isl\d+_osl\d+_con\d+)$")


def _row_precision(row: dict) -> str:
    """The key a result row is paired on: its precision, or its sweep point.

    An inference sweep has no precision; its rows are named
    ``<model>_<xPyD>_isl<I>_osl<O>_con<C>``, and the point is what the two
    phases have in common.
    """
    precision = (row.get("training_precision")
                 or row.get("multi_results", {}).get("precision", ""))
    if precision:
        return precision
    point = _SWEEP_POINT_RE.search(row.get("model", ""))
    return point.group(1) if point else ""


def _usable_metric(val: float) -> bool:
    """Whether a throughput can be scored; NaN would pass any threshold."""
    return math.isfinite(val) and val > 0


def collect_phase_results(
    run_dir: Path,
    metric_key: str,
    extra_aliases: set[str] | None = None,
    extra_tflops_aliases: set[str] | None = None,
    only: list[str] | None = None,
) -> list[dict]:
    """Per-precision metrics for one phase: structured output, else the live log.

    *only* keeps the rows of those precisions (sweep points) and drops the rest.
    """
    perf_results = parse_perf_results(run_dir)
    results: list[dict] = []

    if perf_results:
        # perf_entry_super.json rows are long-format: metric name is a
        # value in the ``metric`` column, performance in ``performance``.
        # Filter to the configured metric_key (or its madengine alias)
        # and key by precision.
        accepted_metrics = set(_METRIC_ALIASES.get(metric_key, {metric_key}))
        # Per-workload, deliberately not merged into _METRIC_ALIASES: the
        # MaxText extractor emits `tok_per_s_per_gpu`, already an average of
        # the last 10 steps, while Megatron emits a per-iteration
        # `tok_per_s_per_gpu` and an averaged `tok_per_s_per_gpu_avg`. Putting
        # the un-suffixed name back in the global table would silently score
        # Llama-3.1-70B on its final iteration again.
        if extra_aliases:
            accepted_metrics |= set(extra_aliases)
        for row in perf_results:
            if row.get("metric") not in accepted_metrics:
                continue
            perf_val = row.get("performance", "")
            precision = _row_precision(row)
            if only is not None and precision not in only:
                continue
            row_status = row.get("status", "")
            if not perf_val:
                continue
            try:
                val = float(perf_val)
            except (ValueError, TypeError):
                continue
            if not _usable_metric(val):
                log.warning("Ignoring structured result: %s %s = %r",
                            precision, metric_key, val)
                continue
            results.append({
                "precision": precision,
                "metric_value": val,
                "status": ("pass" if row_status.upper() in ("", "PASS", "SUCCESS")
                           else "fail"),
                "source": "structured",
            })
            log.info("Structured result: %s %s = %.1f (status=%s)",
                     precision, metric_key, val, row_status)

        # Attach TFLOPS from companion rows, keyed by precision. Same
        # per-workload widening as the throughput metric above, and for the
        # same reason: this extractor writes `TFLOPS_per_gpu` where Megatron
        # writes `TFLOPS_per_gpu_avg`, so without it every MaxText row records
        # a null and the report simply omits TFLOPS.
        accepted_tflops = set(_TFLOPS_METRICS) | set(extra_tflops_aliases or ())
        tflops_by_precision: dict[str, float] = {}
        for row in perf_results:
            if row.get("metric") not in accepted_tflops:
                continue
            prec = _row_precision(row)
            try:
                tflops_by_precision[prec] = float(row["performance"])
            except (KeyError, ValueError, TypeError):
                pass
        for pr in results:
            if "tflops_avg" not in pr:
                pr["tflops_avg"] = tflops_by_precision.get(pr["precision"])

    if not results:
        for run in parse_live_log_metrics(run_dir):
            val = run.get("tokens_per_second_per_gpu")
            if val is None:
                continue
            if not _usable_metric(val):
                log.warning("Ignoring live-log result: %s = %r",
                            run.get("precision"), val)
                continue
            results.append({
                "precision": run.get("precision"),
                "metric_value": val,
                "tflops_avg": run.get("tflops_avg"),
                "status": "pass" if run.get("completed", False) else "fail",
                "source": "live_log",
                "iter": run.get("iter", 0),
                "total": run.get("total", 0),
                "log_file": run.get("log_file"),
            })
            log.info("Live-log result: %s = %.1f (completed=%s)",
                     run.get("precision"), val, run.get("completed"))

    return results


def check_ab_regression(
    baseline: float,
    candidate: float,
    workload_type: str,
) -> tuple[bool, str]:
    """Score the candidate against the baseline measured on the same nodes.

    Returns (is_regression, message).
    """
    if baseline <= 0:
        return False, "Baseline is not a positive number — cannot compare"

    threshold = (
        REGRESSION_THRESHOLD_TRAINING
        if workload_type == "training"
        else REGRESSION_THRESHOLD_INFERENCE
    )
    ratio = candidate / baseline
    msg = (
        f"candidate {candidate:.1f} vs baseline {baseline:.1f} = "
        f"{ratio:.4f} ({ratio - 1:+.1%}), threshold: -{threshold:.0%}"
    )

    # Not `ratio - 1 < -threshold`: the subtraction rounds, so a loss of
    # exactly the threshold failed for some inputs and passed for others.
    if ratio < 1 - threshold:
        return True, f"REGRESSION DETECTED — {msg}"

    return False, f"No regression — {msg}"


def check_ratio_regression(
    ratio: float,
    previous: float,
    workload_type: str,
) -> tuple[bool, str]:
    """Score this run's candidate/stock ratio against the previous run's.

    The stock baseline is the same image every night, so it divides out the
    node pair the scheduler handed each run: the two ratios are comparable
    where the absolute numbers are not. The vs-stock check above cannot do
    this job on its own -- a workload whose image ships a much slower RCCL
    has a ratio far above 1 and passes it whatever the candidate does.

    Returns (is_regression, message).
    """
    threshold = (
        REGRESSION_THRESHOLD_TRAINING
        if workload_type == "training"
        else REGRESSION_THRESHOLD_INFERENCE
    )
    change = ratio / previous
    msg = (
        f"ratio {ratio:.4f} vs {previous:.4f} previously = "
        f"{change:.4f} ({change - 1:+.1%}), threshold: -{threshold:.0%}"
    )
    if change < 1 - threshold:
        return True, f"REGRESSION DETECTED — {msg}"
    return False, f"No regression — {msg}"


def previous_comparison(
    results_dir: Path,
    workload_name: str,
    scale: str,
    precision: str,
    base_image: str,
) -> dict | None:
    """The latest earlier passing run's candidate row for this workload.

    Called before this run appends its own rows, so the latest match is the
    previous run's. Same workload, scale, precision and base image only: a
    ratio depends on all four. The base image is the denominator -- a bump
    moves the ratio with no change in the candidate, and since only passing
    runs become references, matching across it would fail every night after
    the bump with nothing to reset it. After a bump the first run has no
    previous and becomes the new reference.

    Only runs whose own verdict was a pass count. A failed run's ratio would
    otherwise become the reference: a regression that failed one night would
    pass the next, and a run whose candidate never loaded the CI RCCL would
    set it to about 1.0. Rows written before the verdict was recorded have
    none and are skipped for the same reason.
    """
    try:
        with open(results_dir / PERF_DATASTORE) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            lines = f.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if (
            row.get("phase") == CANDIDATE
            and row.get("workload") == workload_name
            and row.get("scale") == scale
            and row.get("precision") == precision
            and row.get("base_image") == base_image
            and row.get("verdict") == "pass"
            and isinstance(row.get("ratio"), (int, float))
            and _usable_metric(float(row["ratio"]))
        ):
            return row
    return None


def append_result(
    results_dir: Path,
    workload_name: str,
    scale: str,
    metric_value: float | None,
    status: str,
    rccl_commit: str,
    extra: dict | None = None,
    precision: str | None = None,
    tflops: float | None = None,
    tokens_per_sec: float | None = None,
) -> None:
    """Append a result entry to the JSONL datastore."""
    results_dir.mkdir(parents=True, exist_ok=True)
    datastore = results_dir / PERF_DATASTORE

    entry = {
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": rccl_commit,
        "workload": workload_name,
        "scale": scale,
        "precision": precision,
        "tflops_per_gpu": tflops,
        "tokens_per_sec_per_gpu": tokens_per_sec,
        "metric_value": metric_value,
        "status": status,
    }
    if extra:
        entry.update(extra)

    # Matrix legs append in parallel on a shared filesystem.
    with open(datastore, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(entry) + "\n")
    log.info("Result appended to %s", datastore)

    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_dir = results_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)


def pair_by_precision(
    baseline_results: list[dict],
    candidate_results: list[dict],
    expected: list[str] | None = None,
) -> list[dict]:
    """Join the two phases into one comparison row per precision.

    A precision missing from a phase, or *expected* but absent from both, keeps
    a row with None so the gate fails instead of skipping it.
    """
    comparisons: dict[str, dict] = {
        precision: {"precision": precision, BASELINE: None, CANDIDATE: None}
        for precision in expected or []
    }
    for phase, rows in ((BASELINE, baseline_results), (CANDIDATE, candidate_results)):
        for row in rows:
            precision = row.get("precision") or ""
            entry = comparisons.setdefault(
                precision,
                {"precision": precision, BASELINE: None, CANDIDATE: None},
            )
            entry[phase] = row

    for entry in comparisons.values():
        base, cand = entry[BASELINE], entry[CANDIDATE]
        entry["ratio"] = None
        if base and cand:
            base_val, cand_val = base.get("metric_value"), cand.get("metric_value")
            if base_val and cand_val:
                entry["ratio"] = cand_val / base_val

    return [comparisons[p] for p in sorted(comparisons)]


def _format_phase(row: dict | None) -> str:
    """One side of a comparison line, or why it is absent."""
    if not row:
        return "missing"
    val = row.get("metric_value")
    if val is None:
        return "no metric"
    tflops = row.get("tflops_avg")
    suffix = f" ({tflops:.1f} TFLOP/s/GPU)" if tflops else ""
    return f"{val:.1f}{suffix} [{row['status']}]"


def generate_summary_report(
    workload_name: str,
    scale: str,
    exit_code: int,
    regression_msg: str,
    rccl_commit: str,
    cluster: str,
    comparisons: list[dict] | None = None,
    nodelist: str = "",
    job_id: str = "",
    phase_rccl: dict[str, str] | None = None,
    unit: str = "tok/s/GPU",
) -> str:
    """Generate a plain-text summary report."""
    status = "PASSED" if exit_code == 0 else "FAILED"
    phase_rccl = phase_rccl or {}
    lines = [
        "RCCL MADEngine Workload Test Report",
        "=" * 40,
        f"Status:     {status}",
        f"Date:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"Workload:   {workload_name}",
        f"Scale:      {scale}",
        f"Cluster:    {cluster}",
        f"Baseline:   RCCL {phase_rccl.get(BASELINE) or 'unknown'}",
        f"Candidate:  RCCL {phase_rccl.get(CANDIDATE) or 'unknown'} (CI build {rccl_commit})",
        f"Nodes:      {nodelist or 'unknown'} (job {job_id or 'unknown'})",
        "",
        "Both numbers below were measured on those same nodes, back to back,",
        "inside one allocation. Baseline is the stock image; candidate is the",
        "same image with the CI-built RCCL laid over it. Throughput in",
        f"{unit}.",
        "",
    ]

    if comparisons:
        for c in comparisons:
            ratio = c.get("ratio")
            ratio_s = f"{ratio:.4f} ({ratio - 1:+.1%})" if ratio else "N/A"
            lines.append(f"{c['precision'] or '?':>5}:  ratio {ratio_s}")
            lines.append(f"        baseline  {_format_phase(c[BASELINE])}")
            lines.append(f"        candidate {_format_phase(c[CANDIDATE])}")
            if c.get("previous_ratio") is not None:
                lines.append(
                    f"        previous  ratio {c['previous_ratio']:.4f} "
                    f"(run {c.get('previous_run_id') or '?'})"
                )
    else:
        lines.append("Throughput: N/A (workload did not produce metrics)")

    lines.append("")
    lines.append("Regression:")
    lines.extend(f"  {line}" for line in regression_msg.splitlines())
    lines.append("")

    run_url = os.environ.get("GITHUB_SERVER_URL", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if run_url and repo and run_id:
        lines.append(f"CI run: {run_url}/{repo}/actions/runs/{run_id}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing CI-built RCCL artifacts",
    )
    parser.add_argument(
        "--candidate-sha",
        required=True,
        help="Exact 40-character source SHA the RCCL artifact was built from",
    )
    parser.add_argument(
        "--artifact-provenance",
        type=Path,
        required=True,
        help="Verified artifact-run provenance JSON",
    )
    parser.add_argument(
        "--candidate-tuner-config",
        type=Path,
        required=True,
        help="gfx950 tuner CSV from the exact candidate source checkout",
    )
    parser.add_argument(
        "--workload",
        type=str,
        required=True,
        choices=list(WORKLOAD_CONFIGS.keys()),
        help="Workload to run",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        required=True,
        choices=list(CLUSTER_CONFIGS.keys()),
        help="Target cluster",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=2,
        help="Number of nodes to allocate (default: 2)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for JSONL datastore and run artifacts (default: cluster-specific path)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for madengine install, overlay build, etc.",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=330,
        help="Timeout for the whole allocation, queueing included, in minutes "
             "(default: 330 — a 4h allocation plus 90min of queueing)",
    )
    parser.add_argument(
        "--notify-email",
        type=str,
        default="",
        help="Send summary report to this email address",
    )
    parser.add_argument(
        "--teams-webhook",
        type=str,
        default="",
        help="Send summary report to this Teams webhook URL",
    )
    parser.add_argument(
        "--skip-overlay-build",
        action="store_true",
        help="Skip Docker overlay build (use pre-built image specified via --overlay-image)",
    )
    parser.add_argument(
        "--overlay-image",
        type=str,
        default="",
        help="Pre-built overlay image to use (requires --skip-overlay-build)",
    )

    args = parser.parse_args()
    # madengine runs inside our allocation only for multi-node workloads; a
    # single-node phase gets its own sbatch, possibly on other nodes.
    if args.nodes < 2:
        parser.error("--nodes must be at least 2 for a same-node A/B")

    workload_config = WORKLOAD_CONFIGS[args.workload]
    cluster_config = CLUSTER_CONFIGS[args.cluster]

    # Before anything takes an allocation. The workflow's `nodes` input
    # defaults to 2, so a hand dispatch that picks a workload validated at a
    # single scale and leaves the rest alone would otherwise burn exclusive
    # cluster time and OOM in step 0.
    required_nodes = workload_config.get("required_nodes")
    if required_nodes is not None and args.nodes != required_nodes:
        log.error(
            "%s is only supported at %d nodes (got %d): its parallelism and "
            "per_device_batch_size were validated at that scale and the memory "
            "headroom does not survive a different one.",
            args.workload, required_nodes, args.nodes,
        )
        sys.exit(1)

    scale = f"{args.nodes}N/{args.nodes * workload_config['gpus_per_node']}GPU"

    results_dir = args.results_dir or Path(cluster_config["results_base"])
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="madengine_ci_"))
    log.info("Work directory: %s", work_dir)
    log.info("Results directory: %s", results_dir)

    # Step 1: Find RCCL library and fingerprint it
    rccl_lib = find_rccl_library(args.artifact_dir)
    log.info("RCCL library: %s", rccl_lib)
    rccl_commit = get_rccl_commit(rccl_lib)
    log.info("RCCL commit/tag: %s", rccl_commit)
    rccl_fingerprint = get_rccl_fingerprint(rccl_lib)
    if rccl_fingerprint["md5"]:
        log.info("RCCL fingerprint: sha256=%s md5=%s version=%s size=%d",
                 rccl_fingerprint["sha256"], rccl_fingerprint["md5"],
                 rccl_fingerprint["version"],
                 rccl_fingerprint["size"])
    try:
        candidate_manifest = candidate_provenance(
            args.candidate_sha,
            args.artifact_provenance,
            args.candidate_tuner_config,
            rccl_lib,
            rccl_fingerprint,
        )
    except RuntimeError as exc:
        log.error("Candidate provenance verification FAILED: %s", exc)
        sys.exit(1)
    write_candidate_provenance(work_dir, candidate_manifest)
    log.info(
        "Candidate provenance verified: source=%s artifact_run=%s tuner=%s sha256=%s",
        candidate_manifest["candidate_sha"],
        candidate_manifest["artifact"].get("artifact_run_id"),
        candidate_manifest["tuner"]["expected_runtime_source"],
        candidate_manifest["tuner"]["source_sha256"],
    )

    # Step 2: Install madengine
    family = workload_config.get("family", "megatron")
    install_madengine(work_dir, family=family)
    check_mad_knobs(work_dir, workload_config)

    primus_dir = work_dir / "MAD" / "scripts" / "Primus" if family == "maxtext" else None

    # Resolved once and written into both A/B Dockerfiles, so the pair cannot
    # straddle a push to the mutable base tag between the two builds.
    base_ref = ""
    if family in ("maxtext", "sglang-disagg") and not args.skip_overlay_build:
        base_ref = resolve_base_digest(workload_config["base_image"])

    # Step 3: Stage the candidate image (or use pre-built). Images are built on
    # the allocation's nodes, never here: this host's disk is shared by runners.
    if args.skip_overlay_build:
        if not args.overlay_image:
            log.error("--skip-overlay-build requires --overlay-image")
            sys.exit(1)
        overlay_image = args.overlay_image
    else:
        overlay_image = stage_rccl_overlay(
            rccl_lib,
            args.candidate_tuner_config,
            workload_config["base_image"],
            cluster_config["gpu_target"],
            work_dir,
            family=family,
            primus_dir=primus_dir,
            from_ref=base_ref,
        )

    # The maxtext baseline cannot be the stock tag: that image ships MaxText
    # but no Primus, so run.sh would not find primus-cli and the baseline
    # phase would die before step 0. Stage base + Primus instead, which also
    # makes the pair differ by the RCCL swap and nothing else. Staged, not
    # built: the nodes build both images of the pair, this host neither.
    if family == "maxtext":
        baseline_image = stage_maxtext_baseline(
            workload_config["base_image"], work_dir, primus_dir,
            # Same token the candidate tag carries, so the pair is unambiguously
            # from one run, and the same pinned base digest.
            run_token=rccl_commit,
            from_ref=base_ref or workload_config["base_image"],
        )
        baseline_pull_only = False
    elif family == "sglang-disagg":
        baseline_image = stage_sglang_baseline(
            workload_config["base_image"], work_dir,
            run_token=rccl_commit,
            from_ref=base_ref or workload_config["base_image"],
        )
        baseline_pull_only = False
    else:
        baseline_image = workload_config["base_image"]
        baseline_pull_only = True

    candidate_manifest.update({
        "candidate_image": {
            "reference": overlay_image,
            "base_reference": base_ref or workload_config["base_image"],
            "digest": "",
        },
        "baseline": {
            "reference": baseline_image,
            "policy": "stock/previous-good image, independent of candidate develop tip",
        },
    })
    write_candidate_provenance(work_dir, candidate_manifest)

    # Step 4: Generate one manifest per A/B phase
    phases: list[tuple[str, Path]] = []
    for phase, image, pull_only in (
        (BASELINE, baseline_image, baseline_pull_only),
        (CANDIDATE, overlay_image, False),
    ):
        run_dir = prepare_phase_dir(work_dir, phase)
        generate_manifest(
            args.workload,
            workload_config,
            cluster_config,
            image,
            args.nodes,
            work_dir,
            # The bind-mount of the CI library is what makes a --skip-overlay-build
            # run the candidate; the baseline must keep the image's own copy.
            rccl_lib=(
                rccl_lib if args.skip_overlay_build and phase == CANDIDATE else None
            ),
            run_dir=run_dir,
            pull_only=pull_only,
            # A maxtext baseline is built on the nodes, so its manifest must
            # advertise ITS OWN dockerfile. Left at the default it would hand
            # a node the candidate recipe and have it build the swapped image
            # under the baseline tag -- an A/B comparing the CI RCCL against
            # itself.
            dockerfile_name=(
                LOCAL_BASELINE_DOCKERFILES.get(family, OVERLAY_DOCKERFILE)
                if phase == BASELINE
                else OVERLAY_DOCKERFILE
            ),
        )
        phases.append((phase, run_dir))

    phase_dirs = dict(phases)

    # Step 4b: before the run, so a failing night can be rerun by hand too. It
    # only copies what the run already produced, and a failure here must not
    # cost the gate its result.
    try:
        write_repro_bundle(
            work_dir, phase_dirs, args.workload, workload_config, cluster_config,
            args.nodes, rccl_fingerprint,
            images={BASELINE: baseline_image, CANDIDATE: overlay_image},
            base_digest=base_ref or registry_digest(workload_config["base_image"]),
        )
    except Exception:
        log.exception("Could not write the repro bundle; continuing without it")

    # Step 5: Run both phases on the same nodes, in one allocation
    allocation = run_ab_in_one_allocation(
        phases,
        work_dir,
        args.nodes,
        args.workload,
        workload_config,
        cluster_config,
        args.timeout_minutes,
        # A --skip-overlay-build image and a stock baseline are not ours to remove.
        built_images=(
            ([] if args.skip_overlay_build else [overlay_image])
            + ([] if baseline_pull_only else [baseline_image])
        ),
        candidate_image=overlay_image,
        candidate_tuner_sha256=candidate_manifest["tuner"]["source_sha256"],
    )
    exit_code = max(allocation["exit_codes"].values(), default=1)

    # Step 5a: Shrink the node logs before anything reads or uploads them.
    for phase, run_dir in phases:
        saved = squash_log_spam(run_dir)
        if saved:
            log.info("%s: dropped %.1f MB of repeated RCCL warnings from the node logs",
                     phase, saved / 1e6)

    # Step 5b: Verify RCCL replacement. Only the candidate is meant to carry the
    # CI library — the baseline runs the image's bundled copy by design.
    rccl_verification_failed = False
    if not allocation["candidate_image_ok"]:
        log.error(
            "Candidate image verification FAILED: %s",
            allocation["candidate_image_message"],
        )
        rccl_verification_failed = True
        exit_code = max(exit_code, 1)
    if rccl_fingerprint.get("version"):
        rccl_ok, rccl_msg = verify_rccl_replacement(
            phase_dirs[CANDIDATE], rccl_fingerprint, args.nodes,
        )
        if rccl_ok:
            log.info("RCCL verification: %s", rccl_msg)
        else:
            log.error("RCCL verification FAILED: %s", rccl_msg)
            rccl_verification_failed = True
            exit_code = max(exit_code, 1)

    tuner_sources = runtime_tuner_sources(phase_dirs[CANDIDATE])
    if family == "sglang-disagg":
        tuner_ok, tuner_msg = verify_candidate_tuner(
            phase_dirs[CANDIDATE], args.nodes,
        )
        if tuner_ok:
            log.info("RCCL tuner verification: %s", tuner_msg)
        else:
            log.error("RCCL tuner verification FAILED: %s", tuner_msg)
            rccl_verification_failed = True
            exit_code = max(exit_code, 1)

    candidate_manifest["candidate_image"].update({
        "digest": allocation["candidate_image_id"],
        "verification": allocation["candidate_image_message"],
    })
    candidate_manifest["runtime"] = {
        "rccl_versions": runtime_rccl_versions(phase_dirs[CANDIDATE]),
        "baseline_rccl_versions": runtime_rccl_versions(phase_dirs[BASELINE]),
        "tuner_sources": tuner_sources,
        "tuner_expected": (
            CANDIDATE_TUNER_PATH if family == "sglang-disagg" else "not asserted"
        ),
    }
    write_candidate_provenance(work_dir, candidate_manifest)

    # Step 6: Parse each phase's results and save its artifacts
    metric_key = workload_config["metric_key"]
    phase_results = {}
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    for phase, run_dir in phases:
        log.info("Collecting %s results from %s", phase, run_dir)
        phase_results[phase] = collect_phase_results(
            run_dir, metric_key,
            extra_aliases=workload_config.get("metric_aliases"),
            extra_tflops_aliases=workload_config.get("tflops_aliases"),
            # A sweep reports many points; only the configured ones are scored.
            only=(workload_config["precisions"]
                  if workload_config["type"] == "inference" else None),
        )

        # Matrix legs share GITHUB_RUN_ID, so the workload keeps them apart.
        run_artifacts = results_dir / "runs" / run_id / args.workload / phase
        try:
            run_artifacts.mkdir(parents=True, exist_ok=True)
            for f in ["perf.csv", "perf_entry_super.csv", "perf_entry_super.json"]:
                src = run_dir / f
                if src.exists():
                    shutil.copy2(str(src), str(run_artifacts / f))
        except OSError as exc:
            log.warning("Could not save %s artifacts to %s: %s",
                        phase, run_artifacts, exc)

    all_rows = [r for rows in phase_results.values() for r in rows]

    # Override exit_code if training actually completed successfully.
    # madengine can report failure (exit code 3) when its perf collector
    # can't parse the output format, even though training ran to completion.
    if exit_code != 0 and not rccl_verification_failed and all_rows:
        every_phase_scored = all(phase_results.get(p) for p, _ in phases)
        all_pass = all(r["status"] == "pass" for r in all_rows)
        has_metric = all(r.get("metric_value") is not None for r in all_rows)
        if every_phase_scored and all_pass and has_metric:
            log.info(
                "Overriding madengine exit code %d → 0: all %d run(s) across "
                "both phases passed with metrics",
                exit_code, len(all_rows),
            )
            exit_code = 0

    # Step 7: Score the candidate against the baseline, per precision
    comparisons = pair_by_precision(
        phase_results.get(BASELINE, []), phase_results.get(CANDIDATE, []),
        expected=workload_config.get("precisions", []),
    )
    regression_msgs = []
    for c in comparisons:
        prec = c["precision"] or "?"
        base, cand = c[BASELINE], c[CANDIDATE]
        if c["ratio"] is None:
            # A missing side is a job failure, not evidence about RCCL.
            regression_msgs.append(
                f"[{prec}] no comparison — baseline={_format_phase(base)}, "
                f"candidate={_format_phase(cand)}"
            )
            log.error("No comparison possible for %s", prec)
            exit_code = max(exit_code, 1)
            continue
        reg, msg = check_ab_regression(
            base["metric_value"], cand["metric_value"], workload_config["type"],
        )
        regression_msgs.append(f"[{prec}] vs stock:    {msg}")
        if reg:
            log.warning(msg)
            exit_code = max(exit_code, 1)
        else:
            log.info(msg)

        # Read before Step 8 appends this run's rows, so it is the previous run.
        prev = previous_comparison(
            results_dir, args.workload, scale, c["precision"],
            workload_config["base_image"],
        )
        if prev is None:
            regression_msgs.append(
                f"[{prec}] vs previous: no earlier comparison recorded — not scored"
            )
            continue
        c["previous_ratio"] = float(prev["ratio"])
        c["previous_run_id"] = prev.get("run_id", "")
        reg, msg = check_ratio_regression(
            c["ratio"], c["previous_ratio"], workload_config["type"],
        )
        regression_msgs.append(
            f"[{prec}] vs previous: {msg} (run {c['previous_run_id'] or '?'})"
        )
        if reg:
            log.warning(msg)
            exit_code = max(exit_code, 1)
        else:
            log.info(msg)

    if not comparisons:
        regression_msgs.append("no metrics from either phase — nothing to compare")
        exit_code = max(exit_code, 1)
    regression_msg = "\n".join(regression_msgs)

    # Step 8: Append to datastore — one record per phase per precision, with
    # absolute values kept for trend analysis and the ratio on the candidate.
    extra = {
        # What previous_comparison() keys the reference on: the stock
        # image is the ratio's denominator.
        "base_image": workload_config["base_image"],
        # The run's own gate verdict, final by now: it covers the phase exit
        # codes, RCCL verification and both regression checks. A later run
        # takes only a passing one as its reference.
        "verdict": "pass" if exit_code == 0 else "fail",
        "cluster": args.cluster,
        "overlay_image": overlay_image,
        "nodelist": allocation["nodelist"],
        "slurm_job_id": allocation["job_id"],
        # run_id names this test run; these name the RCCL binary it tested.
        "artifact_run_id": os.environ.get("ARTIFACT_RUN_ID", ""),
        "candidate_sha": candidate_manifest["candidate_sha"],
        "artifact_source_sha": candidate_manifest["artifact"]["artifact_source_sha"],
        "rccl_md5": rccl_fingerprint["md5"],
        "rccl_sha256": rccl_fingerprint["sha256"],
        "candidate_image_digest": allocation["candidate_image_id"],
        "candidate_tuner_source": ",".join(tuner_sources),
        "candidate_tuner_sha256": candidate_manifest["tuner"]["source_sha256"],
        "candidate_kpack_sha256": ",".join(
            item["sha256"] for item in candidate_manifest["kpack"]
        ),
    }
    # baseline_image, not workload_config["base_image"]: for maxtext the
    # baseline is a locally built Primus image, and recording the stock tag
    # would put a provenance in the datastore that was never run.
    phase_images = {
        BASELINE: baseline_image,
        CANDIDATE: overlay_image,
    }
    if all_rows:
        for c in comparisons:
            for phase in (BASELINE, CANDIDATE):
                row = c[phase]
                if row is None:
                    continue
                phase_extra = {**extra, "phase": phase, "image": phase_images[phase]}
                if phase == CANDIDATE and c["ratio"] is not None:
                    phase_extra["ratio"] = c["ratio"]
                    if c.get("previous_ratio") is not None:
                        phase_extra["previous_ratio"] = c["previous_ratio"]
                        phase_extra["previous_run_id"] = c["previous_run_id"]
                append_result(
                    results_dir,
                    args.workload,
                    scale,
                    row.get("metric_value"),
                    row["status"],
                    rccl_commit,
                    extra=phase_extra,
                    precision=c["precision"],
                    tflops=row.get("tflops_avg"),
                    # Per GPU for training only; inference scores the whole
                    # deployment, and metric_value already carries that.
                    tokens_per_sec=(row.get("metric_value")
                                    if workload_config["type"] == "training" else None),
                )
    else:
        append_result(
            results_dir,
            args.workload,
            scale,
            None,
            "fail",
            rccl_commit,
            extra=extra,
        )

    # Step 9: Generate and distribute report
    status = "pass" if exit_code == 0 else "fail"
    report = generate_summary_report(
        args.workload, scale, exit_code,
        regression_msg, rccl_commit, args.cluster,
        comparisons=comparisons,
        nodelist=allocation["nodelist"],
        job_id=allocation["job_id"],
        phase_rccl={
            phase: ", ".join(runtime_rccl_versions(run_dir))
            for phase, run_dir in phases
        },
        unit=workload_config.get("metric_unit", "tok/s/GPU"),
    )
    log.info("\n%s", report)
    write_github_summary(report)
    set_github_output("madengine_status", status)
    candidate_values = [
        c[CANDIDATE]["metric_value"]
        for c in comparisons
        if c[CANDIDATE] and c[CANDIDATE].get("metric_value") is not None
    ]
    if candidate_values:
        set_github_output("madengine_metric", f"{candidate_values[-1]:.1f}")
    ratios = [c["ratio"] for c in comparisons if c["ratio"] is not None]
    if ratios:
        set_github_output("madengine_ratio", f"{min(ratios):.4f}")

    summary_path = work_dir / "madengine_summary.txt"
    summary_path.write_text(report)

    report_status = "PASSED" if exit_code == 0 else "FAILED"
    if args.notify_email:
        send_email_report(report, args.notify_email, report_status,
                          subject_prefix=f"RCCL MADEngine {args.workload}")
    if args.teams_webhook:
        send_teams_webhook(report, args.teams_webhook, report_status,
                           subject_prefix=f"RCCL MADEngine {args.workload}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
