#!/usr/bin/env python3
"""Run the causal-LM smoke test on CPU, real GPU, and simulated gfx1250."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from compare import compare_one
from common import (
    ALLOW_PATTERNS,
    ATOL,
    DEFAULT_MANIFEST,
    IGNORE_PATTERNS,
    MAX_NEW_TOKENS,
    PROMPTS,
    ROCJITSU_ROOT,
    RTOL,
    load_models,
    prepend_env_path,
    resolve_model,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATE_SCRIPT = SCRIPT_DIR / "generate.py"
TARGETS = ("cpu", "real_gpu", "sim_gfx1250")
TIMEOUT_SECONDS = {
    "cpu": 600.0,
    "real_gpu": 600.0,
    "sim_gfx1250": 900.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="tinyllama_1b", help="Model key from models.json")
    parser.add_argument("--model-cache-dir", type=Path, default=SCRIPT_DIR / "model_cache")
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--rocm-python", default=os.environ.get("ROCM_PYTHON"))
    parser.add_argument("--rocm-lib-dir", default=os.environ.get("ROCM_LIB_DIR"))
    parser.add_argument("--rocjitsu-build", default=os.environ.get("ROCJITSU_BUILD"))
    parser.add_argument(
        "--rocjitsu-config",
        type=Path,
        default=Path(os.environ["ROCJITSU_CONFIG"])
        if os.environ.get("ROCJITSU_CONFIG")
        else ROCJITSU_ROOT / "configs" / "gfx1250.json",
    )
    return parser.parse_args()


def require_value(name: str, value: str | None) -> str:
    if value:
        return value
    raise SystemExit(f"{name} is required; set ${name} or pass --{name.lower().replace('_', '-')}")


def make_env(
    *,
    rocm_lib_dir: str,
    extra_ld_paths: list[str | Path] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    if env.get("HF_DEPS"):
        env["PYTHONPATH"] = prepend_env_path(env.get("PYTHONPATH"), [env["HF_DEPS"]])

    ld_paths: list[str | Path] = []
    if extra_ld_paths:
        ld_paths.extend(extra_ld_paths)
    ld_paths.append(rocm_lib_dir)
    env["LD_LIBRARY_PATH"] = prepend_env_path(env.get("LD_LIBRARY_PATH"), ld_paths)
    return env


def generation_command(
    *,
    rocm_python: str,
    model_id: str,
    model_path: Path,
    device: str,
    output_json: Path,
) -> list[str]:
    return [
        rocm_python,
        str(GENERATE_SCRIPT),
        "--device",
        device,
        "--model-id",
        model_id,
        "--model-path",
        str(model_path),
        "--output-json",
        str(output_json),
    ]


def run_step(
    *,
    name: str,
    command: list[str],
    env: dict[str, str],
    results_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    stdout_path = results_dir / f"{name}.stdout.log"
    stderr_path = results_dir / f"{name}.stderr.log"
    time_path = results_dir / f"{name}.time.json"
    started = time.monotonic()
    timed_out = False
    returncode: int | None

    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=SCRIPT_DIR,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
            stderr.write(f"\n{name} timed out after {timeout} seconds\n")
        except OSError as exc:
            returncode = None
            stderr.write(f"\n{name} failed to start: {exc}\n")

    payload = {
        "name": name,
        "command": command,
        "command_string": shlex.join(command),
        "cwd": str(SCRIPT_DIR),
        "elapsed_seconds": time.monotonic() - started,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "timeout_seconds": timeout,
    }
    write_json(time_path, payload)
    return payload


def step_failed(step: dict[str, Any]) -> bool:
    return step.get("timed_out") or step.get("returncode") != 0


def download_model(model: dict[str, str], model_cache_dir: Path, results_dir: Path) -> dict[str, Any]:
    stdout_path = results_dir / "download.stdout.log"
    stderr_path = results_dir / "download.stderr.log"
    time_path = results_dir / "download.time.json"
    started = time.monotonic()
    local_dir = model_cache_dir / model["local_dir"]
    returncode = 0

    try:
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(
            repo_id=model["id"],
            local_dir=local_dir,
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )
        stdout_path.write_text(
            "download_summary_json="
            + json.dumps(
                {
                    "status": "ok",
                    "model": model["key"],
                    "model_id": model["id"],
                    "local_dir": str(local_dir),
                    "snapshot_path": str(snapshot_path),
                },
                sort_keys=True,
            )
            + "\n"
        )
        stderr_path.write_text("")
    except Exception as exc:  # noqa: BLE001 - keep the validation runner diagnostic.
        returncode = 2
        stdout_path.write_text("")
        stderr_path.write_text(
            "download_error_json="
            + json.dumps(
                {
                    "status": "failed",
                    "model": model["key"],
                    "model_id": model["id"],
                    "local_dir": str(local_dir),
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            )
            + "\n"
        )

    payload = {
        "name": "download",
        "command": ["huggingface_hub.snapshot_download", model["id"]],
        "cwd": str(SCRIPT_DIR),
        "elapsed_seconds": time.monotonic() - started,
        "returncode": returncode,
        "timed_out": False,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    write_json(time_path, payload)
    return payload


def write_summary(
    *,
    summary_path: Path,
    status: str,
    model: dict[str, str],
    model_path: Path,
    outputs: dict[str, Path],
    steps: list[dict[str, Any]],
    compare_results: list[dict[str, Any]],
) -> None:
    payload = {
        "status": status,
        "model": model,
        "model_path": str(model_path),
        "target_outputs": {key: str(value) for key, value in outputs.items()},
        "validation": {
            "targets": list(TARGETS),
            "prompts": PROMPTS,
            "prompt_count": len(PROMPTS),
            "max_new_tokens": MAX_NEW_TOKENS,
            "dtype": "float32",
            "attention": "eager",
            "rtol": RTOL,
            "atol": ATOL,
        },
        "steps": steps,
        "compare_results": compare_results,
    }
    write_json(summary_path, payload)
    print("validation_summary_json=" + json.dumps(payload, sort_keys=True))


def main() -> int:
    args = parse_args()
    rocm_python = require_value("ROCM_PYTHON", args.rocm_python)
    rocm_lib_dir = require_value("ROCM_LIB_DIR", args.rocm_lib_dir)
    rocjitsu_build = require_value("ROCJITSU_BUILD", args.rocjitsu_build)

    model = resolve_model(load_models(DEFAULT_MANIFEST), args.model)
    model_cache_dir = args.model_cache_dir.resolve()
    model_path = model_cache_dir / model["local_dir"]
    results_dir = (args.results_dir / model["key"]).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "cpu": results_dir / f"{model['key']}_cpu.json",
        "real_gpu": results_dir / f"{model['key']}_real_gpu.json",
        "sim_gfx1250": results_dir / f"{model['key']}_sim_gfx1250.json",
    }
    summary_path = results_dir / f"{model['key']}_summary.json"
    for path in [*outputs.values(), summary_path]:
        path.unlink(missing_ok=True)

    base_env = make_env(rocm_lib_dir=rocm_lib_dir)
    steps: list[dict[str, Any]] = []

    def finish(status: str, compare_results: list[dict[str, Any]] | None = None) -> int:
        write_summary(
            summary_path=summary_path,
            status=status,
            model=model,
            model_path=model_path,
            outputs=outputs,
            steps=steps,
            compare_results=compare_results or [],
        )
        return 0 if status == "ok" else 1

    steps.append(download_model(model, model_cache_dir, results_dir))
    if step_failed(steps[-1]):
        return finish("failed")

    real_gpu_env = dict(base_env)
    real_gpu_env["HSA_ENABLE_SDMA"] = "1"

    runtime_root = Path(os.environ.get("TMPDIR", "/tmp")).resolve()
    runtime_dir = runtime_root / f"rocjitsu-causal-lm-{model['key']}"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sim_env = make_env(
        rocm_lib_dir=rocm_lib_dir,
        extra_ld_paths=[
            rocjitsu_build,
            Path(rocjitsu_build) / "lib" / "rocjitsu" / "src" / "rocjitsu" / "hooks",
        ],
    )
    sim_env["XDG_RUNTIME_DIR"] = str(runtime_root)
    sim_env["ROCJITSU_RUNTIME_DIR"] = str(runtime_dir)
    sim_env["HSA_ENABLE_SDMA"] = "1"
    sim_env["HSA_HOTSWAP_DISABLE"] = "1"

    target_steps = [
        (
            "cpu",
            generation_command(
                rocm_python=rocm_python,
                model_id=model["id"],
                model_path=model_path,
                device="cpu",
                output_json=outputs["cpu"],
            ),
            base_env,
        ),
        (
            "real_gpu",
            generation_command(
                rocm_python=rocm_python,
                model_id=model["id"],
                model_path=model_path,
                device="cuda",
                output_json=outputs["real_gpu"],
            ),
            real_gpu_env,
        ),
        (
            "sim_gfx1250",
            [
                str(Path(rocjitsu_build) / "tools" / "rocjitsu" / "rocjitsu"),
                "--daemon",
                "--config",
                str(args.rocjitsu_config.resolve()),
                "--",
                *generation_command(
                    rocm_python=rocm_python,
                    model_id=model["id"],
                    model_path=model_path,
                    device="cuda",
                    output_json=outputs["sim_gfx1250"],
                ),
            ],
            sim_env,
        ),
    ]
    for name, command, env in target_steps:
        steps.append(
            run_step(
                name=name,
                command=command,
                env=env,
                results_dir=results_dir,
                timeout=TIMEOUT_SECONDS[name],
            )
        )
        if step_failed(steps[-1]):
            return finish("failed")

    compare_results = [
        compare_one(outputs["cpu"], outputs["real_gpu"]),
        compare_one(outputs["cpu"], outputs["sim_gfx1250"]),
    ]
    failed = any(
        not result["prompt_match"]
        or not result["sequence_ids_match"]
        or not result["new_token_ids_match"]
        or not result["allclose"]
        for result in compare_results
    )
    return finish("failed" if failed else "ok", compare_results)


if __name__ == "__main__":
    raise SystemExit(main())
