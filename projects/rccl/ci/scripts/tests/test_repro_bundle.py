"""Tests for the repro bundle written by test_madengine.py.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import test_madengine as tm

RUBY = tm.CLUSTER_CONFIGS["ruby"]
CI_ENV = {
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_REPOSITORY": "ROCm/rocm-systems",
    "GITHUB_RUN_ID": "38523607643",
    "GITHUB_RUN_ATTEMPT": "1",
    "GITHUB_SHA": "2b88119d18cf83a0b4127ad441b5ec813bd93580",
    "ARTIFACT_RUN_ID": "38523607643",
    "SLURM_TMPDIR": "/var/tmp",
}
FINGERPRINT = {"md5": "1f9f7bd1a083f5c3184ade694bb984ae", "version": "2.32.3-develop:2b88119"}


def build_info_resolves(manifest: dict, cwd: Path) -> tuple[Path, Path]:
    """The Dockerfile and context madengine would build from, run from *cwd*.

    Mirrors ContainerRunner._build_local_image_from_manifest (madengine 00ee3e80):
    the Dockerfile comes from built_images, the context from built_models, and a
    missing context falls back to ./docker, then to the Dockerfile's directory.
    """
    dockerfile = manifest["built_images"]["overlay"]["dockerfile"]
    context = manifest["built_models"]["overlay"].get("dockercontext", "") or "./docker"
    if not (cwd / context).exists():
        context = os.path.dirname(dockerfile) or "."
    return cwd / dockerfile, cwd / context


class WorkDir:
    """A WORK_DIR laid out as main() leaves it after step 4, minus Docker."""

    def __init__(self, root: Path, workload: str):
        self.path = root / "work"
        self.workload = workload
        self.config = tm.WORKLOAD_CONFIGS[workload]
        family = self.config.get("family", "megatron")
        ctx = self.path / tm.OVERLAY_CTX
        libs = ctx / "rccl_libs"
        (libs / ".kpack").mkdir(parents=True)
        lib = b"\x7fELF librccl" * 1000
        for name in ("librccl.so", "librccl.so.1", "librccl.so.1.0"):
            (libs / name).write_bytes(lib)
        (libs / ".kpack" / "rccl_lib_gfx950.kpack").write_bytes(b"kpack")
        (ctx / tm.OVERLAY_DOCKERFILE).write_text("FROM base\nCOPY rccl_libs/ /tmp/rccl_ci/\n")
        scripts = "jax-maxtext" if family == "maxtext" else "primus_megatron-lm"
        (self.path / "scripts" / scripts).mkdir(parents=True)
        (self.path / "scripts" / scripts / "run.sh").write_text("#!/bin/bash\n")
        if family == "maxtext":
            (ctx / tm.MAXTEXT_BASELINE_DOCKERFILE).write_text("FROM base\n")
            (ctx / "Primus").mkdir()
            (ctx / "Primus" / "primus-cli").write_text("#!/bin/bash\n")

        self.images = {tm.BASELINE: "base:tag" if family != "maxtext" else "base-primus",
                       tm.CANDIDATE: "base:tag-rccl-gfx950-run38523607643"}
        self.phase_dirs = {}
        for phase in (tm.BASELINE, tm.CANDIDATE):
            run_dir = tm.prepare_phase_dir(self.path, phase)
            maxtext_baseline = family == "maxtext" and phase == tm.BASELINE
            tm.generate_manifest(
                workload, self.config, RUBY, self.images[phase],
                self.config.get("required_nodes", 2), self.path,
                run_dir=run_dir,
                pull_only=family != "maxtext" and phase == tm.BASELINE,
                dockerfile_name=(tm.MAXTEXT_BASELINE_DOCKERFILE if maxtext_baseline
                                 else tm.OVERLAY_DOCKERFILE),
                local_only=maxtext_baseline,
            )
            self.phase_dirs[phase] = run_dir

    def write_bundle(self) -> Path:
        return tm.write_repro_bundle(
            self.path, self.phase_dirs, self.workload, self.config, RUBY,
            self.config.get("required_nodes", 2), FINGERPRINT, self.images,
            base_digest="rocm/primus:v26.4@sha256:8c8ecc6f", env=CI_ENV,
        )


class ReproManifestTest(unittest.TestCase):
    def test_paths_become_bundle_relative_and_the_input_is_untouched(self):
        work = Path("/w")
        manifest = {
            "built_images": {"overlay": {"dockerfile": "/w/overlay_ctx/Dockerfile.rccl-overlay"}},
            "built_models": {"overlay": {"name": "m"}},
        }
        out = tm.repro_manifest(manifest, work)
        self.assertEqual(out["built_images"]["overlay"]["dockerfile"],
                         "overlay_ctx/Dockerfile.rccl-overlay")
        self.assertEqual(out["built_models"]["overlay"]["dockercontext"], "overlay_ctx")
        self.assertEqual(manifest["built_images"]["overlay"]["dockerfile"],
                         "/w/overlay_ctx/Dockerfile.rccl-overlay")
        self.assertNotIn("dockercontext", manifest["built_models"]["overlay"])

    def test_pull_only_and_foreign_dockerfiles_are_left_alone(self):
        work = Path("/w")
        pull_only = {"built_images": {"overlay": {"docker_image": "rocm/primus:v26.4"}},
                     "built_models": {"overlay": {}}}
        self.assertEqual(tm.repro_manifest(pull_only, work), pull_only)
        foreign = {"built_images": {"overlay": {"dockerfile": "/elsewhere/Dockerfile"}},
                   "built_models": {"overlay": {}}}
        self.assertEqual(tm.repro_manifest(foreign, work), foreign)


class CopyOverlayCtxTest(unittest.TestCase):
    def test_drops_only_byte_identical_librccl_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "src", Path(tmp) / "dst"
            libs = src / "rccl_libs"
            (libs / ".kpack").mkdir(parents=True)
            (libs / "librccl.so").write_bytes(b"A" * 64)
            (libs / "librccl.so.1").write_bytes(b"A" * 64)
            (libs / "librccl.so.1.0").write_bytes(b"B" * 64)
            (libs / ".kpack" / "rccl.kpack").write_bytes(b"K")
            (src / "Dockerfile").write_text("FROM x\n")
            tm.copy_overlay_ctx(src, dst)
            kept = sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file())
            self.assertEqual(kept, ["Dockerfile", "rccl_libs/.kpack/rccl.kpack",
                                    "rccl_libs/librccl.so", "rccl_libs/librccl.so.1.0"])


class WriteReproBundleTest(unittest.TestCase):
    def check_bundle(self, workload: str) -> tuple[Path, WorkDir]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        work = WorkDir(Path(tmp.name), workload)
        ci_manifests = {p: (d / "manifest.json").read_text() for p, d in work.phase_dirs.items()}

        bundle = work.write_bundle()

        self.assertEqual(bundle, work.path / tm.REPRO_DIR)
        for phase, text in ci_manifests.items():
            self.assertEqual((work.phase_dirs[phase] / "manifest.json").read_text(), text,
                             "the CI manifests must not change")
        self.assertTrue((bundle / "scripts").is_dir())
        self.assertTrue((bundle / "REPRO.md").is_file())
        self.assertFalse((bundle / tm.OVERLAY_CTX / "rccl_libs" / "librccl.so.1").exists())
        self.assertTrue((bundle / tm.OVERLAY_CTX / "rccl_libs" / ".kpack").is_dir())

        candidate = json.loads((bundle / tm.REPRO_MANIFESTS[tm.CANDIDATE]).read_text())
        dockerfile, context = build_info_resolves(candidate, bundle)
        self.assertEqual(dockerfile, bundle / tm.OVERLAY_CTX / tm.OVERLAY_DOCKERFILE)
        self.assertTrue(dockerfile.is_file())
        self.assertEqual(context, bundle / tm.OVERLAY_CTX)
        script = candidate["built_models"]["overlay"]["scripts"]
        self.assertTrue((bundle / script).is_file(), script)
        return bundle, work

    def test_megatron_bundle(self):
        bundle, _ = self.check_bundle("llama-3.1-70b-training")
        baseline = json.loads((bundle / tm.REPRO_MANIFESTS[tm.BASELINE]).read_text())
        self.assertNotIn("dockerfile", baseline["built_images"]["overlay"])
        self.assertNotIn("dockercontext", baseline["built_models"]["overlay"])

    def test_maxtext_baseline_keeps_its_own_dockerfile(self):
        bundle, _ = self.check_bundle("llama-3.1-405b-maxtext")
        baseline = json.loads((bundle / tm.REPRO_MANIFESTS[tm.BASELINE]).read_text())
        dockerfile, context = build_info_resolves(baseline, bundle)
        self.assertEqual(dockerfile, bundle / tm.OVERLAY_CTX / tm.MAXTEXT_BASELINE_DOCKERFILE)
        self.assertEqual(context, bundle / tm.OVERLAY_CTX)
        self.assertTrue((bundle / tm.OVERLAY_CTX / "Primus" / "primus-cli").is_file())

    def test_repro_env_sources_and_carries_provenance(self):
        bundle, _ = self.check_bundle("llama-3.1-70b-training")
        names = ["CI_RUN_URL", "RCCL_ARTIFACTS_S3", "RCCL_MD5", "MAD_REF", "MADENGINE_REF",
                 "BASE_DIGEST", "SLURM_TMPDIR", "REPRO_PARTITION", "REPRO_NODES"]
        script = "source repro.env; " + "; ".join(f'echo "{n}=${n}"' for n in names)
        out = subprocess.run(["bash", "-c", script], cwd=bundle, check=True,
                             capture_output=True, text=True).stdout
        got = dict(line.split("=", 1) for line in out.splitlines())
        self.assertEqual(got["CI_RUN_URL"],
                         "https://github.com/ROCm/rocm-systems/actions/runs/38523607643")
        self.assertEqual(got["RCCL_ARTIFACTS_S3"],
                         "s3://therock-ci-artifacts-external/ROCm-rocm-systems/38523607643-linux/")
        self.assertEqual(got["RCCL_MD5"], FINGERPRINT["md5"])
        self.assertEqual(got["MAD_REF"], tm.MAD_REF)
        self.assertEqual(got["MADENGINE_REF"], tm.MADENGINE_REF)
        self.assertEqual(got["BASE_DIGEST"], "rocm/primus:v26.4@sha256:8c8ecc6f")
        self.assertEqual(got["SLURM_TMPDIR"], "/var/tmp")
        self.assertEqual(got["REPRO_PARTITION"], "meta64")
        self.assertEqual(got["REPRO_NODES"], "2")
        env_text = (bundle / "repro.env").read_text()
        self.assertNotIn("SLURM_PARTITION", env_text)
        self.assertNotIn("PRIMUS_REF", env_text)
        self.assertIn("38523607643", (bundle / "REPRO.md").read_text())

    def test_rewriting_an_existing_bundle_replaces_it(self):
        bundle, work = self.check_bundle("llama-3.1-70b-training")
        (bundle / "stale").write_text("x")
        work.write_bundle()
        self.assertFalse((bundle / "stale").exists())


if __name__ == "__main__":
    unittest.main()
