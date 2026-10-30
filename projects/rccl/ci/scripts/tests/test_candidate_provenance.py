"""Fail-closed candidate resolution, composition and runtime verification."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import candidate_provenance as cp
import test_madengine as tm

SHA = "a" * 40
OTHER_SHA = "b" * 40


class CandidateResolutionTest(unittest.TestCase):
    def test_develop_run_uses_immutable_event_sha(self):
        self.assertEqual(
            cp.resolve_candidate_sha(
                event_name="schedule",
                github_ref="refs/heads/develop",
                github_sha=SHA,
            ),
            SHA,
        )

    def test_reuse_requires_explicit_exact_sha(self):
        with self.assertRaisesRegex(cp.ProvenanceError, "candidate_sha is required"):
            cp.resolve_candidate_sha(
                event_name="workflow_dispatch",
                github_ref="refs/heads/develop",
                github_sha=SHA,
                artifact_run_id="123",
            )
        self.assertEqual(
            cp.resolve_candidate_sha(
                event_name="workflow_dispatch",
                github_ref="refs/heads/topic",
                github_sha=OTHER_SHA,
                candidate_override=SHA,
                artifact_run_id="123",
            ),
            SHA,
        )

    def test_non_develop_without_two_part_override_fails(self):
        with self.assertRaisesRegex(cp.ProvenanceError, "dispatch this workflow from develop"):
            cp.resolve_candidate_sha(
                event_name="workflow_dispatch",
                github_ref="refs/heads/topic",
                github_sha=SHA,
            )

    def test_override_different_from_run_requires_artifact_id(self):
        with self.assertRaisesRegex(
            cp.ProvenanceError, "candidate_sha differs from this workflow"
        ):
            cp.resolve_candidate_sha(
                event_name="workflow_dispatch",
                github_ref="refs/heads/topic",
                github_sha=OTHER_SHA,
                candidate_override=SHA,
            )

    def test_artifact_run_source_must_match(self):
        run = {
            "head_sha": SHA,
            "html_url": "https://github.com/ROCm/rocm-systems/actions/runs/123",
            "event": "schedule",
            "status": "completed",
            "conclusion": "failure",
        }
        manifest = cp.verify_artifact_run(
            repository="ROCm/rocm-systems",
            run_id="123",
            expected_sha=SHA,
            token="unused",
            run=run,
        )
        self.assertEqual(manifest["artifact_source_sha"], SHA)
        # The overall run may fail in an unrelated matrix leg; exact artifact
        # availability is checked by the following fetch.
        self.assertEqual(manifest["artifact_source_conclusion"], "failure")
        with self.assertRaisesRegex(cp.ProvenanceError, "source mismatch"):
            cp.verify_artifact_run(
                repository="ROCm/rocm-systems",
                run_id="123",
                expected_sha=OTHER_SHA,
                token="unused",
                run=run,
            )


class CandidateCompositionTest(unittest.TestCase):
    def test_manifest_ties_library_and_tuner_to_exact_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib = root / "librccl.so"
            lib.write_bytes(b"candidate")
            tuner = root / "rccl_tuner_gfx950.csv"
            tuner.write_text("collective,minBytes\n")
            artifact = root / "artifact.json"
            artifact.write_text(json.dumps({
                "candidate_sha": SHA,
                "artifact_source_sha": SHA,
                "artifact_run_id": "123",
            }))
            fp = {
                "md5": "m",
                "sha256": tm.sha256_file(lib),
                "version": f"2.40.0-develop:{SHA[:7]}",
                "size": lib.stat().st_size,
            }
            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False):
                manifest = tm.candidate_provenance(SHA, artifact, tuner, lib, fp)
            self.assertEqual(manifest["candidate_sha"], SHA)
            self.assertEqual(manifest["tuner"]["source_sha256"], tm.sha256_file(tuner))
            self.assertEqual(
                manifest["tuner"]["expected_runtime_source"],
                tm.CANDIDATE_TUNER_PATH,
            )

            bad = dict(fp, version=f"2.40.0-develop:{OTHER_SHA[:7]}")
            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    self.assertRaisesRegex(RuntimeError, "does not match"):
                tm.candidate_provenance(SHA, artifact, tuner, lib, bad)

            artifact.write_text(json.dumps({
                "candidate_sha": OTHER_SHA,
                "artifact_source_sha": SHA,
                "artifact_run_id": "123",
            }))
            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                tm.candidate_provenance(SHA, artifact, tuner, lib, fp)

            artifact.write_text(json.dumps({
                "candidate_sha": SHA,
                "artifact_source_sha": SHA,
                "artifact_run_id": "123",
            }))
            dirty = dict(fp, version=f"2.40.0-develop:{SHA[:7]}+")
            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    self.assertRaisesRegex(RuntimeError, "does not match"):
                tm.candidate_provenance(SHA, artifact, tuner, lib, dirty)

            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=True), \
                    mock.patch.object(tm, "rccl_kpack_files", return_value=[]), \
                    self.assertRaisesRegex(RuntimeError, "no .kpack files"):
                tm.candidate_provenance(SHA, artifact, tuner, lib, fp)

    def test_overlay_removes_stock_assets_before_installing_candidate(self):
        snippet = tm._librccl_swap_snippet(kpack=True)
        self.assertIn('rm -f "$d/.kpack"/rccl*.kpack', snippet)
        self.assertIn("RCCL_TUNER_REMOVED.txt", snippet)
        self.assertIn('name "rccl_tuner*.csv"', snippet)
        self.assertIn(tm.CANDIDATE_TUNER_PATH, snippet)
        self.assertLess(
            snippet.index('rm -f "$d/.kpack"/rccl*.kpack'),
            snippet.index('cp "$src" "$d/.kpack/"'),
        )

    def test_sdk_overlay_pairs_kpack_with_each_replaced_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts" / "lib"
            artifacts.mkdir(parents=True)
            lib = artifacts / "librccl.so"
            lib.write_bytes(b"not-an-elf")
            kpack = root / "artifacts" / ".kpack" / "rccl_lib_gfx950.kpack"
            kpack.parent.mkdir()
            kpack.write_bytes(b"kpack")
            tuner = root / "rccl_tuner_gfx950.csv"
            tuner.write_text("collective,minBytes\n")
            work = root / "work"
            work.mkdir()
            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=True):
                tm.stage_rccl_overlay(
                    lib, tuner, "base:image", "gfx950", work, family="megatron"
                )
            dockerfile = (work / tm.OVERLAY_CTX / tm.OVERLAY_DOCKERFILE).read_text()
            self.assertIn('SDK_KPACK_LIB="', dockerfile)
            self.assertIn('SDK_KPACK_DEV="', dockerfile)
            self.assertIn('for d in "$SDK_KPACK_LIB" "$SDK_KPACK_DEV"', dockerfile)
            self.assertTrue(
                (work / tm.OVERLAY_CTX / "rccl_libs" / "tuner" / tuner.name).is_file()
            )

    def test_skip_overlay_bind_mounts_all_assets_and_skips_image_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            (work / "scripts" / "primus_megatron-lm").mkdir(parents=True)
            artifacts = root / "artifacts"
            lib_dir = artifacts / "lib"
            lib_dir.mkdir(parents=True)
            lib = lib_dir / "librccl.so"
            lib.write_bytes(b"candidate")
            kpack = artifacts / ".kpack" / "rccl_lib_gfx950.kpack"
            kpack.parent.mkdir()
            kpack.write_bytes(b"kpack")
            tuner = root / "rccl_tuner_gfx950.csv"
            tuner.write_text("collective,minBytes\n")

            manifest_path = tm.generate_manifest(
                "llama-3.1-70b-training",
                tm.WORKLOAD_CONFIGS["llama-3.1-70b-training"],
                tm.CLUSTER_CONFIGS["ruby"],
                "prebuilt:helper",
                2,
                work,
                rccl_lib=lib,
                candidate_tuner_config=tuner,
                run_dir=tm.prepare_phase_dir(work, tm.CANDIDATE),
            )
            manifest = json.loads(manifest_path.read_text())
            model = manifest["built_models"]["overlay"]
            opts = model["additional_docker_run_options"]
            self.assertIn(f"-v {tuner.resolve()}:{tm.BIND_MOUNT_TUNER_PATH}:ro", opts)
            self.assertIn("_rocm_sdk_libraries/.kpack/rccl_lib_gfx950.kpack", opts)
            self.assertIn("_rocm_sdk_devel/.kpack/rccl_lib_gfx950.kpack", opts)
            for env in (
                manifest["context"]["docker_env_vars"],
                manifest["deployment_config"]["env_vars"],
            ):
                self.assertEqual(
                    env["NCCL_TUNER_CONFIG_FILE"], tm.BIND_MOUNT_TUNER_PATH
                )
                self.assertIn("TUNING", env["NCCL_DEBUG_SUBSYS"])
            self.assertEqual(
                tm.candidate_image_verification_target("prebuilt:helper", True), ""
            )
            self.assertEqual(
                tm.candidate_image_verification_target("built:candidate", False),
                "built:candidate",
            )


class RuntimeVerificationTest(unittest.TestCase):
    def node_logs(self, root: Path, source: str) -> Path:
        logs = root / "slurm_output" / "run_logs" / "42"
        logs.mkdir(parents=True)
        line = (
            "h NCCL INFO RCCL version : 2.40.0-develop:aaaaaaa\n"
            f"h NCCL INFO TUNER/CsvTuner: Loaded 4 tuning configurations from {source}\n"
        )
        for node in range(4):
            (logs / f"prefill_NODE{node}.log").write_text(line)
        return root

    def test_candidate_source_passes_and_stale_stock_source_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.node_logs(Path(tmp), tm.CANDIDATE_TUNER_PATH)
            ok, message = tm.verify_candidate_tuner(run, 4)
            self.assertTrue(ok, message)
            stale = "/opt/rocm/share/rccl/tuner/rccl_tuner_gfx950.csv"
            (run / "slurm_output/run_logs/42/prefill_NODE3.log").write_text(
                f"h NCCL INFO TUNER/CsvTuner: Loaded 4 tuning configurations from {stale}\n"
            )
            ok, message = tm.verify_candidate_tuner(run, 4)
            self.assertFalse(ok)
            self.assertIn("stock disk CSV", message)

    def test_image_digest_must_match_on_every_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_id = "sha256:" + "c" * 64
            tuner_sha = "e" * 64
            for node in range(4):
                (root / f"n{node}").write_text(
                    f"n{node}\t{image_id}\t{tuner_sha}\n"
                )
            ok, _, got = tm.verify_image_identities(root, 4, tuner_sha)
            self.assertTrue(ok)
            self.assertEqual(got, image_id)
            (root / "n3").write_text(
                "n3\tsha256:" + "d" * 64 + f"\t{tuner_sha}\n"
            )
            ok, message, _ = tm.verify_image_identities(root, 4, tuner_sha)
            self.assertFalse(ok)
            self.assertIn("mismatch", message)


if __name__ == "__main__":
    unittest.main()
