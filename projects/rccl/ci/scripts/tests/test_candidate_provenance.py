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
    def test_fresh_external_work_dir_is_created_before_install(self):
        class InstallReached(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "new" / "work"
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            lib = artifact_dir / "librccl.so"
            lib.write_bytes(b"candidate")
            tuner = root / "rccl_tuner_gfx950.csv"
            tuner.write_text("collective,minBytes\n")
            artifact = root / "candidate-artifact-run.json"
            artifact.write_text(json.dumps({
                "candidate_sha": SHA,
                "artifact_source_sha": SHA,
                "artifact_run_id": "123",
            }))
            fingerprint = {
                "md5": "m",
                "sha256": tm.sha256_file(lib),
                "version": f"2.40.0-develop:{SHA[:7]}",
                "size": lib.stat().st_size,
            }
            argv = [
                "test_madengine.py",
                "--artifact-dir", os.fspath(artifact_dir),
                "--candidate-sha", SHA,
                "--artifact-provenance", os.fspath(artifact),
                "--candidate-tuner-config", os.fspath(tuner),
                "--workload", "llama-3.1-70b-training",
                "--cluster", "ruby",
                "--nodes", "2",
                "--work-dir", os.fspath(work),
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(tm, "find_rccl_library", return_value=lib), \
                    mock.patch.object(tm, "get_rccl_fingerprint",
                                      return_value=fingerprint), \
                    mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    mock.patch.object(tm, "rccl_kpack_files", return_value=[]), \
                    mock.patch.object(tm, "install_madengine",
                                      side_effect=InstallReached), \
                    self.assertRaises(InstallReached):
                tm.main()

            self.assertTrue(work.is_dir())
            self.assertEqual(work.stat().st_mode & 0o777, 0o755)
            self.assertTrue((work / tm.CANDIDATE_PROVENANCE).is_file())

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
            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    mock.patch.object(tm, "rccl_kpack_files", return_value=[]):
                manifest = tm.candidate_provenance(SHA, artifact, tuner, lib, fp)
            self.assertEqual(manifest["candidate_sha"], SHA)
            self.assertEqual(manifest["tuner"]["source_sha256"], tm.sha256_file(tuner))
            self.assertEqual(
                manifest["tuner"]["expected_runtime_source"],
                tm.CANDIDATE_TUNER_PATH,
            )

            bad = dict(fp, version=f"2.40.0-develop:{OTHER_SHA[:7]}")
            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    mock.patch.object(tm, "rccl_kpack_files", return_value=[]), \
                    self.assertRaisesRegex(RuntimeError, "does not match"):
                tm.candidate_provenance(SHA, artifact, tuner, lib, bad)

            artifact.write_text(json.dumps({
                "candidate_sha": OTHER_SHA,
                "artifact_source_sha": SHA,
                "artifact_run_id": "123",
            }))
            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    mock.patch.object(tm, "rccl_kpack_files", return_value=[]), \
                    self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                tm.candidate_provenance(SHA, artifact, tuner, lib, fp)

            artifact.write_text(json.dumps({
                "candidate_sha": SHA,
                "artifact_source_sha": SHA,
                "artifact_run_id": "123",
            }))
            dirty = dict(fp, version=f"2.40.0-develop:{SHA[:7]}+")
            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    mock.patch.object(tm, "rccl_kpack_files", return_value=[]), \
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

    def test_reused_workdir_stages_exact_current_asset_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            work.mkdir()

            def artifact(name: str, lib_bytes: bytes, kpack_names: list[str],
                         tuner_bytes: bytes) -> tuple[Path, Path]:
                artifact_root = root / name
                lib_dir = artifact_root / "lib"
                lib_dir.mkdir(parents=True)
                lib = lib_dir / "librccl.so"
                lib.write_bytes(lib_bytes)
                (lib_dir / "librccl-extra.so").write_bytes(lib_bytes + b"-extra")
                kpack_dir = artifact_root / ".kpack"
                kpack_dir.mkdir()
                for kpack_name in kpack_names:
                    (kpack_dir / kpack_name).write_bytes(
                        f"{name}:{kpack_name}".encode()
                    )
                tuner = artifact_root / "rccl_tuner_gfx950.csv"
                tuner.write_bytes(tuner_bytes)
                return lib, tuner

            lib_a, tuner_a = artifact(
                "a", b"lib-a", ["rccl_old.kpack", "rccl_removed.kpack"], b"tuner-a"
            )
            lib_b, tuner_b = artifact(
                "b", b"lib-b", ["rccl_new.kpack"], b"tuner-b"
            )
            # The second artifact no longer contains this A-only library.
            (lib_b.parent / "librccl-extra.so").unlink()

            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=True), \
                    mock.patch.object(tm, "_needed_sonames", return_value=[]):
                tm.stage_rccl_overlay(
                    lib_a, tuner_a, "base:image", "gfx950", work,
                    family="megatron",
                )
                tm.stage_rccl_overlay(
                    lib_b, tuner_b, "base:image", "gfx950", work,
                    family="megatron",
                )

            staging = work / tm.OVERLAY_CTX / "rccl_libs"
            expected_sources = tm.expected_overlay_asset_sources(lib_b, tuner_b)
            staged = tm.verify_staged_overlay_assets(staging, expected_sources)
            self.assertEqual(
                {item["path"] for item in staged["files"]},
                {
                    "librccl.so",
                    ".kpack/rccl_new.kpack",
                    "tuner/rccl_tuner_gfx950.csv",
                },
            )
            self.assertEqual((staging / "librccl.so").read_bytes(), b"lib-b")
            self.assertEqual(
                (staging / "tuner" / tuner_b.name).read_bytes(), b"tuner-b"
            )
            self.assertFalse((staging / ".kpack" / "rccl_old.kpack").exists())
            self.assertFalse((staging / ".kpack" / "rccl_removed.kpack").exists())
            self.assertFalse((staging / "librccl-extra.so").exists())
            self.assertEqual(
                staged["asset_id"], tm.rccl_asset_identity(lib_b, tuner_b)
            )
            for item in staged["files"]:
                self.assertEqual(
                    (staging / item["path"]).stat().st_mode & 0o777,
                    0o644,
                )

    def test_reused_workdir_rejects_symlinked_staging_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            context = work / tm.OVERLAY_CTX
            context.mkdir(parents=True)
            outside = root / "outside"
            outside.mkdir()
            marker = outside / "keep"
            marker.write_text("keep\n")
            (context / "rccl_libs").symlink_to(outside, target_is_directory=True)
            lib_dir = root / "artifact" / "lib"
            lib_dir.mkdir(parents=True)
            lib = lib_dir / "librccl.so"
            lib.write_bytes(b"lib")
            tuner = root / "artifact" / "rccl_tuner_gfx950.csv"
            tuner.write_text("tuner\n")

            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    mock.patch.object(tm, "_needed_sonames", return_value=[]), \
                    self.assertRaisesRegex(RuntimeError, "untrusted.*staging"):
                tm.stage_rccl_overlay(
                    lib, tuner, "base:image", "gfx950", work,
                    family="megatron",
                )
            self.assertEqual(marker.read_text(), "keep\n")
            self.assertTrue((context / "rccl_libs").is_symlink())

    def test_reused_workdir_rejects_symlinked_overlay_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            work.mkdir()
            outside = root / "outside"
            outside.mkdir()
            marker = outside / "keep"
            marker.write_text("keep\n")
            (work / tm.OVERLAY_CTX).symlink_to(outside, target_is_directory=True)
            lib_dir = root / "artifact" / "lib"
            lib_dir.mkdir(parents=True)
            lib = lib_dir / "librccl.so"
            lib.write_bytes(b"lib")
            tuner = root / "artifact" / "rccl_tuner_gfx950.csv"
            tuner.write_text("tuner\n")

            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    mock.patch.object(tm, "_needed_sonames", return_value=[]), \
                    self.assertRaisesRegex(RuntimeError, "untrusted.*context"):
                tm.stage_rccl_overlay(
                    lib, tuner, "base:image", "gfx950", work,
                    family="megatron",
                )
            self.assertEqual(marker.read_text(), "keep\n")
            self.assertTrue((work / tm.OVERLAY_CTX).is_symlink())

    def test_same_candidate_on_different_base_has_different_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            work.mkdir()
            lib_dir = root / "artifact" / "lib"
            lib_dir.mkdir(parents=True)
            lib = lib_dir / "librccl.so"
            lib.write_bytes(b"candidate")
            tuner = root / "artifact" / "rccl_tuner_gfx950.csv"
            tuner.write_text("tuner\n")
            base_a = "rocm/primus@sha256:" + "a" * 64
            base_b = "rocm/primus@sha256:" + "b" * 64

            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    mock.patch.object(tm, "_needed_sonames", return_value=[]):
                tag_a, staged_a = tm.stage_rccl_overlay(
                    lib, tuner, "rocm/primus:v26.4", "gfx950", work,
                    family="megatron", from_ref=base_a,
                )
                tag_b, staged_b = tm.stage_rccl_overlay(
                    lib, tuner, "rocm/primus:v26.4", "gfx950", work,
                    family="megatron", from_ref=base_b,
                )

            self.assertNotEqual(tag_a, tag_b)
            self.assertNotEqual(
                staged_a["overlay_asset_id"], staged_b["overlay_asset_id"]
            )
            self.assertNotEqual(
                tm.candidate_overlay_identity("asset", base_b, "a" * 64),
                tm.candidate_overlay_identity("asset", base_b, "b" * 64),
            )
            self.assertEqual(staged_b["base_digest"], base_b)
            self.assertEqual(len(staged_b["recipe_sha256"]), 64)
            dockerfile = (
                work / tm.OVERLAY_CTX / tm.OVERLAY_DOCKERFILE
            ).read_text()
            self.assertIn(f"FROM {base_b}\n", dockerfile)
            self.assertIn(
                f'LABEL rccl.candidate.base-digest="{base_b}"',
                dockerfile,
            )
            self.assertIn(
                f'LABEL rccl.candidate.overlay-id="'
                f'{staged_b["overlay_asset_id"]}"',
                dockerfile,
            )
            self.assertIn(
                f'LABEL rccl.candidate.recipe-sha256="'
                f'{staged_b["recipe_sha256"]}"',
                dockerfile,
            )

            (work / "scripts" / "primus_megatron-lm").mkdir(parents=True)
            baseline_manifest = tm.generate_manifest(
                "llama-3.1-70b-training",
                tm.WORKLOAD_CONFIGS["llama-3.1-70b-training"],
                tm.CLUSTER_CONFIGS["ruby"],
                base_b,
                2,
                work,
                run_dir=tm.prepare_phase_dir(work, tm.BASELINE),
                pull_only=True,
            )
            baseline = json.loads(baseline_manifest.read_text())
            self.assertEqual(
                baseline["built_images"]["overlay"]["docker_image"], base_b
            )

    def test_maxtext_identity_records_exact_staged_primus_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            work.mkdir()
            lib_dir = root / "artifact" / "lib"
            lib_dir.mkdir(parents=True)
            lib = lib_dir / "librccl.so"
            lib.write_bytes(b"candidate")
            tuner = root / "artifact" / "rccl_tuner_gfx950.csv"
            tuner.write_text("tuner\n")
            primus = root / "Primus"
            primus.mkdir()
            (primus / "primus-cli").write_text("#!/bin/sh\n")
            (primus / "requirements-jax.txt").write_text("jax\n")
            (primus / "payload.txt").write_text("one\n")
            base = "rocm/jax-training@sha256:" + "a" * 64

            with mock.patch.object(tm, "_rccl_uses_kpack", return_value=False), \
                    mock.patch.object(tm, "_needed_sonames", return_value=[]):
                _, staged = tm.stage_rccl_overlay(
                    lib, tuner, "rocm/jax-training:maxtext-v26.6", "gfx950",
                    work, family="maxtext", primus_dir=primus, from_ref=base,
                )
            staged_primus = work / tm.OVERLAY_CTX / "Primus"
            self.assertEqual(
                staged["primus_tree_sha256"],
                tm.build_context_tree_identity(staged_primus),
            )
            dockerfile_text = (
                work / tm.OVERLAY_CTX / tm.OVERLAY_DOCKERFILE
            ).read_text()
            label_free_recipe = "".join(
                line for line in dockerfile_text.splitlines(keepends=True)
                if not line.startswith("LABEL rccl.candidate.")
            )
            self.assertEqual(
                staged["recipe_sha256"],
                tm.candidate_recipe_identity(
                    label_free_recipe, staged["primus_tree_sha256"]
                ),
            )
            original_identity = staged["overlay_asset_id"]
            payload = staged_primus / "payload.txt"
            payload.chmod(0o700)
            mode_changed_tree = tm.build_context_tree_identity(staged_primus)
            self.assertNotEqual(
                mode_changed_tree, staged["primus_tree_sha256"]
            )
            payload.chmod(0o644)
            (staged_primus / "payload.txt").write_text("two\n")
            changed_tree = tm.build_context_tree_identity(staged_primus)
            self.assertNotEqual(changed_tree, staged["primus_tree_sha256"])
            self.assertNotEqual(
                tm.candidate_overlay_identity(
                    staged["candidate_asset_id"],
                    base,
                    tm.candidate_recipe_identity(
                        "same recipe", staged["primus_tree_sha256"]
                    ),
                ),
                tm.candidate_overlay_identity(
                    staged["candidate_asset_id"],
                    base,
                    tm.candidate_recipe_identity("same recipe", changed_tree),
                ),
            )
            self.assertRegex(original_identity, r"^[0-9a-f]{16}$")

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
            base_digest = "rocm/primus@sha256:" + "a" * 64
            overlay_id = "f" * 16
            for node in range(4):
                (root / f"n{node}").write_text(
                    f"n{node}\t{image_id}\t{tuner_sha}\t"
                    f"{base_digest}\t{overlay_id}\n"
                )
            ok, _, got = tm.verify_image_identities(
                root, 4, tuner_sha, base_digest, overlay_id
            )
            self.assertTrue(ok)
            self.assertEqual(got, image_id)
            (root / "n3").write_text(
                "n3\tsha256:" + "d" * 64
                + f"\t{tuner_sha}\t{base_digest}\t{overlay_id}\n"
            )
            ok, message, _ = tm.verify_image_identities(
                root, 4, tuner_sha, base_digest, overlay_id
            )
            self.assertFalse(ok)
            self.assertIn("mismatch", message)

            (root / "n3").write_text(
                f"n3\t{image_id}\t{tuner_sha}\t"
                f"rocm/primus@sha256:{'b' * 64}\t{overlay_id}\n"
            )
            ok, message, _ = tm.verify_image_identities(
                root, 4, tuner_sha, base_digest, overlay_id
            )
            self.assertFalse(ok)
            self.assertIn("base digest mismatch", message)


if __name__ == "__main__":
    unittest.main()
