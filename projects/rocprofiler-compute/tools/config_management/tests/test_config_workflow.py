##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import master_config_workflow_script as mws  # noqa


def build_panel_dict(
    panel_id: int, title: str, tables: tuple, descriptions: dict = None
) -> dict:
    """
    tables: tuple of (table_id, table_title, metrics_dict)
    metrics_dict example:
      {
        "Metric A": {"avg": "AVG(A)", "min": "MIN(A)", "max": "MAX(A)", "unit": "pct"},
        ...
      }
    """
    data_sources = []
    for tid, ttitle, metrics in tables:
        data_sources.append({
            "metric_table": {
                "id": tid,
                "title": ttitle,
                "header": {
                    "metric": "Metric",
                    "avg": "Avg",
                    "min": "Min",
                    "max": "Max",
                    "unit": "Unit",
                },
                "metric": metrics or {},
            }
        })

    panel = {
        "Panel Config": {
            "id": panel_id,
            "title": title,
            "data source": data_sources,
        }
    }
    if descriptions:
        panel["Panel Config"]["metrics_description"] = descriptions
    return panel


def write_yaml(path: Path, obj: dict) -> None:
    path.write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


class TestUserFlows(unittest.TestCase):
    """
    These tests assert the interactive user flows are wired correctly:
      - When a new delta is detected (for latest), we ask 1) in-place vs 2) create-new
      - When a YAML is edited (for latest), we ask 1) in-place vs 2) create-new
      - For older-arch edits, we do NOT ask to create-new; only confirm in-place
    We patch the master script’s helpers to avoid running subprocesses.
    """

    @patch.object(mws, "validate_delta_structure", return_value=(True, ""))
    @patch.object(mws, "update_latest_arch_from_delta", return_value=True)
    @patch.object(mws, "promote_new_arch_from_delta", return_value=True)
    @patch.object(mws, "get_latest_arch", return_value="gfx950")
    @patch.object(mws, "prompt_yes_no", return_value=True)
    def test_delta_on_latest_branching(
        self, _yesno, _get_latest, promo_new_from_delta, update_latest_delta, _validate
    ):
        # choice 1: in-place
        with patch("builtins.input", side_effect=["1"]):
            ok = mws.handle_delta_file(
                "/tmp/fake.yaml",
                "gfx950",
                {"paths": {"template": "T", "configs_root": "C"}},
                dry_run=False,
            )
            assert ok
            update_latest_delta.assert_called_once()
            promo_new_from_delta.assert_not_called()

        update_latest_delta.reset_mock()
        promo_new_from_delta.reset_mock()

        # choice 2: create new arch
        with patch("builtins.input", side_effect=["2", "gfx955"]):
            ok = mws.handle_delta_file(
                "/tmp/fake.yaml",
                "gfx950",
                {"paths": {"template": "T", "configs_root": "C"}},
                dry_run=False,
            )
            assert ok
            promo_new_from_delta.assert_called_once_with(
                "gfx950",
                "gfx955",
                "/tmp/fake.yaml",
                {"paths": {"template": "T", "configs_root": "C"}},
            )
            update_latest_delta.assert_not_called()

    @patch.object(mws, "update_latest_arch_from_edits", return_value=True)
    @patch.object(mws, "promote_new_arch_from_latest_edits", return_value=True)
    @patch.object(mws, "get_latest_arch", return_value="gfx950")
    @patch.object(mws, "prompt_yes_no", return_value=True)
    def test_direct_edits_on_latest_branching(
        self, _yesno, _get_latest, promo_from_edits, update_latest_edits
    ):
        # choice 1: in-place update latest
        with patch("builtins.input", side_effect=["1"]):
            ok = mws.handle_direct_edits(
                "gfx950",
                ["file.yaml"],
                {"paths": {"template": "T", "configs_root": "C"}},
                dry_run=False,
            )
            self.assertTrue(ok)
            update_latest_edits.assert_called_once_with(
                "gfx950", {"paths": {"template": "T", "configs_root": "C"}}
            )
            promo_from_edits.assert_not_called()

        update_latest_edits.reset_mock()
        promo_from_edits.reset_mock()

        # choice 2: promote a new arch from edits
        with patch("builtins.input", side_effect=["2", "gfx955"]):
            ok = mws.handle_direct_edits(
                "gfx950",
                ["file.yaml"],
                {"paths": {"template": "T", "configs_root": "C"}},
                dry_run=False,
            )
            self.assertTrue(ok)
            promo_from_edits.assert_called_once_with(
                "gfx950", "gfx955", {"paths": {"template": "T", "configs_root": "C"}}
            )
            update_latest_edits.assert_not_called()

    @patch.object(mws, "get_latest_arch", return_value="gfx950")
    @patch.object(mws, "prompt_yes_no", return_value=True)
    def test_edits_on_older_arch_no_create_new_prompt(self, _yesno, _get_latest):
        # For older arch (e.g., gfx940), we should NOT prompt for 1/2 input branch.
        with (
            patch("builtins.input") as mock_input,
            patch.object(mws, "update_older_arch_from_edits", return_value=True) as upd,
        ):
            ok = mws.handle_direct_edits(
                "gfx940",
                ["file.yaml"],
                {"paths": {"template": "T", "configs_root": "C"}},
                dry_run=False,
            )
            self.assertTrue(ok)
            upd.assert_called_once()
            mock_input.assert_not_called()

    @patch.object(mws, "validate_delta_structure", return_value=(True, ""))
    @patch.object(mws, "get_latest_arch", return_value="gfx950")
    @patch.object(mws, "prompt_yes_no", return_value=True)
    def test_delta_on_older_arch_in_place_only(self, _yesno, _get_latest, _valid):
        with (
            patch("builtins.input") as mock_input,
            patch.object(mws, "update_older_arch_from_delta", return_value=True) as upd,
        ):
            ok = mws.handle_delta_file(
                "/tmp/old_delta.yaml",
                "gfx940",
                {"paths": {"template": "T", "configs_root": "C"}},
                dry_run=False,
            )
            self.assertTrue(ok)
            upd.assert_called_once()
            mock_input.assert_not_called()


class TestDeltaAndEditsSemantics(unittest.TestCase):
    """
    End-to-end tests for:
      - generating a delta (add/del/mod of metrics + descriptions) and applying it
      - detecting edits (add/del/mod) via the delta generator
    These use the generate_config_deltas.py and apply_config_deltas.py scripts directly
    with a temporary file layout.
    """

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="cfgwf_"))
        # Create minimal directory layout
        self.configs_root = (
            self.tmpdir / "src" / "rocprof_compute_soc" / "analysis_configs"
        )
        self.configs_root.mkdir(parents=True, exist_ok=True)

        self.gfx_prev = self.configs_root / "gfx950"
        self.gfx_curr = self.configs_root / "gfx955"
        self.gfx_prev.mkdir()
        self.gfx_curr.mkdir()

        # One shared yaml filename in both dirs (panel 1400)
        self.file_name = "1400_scalar_l1_data_cache.yaml"

        # Previous (baseline) YAML: has metric A and description
        prev_obj = build_panel_dict(
            1401,
            "Scalar L1D Speed-of-Light",
            tables=(
                (
                    1401,
                    "Scalar L1D SoL Table",
                    {
                        "Metric A": {
                            "avg": "AVG(A)",
                            "min": "MIN(A)",
                            "max": "MAX(A)",
                            "unit": "pct",
                        },
                    },
                ),
            ),
            descriptions={"Metric A": {"plain": "A plain", "rst": "A rst"}},
        )
        write_yaml(self.gfx_prev / self.file_name, prev_obj)

        curr_obj = build_panel_dict(
            1401,
            "Scalar L1D Speed-of-Light",
            tables=(
                (
                    1401,
                    "Scalar L1D SoL Table",
                    {
                        "Metric A": {
                            "avg": "AVG(A_new)",
                            "min": "MIN(A)",
                            "max": "MAX(A)",
                            "unit": "pct",
                        },  # MOD
                        "Metric B": {
                            "avg": "AVG(B)",
                            "min": "MIN(B)",
                            "max": "MAX(B)",
                            "unit": "cycles",
                        },  # ADD
                    },
                ),
            ),
            descriptions={
                "Metric A": {"plain": "A plain (new)", "rst": "A rst (new)"},  # MOD
                "Metric B": {"plain": "B plain", "rst": "B rst"},  # ADD
            },
        )
        write_yaml(self.gfx_curr / self.file_name, curr_obj)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_generate_delta_and_apply_roundtrip(self):
        """
        1) Generate delta: curr (gfx955) vs prev (gfx950) -> stored in prev/config_delta
        2) Apply delta onto a copy of prev -> must equal curr
        """
        # Run generator
        # (call the script's main via subprocess to mimic actual behavior)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "generate_config_deltas.py"),
            str(self.gfx_curr),
            str(self.gfx_prev),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        # Find delta file
        delta_dir = self.gfx_prev / "config_delta"
        self.assertTrue(delta_dir.is_dir(), "config_delta directory not created")
        deltas = sorted(delta_dir.glob("*_diff.yaml"))

        delta_text = (
            self.gfx_prev
            / "config_delta"
            / sorted((self.gfx_prev / "config_delta").glob("*_diff.yaml"))[-1].name
        ).read_text(encoding="utf-8")
        assert "AVG(A_new)" in delta_text, (
            f"Delta is missing the expected modification:\n{delta_text}"
        )

        self.assertTrue(deltas, "No delta file created")
        delta_file = deltas[-1]

        # Apply delta to a clone of prev -> expect to match curr
        out_clone = self.tmpdir / "out_clone"
        out_clone.mkdir()
        # Use apply_config_deltas.py exactly as workflow does
        cmd2 = [
            sys.executable,
            str(REPO_ROOT / "apply_config_deltas.py"),
            str(self.gfx_prev),
            str(delta_file),
            str(out_clone),
        ]
        res2 = subprocess.run(cmd2, capture_output=True, text=True)
        self.assertEqual(res2.returncode, 0, msg=res2.stderr)

        # Compare resulting YAML with curr YAML
        produced = (out_clone / self.file_name).read_text(encoding="utf-8")
        expected = (self.gfx_curr / self.file_name).read_text(encoding="utf-8")

        self.assertEqual(yaml.safe_load(produced), yaml.safe_load(expected))

    def test_delta_semantics_add_del_mod(self):
        """
        Read the generated delta and ensure categories capture:
          - Modification of Metric A (avg + description)
          - Addition of Metric B (metric + description)
        """
        # Generate delta
        cmd = [
            sys.executable,
            str(REPO_ROOT / "generate_config_deltas.py"),
            str(self.gfx_curr),
            str(self.gfx_prev),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        delta_dir = self.gfx_prev / "config_delta"
        delta_file = sorted(delta_dir.glob("*_diff.yaml"))[-1]
        delta_text = delta_file.read_text(encoding="utf-8")

        # Basic sanity: categories present
        self.assertIn("Addition:", delta_text)
        self.assertIn("Deletion:", delta_text)
        self.assertIn("Modification:", delta_text)

        # Additions should include Metric B and its description
        self.assertIn("Metric B:", delta_text)
        self.assertIn("B plain", delta_text)
        self.assertIn("B rst", delta_text)

        # Modifications should include Metric A changes
        self.assertIn("Metric A:", delta_text)
        self.assertIn("AVG(A_new)", delta_text)
        self.assertIn("A plain (new)", delta_text)
        self.assertIn("A rst (new)", delta_text)

        # No deletions expected for this setup
        # (still check Deletion section exists but may render as [])
        # Ensure there is no "Metric C" ghost, etc.
        self.assertNotIn("Metric C:", delta_text)

    def test_edit_detection_via_delta_generator(self):
        """
        Using the same prev/curr pair, ensure the generator correctly
        identifies additions and modifications (no deletions in this case).
        (This stands in for 'edited existing config yaml'.)
        """
        # Generate delta from prev -> curr (same as edits applied)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "generate_config_deltas.py"),
            str(self.gfx_curr),
            str(self.gfx_prev),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        # Validate expected markers
        delta_dir = self.gfx_prev / "config_delta"
        delta_file = sorted(delta_dir.glob("*_diff.yaml"))[-1]
        txt = delta_file.read_text(encoding="utf-8")

        # Additions: Metric B, Descriptions for B
        self.assertIn("Metric B:", txt)
        self.assertIn("B plain", txt)
        self.assertIn("B rst", txt)

        # Modifications: Metric A avg and descriptions
        self.assertIn("AVG(A_new)", txt)
        self.assertIn("A plain (new)", txt)
        self.assertIn("A rst (new)", txt)
