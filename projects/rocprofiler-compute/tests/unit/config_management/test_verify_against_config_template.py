# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Mapping from an arch directory to its analysis-config template."""

import sys
from pathlib import Path

import pytest
from config_management.verify_against_config_template import (
    GFX120X_TEMPLATE_FILENAME,
    GFX125X_TEMPLATE_FILENAME,
    gfx12_template_filename,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_CONFIGS = REPO_ROOT / "src" / "rocprof_compute_soc" / "analysis_configs"
HBM_TITLES = (
    "EA to HBM",
    "GL2C-EA Interface - HBM Bandwidth",
)


def _write_empty_template(path: Path) -> None:
    path.write_text("panels: []\n", encoding="utf-8")


def _run_verifier(configs: Path, monkeypatch, capsys) -> str:
    """Run the verifier on configs and return stdout. Expects failure."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_against_config_template.py", str(configs)],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    return capsys.readouterr().out


def test_gfx1201_maps_only_to_the_gfx120x_template():
    assert gfx12_template_filename("gfx1201") == GFX120X_TEMPLATE_FILENAME
    assert gfx12_template_filename("gfx1200") == GFX120X_TEMPLATE_FILENAME
    assert gfx12_template_filename("gfx120f") == GFX120X_TEMPLATE_FILENAME
    assert gfx12_template_filename("GFX1201") == GFX120X_TEMPLATE_FILENAME


def test_gfx1250_maps_only_to_the_gfx125x_template():
    assert gfx12_template_filename("gfx1250") == GFX125X_TEMPLATE_FILENAME
    assert gfx12_template_filename("gfx125f") == GFX125X_TEMPLATE_FILENAME


def test_gfx120_and_gfx125_do_not_share_a_template():
    assert gfx12_template_filename("gfx1201") != gfx12_template_filename("gfx1250")


@pytest.mark.parametrize(
    "arch",
    ["gfx1210", "gfx1220", "gfx1260", "gfx12ab", "gfx129f"],
)
def test_unknown_gfx12_arch_fails_instead_of_picking_a_family(arch):
    with pytest.raises(ValueError, match="no analysis-config template"):
        gfx12_template_filename(arch)


@pytest.mark.parametrize(
    "arch",
    ["gfx1151", "gfx942", "gfx908", "gfx120", "gfx125", "gfx12"],
)
def test_non_gfx12xx_arch_is_not_a_gfx12_family(arch):
    assert gfx12_template_filename(arch) is None


def test_unknown_gfx12_directory_does_not_use_either_family(
    tmp_path, capsys, monkeypatch
):
    configs = tmp_path / "analysis_configs"
    configs.mkdir()
    _write_empty_template(configs / "gfx9_config_template.yaml")
    _write_empty_template(configs / GFX120X_TEMPLATE_FILENAME)
    _write_empty_template(configs / GFX125X_TEMPLATE_FILENAME)
    (configs / "gfx1210").mkdir()
    output = _run_verifier(configs, monkeypatch, capsys)
    assert "no analysis-config template for gfx12 arch gfx1210" in output
    assert "Using gfx9 template" not in output
    assert "Using gfx120x template" not in output
    assert "Using gfx125x template" not in output


def test_gfx1201_does_not_fall_back_to_the_gfx125x_template(
    tmp_path, capsys, monkeypatch
):
    configs = tmp_path / "analysis_configs"
    configs.mkdir()
    _write_empty_template(configs / "gfx9_config_template.yaml")
    _write_empty_template(configs / GFX125X_TEMPLATE_FILENAME)
    (configs / "gfx1201").mkdir()
    output = _run_verifier(configs, monkeypatch, capsys)
    assert "gfx120x architecture(s) present but template missing" in output
    assert "Using gfx125x template" not in output


def test_family_templates_keep_the_shared_hbm_titles():
    """The split copies the gfx12 contract and does not rename HBM titles."""
    gfx120x = (ANALYSIS_CONFIGS / GFX120X_TEMPLATE_FILENAME).read_bytes()
    gfx125x = (ANALYSIS_CONFIGS / GFX125X_TEMPLATE_FILENAME).read_bytes()
    assert gfx120x == gfx125x
    text = gfx120x.decode("utf-8")
    for title in HBM_TITLES:
        assert title in text
    assert not (ANALYSIS_CONFIGS / "gfx12_config_template.yaml").exists()
    for arch in ("gfx1201", "gfx1250"):
        panel_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ANALYSIS_CONFIGS / arch).glob("*.yaml")
        )
        for title in HBM_TITLES:
            assert title in panel_text
