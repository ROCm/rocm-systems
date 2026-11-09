# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Architecture templates preserve their own strict panel contracts."""

import sys
from pathlib import Path

import pytest
import yaml
from config_management import (
    parse_config_template,
    utils_ruamel,
    verify_against_config_template,
)

MINIMAL_PANEL = """\
Panel Config:
  id: 200
  title: Navi48 Metrics
  alias: sol
  data source:
  - metric_table:
      id: 201
      title: System Speed-of-Light
      header:
        metric: Metric
        value: Avg
      metric: {}
  metrics_description: {}
"""


@pytest.fixture
def navi48_configs(tmp_path: Path) -> Path:
    arch_dir = tmp_path / "gfx1201"
    arch_dir.mkdir()
    (arch_dir / "0200_metrics.yaml").write_text(MINIMAL_PANEL, encoding="utf-8")
    panels = parse_config_template.build_template_from_directory(arch_dir, None)
    utils_ruamel.save_yaml(
        {"panels": panels}, tmp_path / "gfx1201_config_template.yaml"
    )
    return tmp_path


@pytest.mark.parametrize("gpu_arch", ["gfx1201", "gfx1151", "gfx942"])
def test_automatic_architecture_template_precedes_family_template(tmp_path, gpu_arch):
    arch_dir = tmp_path / gpu_arch
    template = tmp_path / f"{gpu_arch}_config_template.yaml"
    template.touch()

    selected = verify_against_config_template.template_for_arch(arch_dir)

    assert selected == template


@pytest.mark.parametrize(
    ("gpu_arch", "expected_template"),
    [
        pytest.param("gfx1201", "custom12.yaml", id="navi48_family"),
        pytest.param("gfx1250", "custom12.yaml", id="gfx1250_family"),
        pytest.param("gfx115x", "custom11.yaml", id="rdna35_shared"),
        pytest.param("gfx1151", "custom11.yaml", id="rdna35_variant"),
        pytest.param("gfx942", "custom9.yaml", id="cdna"),
    ],
)
def test_family_template_fallback_preserves_custom_paths(
    tmp_path, gpu_arch, expected_template
):
    selected = verify_against_config_template.template_for_arch(
        tmp_path / gpu_arch,
        Path("custom9.yaml"),
        Path("custom11.yaml"),
        Path("custom12.yaml"),
    )

    assert selected == Path(expected_template)


@pytest.mark.parametrize(
    ("gpu_arch", "template_option"),
    [
        pytest.param("gfx1201", "--gfx12-template", id="navi48"),
        pytest.param("gfx1151", "--gfx11-template", id="rdna35"),
        pytest.param("gfx942", "--gfx9-template", id="cdna"),
    ],
)
def test_explicit_family_template_overrides_architecture_template(
    navi48_configs, gpu_arch, template_option, monkeypatch, capsys
):
    arch_dir = navi48_configs / "gfx1201"
    arch_dir.rename(navi48_configs / gpu_arch)
    automatic_template = navi48_configs / "gfx1201_config_template.yaml"
    automatic_template.rename(navi48_configs / f"{gpu_arch}_config_template.yaml")
    custom_template = navi48_configs / "custom_template.yaml"
    template = yaml.safe_load(
        (navi48_configs / f"{gpu_arch}_config_template.yaml").read_text(
            encoding="utf-8"
        )
    )
    template["panels"][0]["panel_title"] = "Explicit template contract"
    custom_template.write_text(yaml.safe_dump(template), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", str(navi48_configs), template_option, str(custom_template)],
    )

    with pytest.raises(SystemExit) as error:
        verify_against_config_template.main()

    assert error.value.code == 1
    output = capsys.readouterr().out
    assert "Using custom_template.yaml" in output
    assert "Explicit template contract" in output


def test_architecture_template_does_not_require_unrelated_family_templates(
    navi48_configs, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["verify", str(navi48_configs)])

    verify_against_config_template.main()

    assert "PASS [gfx1201]" in capsys.readouterr().out


def test_architecture_template_still_rejects_changed_panel_structure(
    navi48_configs, monkeypatch, capsys
):
    panel_path = navi48_configs / "gfx1201" / "0200_metrics.yaml"
    panel = yaml.safe_load(panel_path.read_text(encoding="utf-8"))
    panel["Panel Config"]["data source"][0]["metric_table"]["title"] = "Wrong table"
    panel_path.write_text(yaml.safe_dump(panel), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["verify", str(navi48_configs)])

    with pytest.raises(SystemExit) as error:
        verify_against_config_template.main()

    assert error.value.code == 1
    assert "Data source #1 mismatch" in capsys.readouterr().out


def test_missing_selected_template_fails_with_architecture_context(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "gfx1201").mkdir()
    monkeypatch.setattr(sys, "argv", ["verify", str(tmp_path)])

    with pytest.raises(SystemExit) as error:
        verify_against_config_template.main()

    assert error.value.code == 1
    assert "gfx1201 template not found" in capsys.readouterr().out
