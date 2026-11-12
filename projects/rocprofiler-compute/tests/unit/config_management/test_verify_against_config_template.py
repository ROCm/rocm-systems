# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Every GFX12 architecture directory resolves to an existing config template."""

import pytest
from config_management import verify_against_config_template as verify

import config

GFX12_ARCHES = sorted(
    p.name
    for p in config.analysis_configs_dir.iterdir()
    if p.is_dir() and verify.is_gfx12_ip_variant(p.name)
)


@pytest.mark.parametrize("arch", GFX12_ARCHES)
def test_gfx12_arch_resolves_to_existing_template(arch):
    path = verify.gfx12_template_path(config.analysis_configs_dir, arch)
    assert path.is_file(), path
