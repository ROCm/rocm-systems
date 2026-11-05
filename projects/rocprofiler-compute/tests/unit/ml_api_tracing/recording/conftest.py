# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Isolate the registry shared by recording contract and installer tests."""

import pytest

from ml_api_tracing.recording import backend_contract


@pytest.fixture
def fresh_registry(monkeypatch):
    monkeypatch.setattr(backend_contract, "_REGISTRY", {})
    return backend_contract
