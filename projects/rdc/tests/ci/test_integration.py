"""
4.3  Integration Tests
======================
Validates Prometheus exporter, Grafana dashboard JSON, and RAS/ECC readiness.
"""

import json
import os
import subprocess
import sys
import pytest


# ---------------------------------------------------------------------------
# Prometheus exporter
# ---------------------------------------------------------------------------
class TestPrometheusExporter:
    """Verify the Prometheus exporter script and config."""

    def test_prometheus_script_exists(self, python_binding_dir):
        script = os.path.join(python_binding_dir, "rdc_prometheus.py")
        assert os.path.isfile(script), "rdc_prometheus.py not found"

    def test_prometheus_help(self, python_binding_dir):
        script = os.path.join(python_binding_dir, "rdc_prometheus.py")
        r = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=python_binding_dir,
        )
        combined = (r.stdout + r.stderr).lower()
        assert r.returncode == 0, r.stderr
        assert "listen_port" in combined or "usage" in combined

    def test_prometheus_config_valid_json(self, python_binding_dir):
        cfg = os.path.join(python_binding_dir, "prometheus_targets.json")
        if not os.path.isfile(cfg):
            pytest.skip("prometheus_targets.json not present")
        with open(cfg) as f:
            data = json.load(f)
        assert isinstance(data, (list, dict)), "Unexpected JSON structure"

    def test_prometheus_example_yml_exists(self, python_binding_dir):
        cfg = os.path.join(python_binding_dir, "rdc_prometheus_example.yml")
        assert os.path.isfile(cfg), "rdc_prometheus_example.yml not found"


# ---------------------------------------------------------------------------
# Grafana dashboard
# ---------------------------------------------------------------------------
class TestGrafanaDashboard:
    """Verify the Grafana dashboard JSON is valid and well-formed."""

    def test_dashboard_json_valid(self, python_binding_dir):
        path = os.path.join(python_binding_dir, "rdc_grafana_dashboard_example.json")
        if not os.path.isfile(path):
            pytest.skip("Grafana dashboard JSON not found")
        with open(path) as f:
            data = json.load(f)
        assert "panels" in data or "rows" in data or "title" in data, (
            "Dashboard JSON missing expected Grafana keys"
        )

    def test_dashboard_has_panels(self, python_binding_dir):
        path = os.path.join(python_binding_dir, "rdc_grafana_dashboard_example.json")
        if not os.path.isfile(path):
            pytest.skip("Grafana dashboard JSON not found")
        with open(path) as f:
            data = json.load(f)
        panels = data.get("panels", [])
        assert len(panels) > 0, "Dashboard has no panels"


# ---------------------------------------------------------------------------
# RAS / ECC field support
# ---------------------------------------------------------------------------
class TestRASFields:
    """Verify RAS/ECC field IDs are defined in the Python bindings."""

    def test_ecc_fields_defined(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from rdc_bootstrap import rdc_field_t

            ecc_fields = [attr for attr in dir(rdc_field_t) if "ECC" in attr]
            assert len(ecc_fields) > 0, "No ECC field IDs found in rdc_field_t"
        finally:
            sys.path.pop(0)

    def test_xgmi_fields_defined(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from rdc_bootstrap import rdc_field_t

            xgmi_fields = [attr for attr in dir(rdc_field_t) if "XGMI" in attr]
            assert len(xgmi_fields) > 0, "No XGMI field IDs found in rdc_field_t"
        finally:
            sys.path.pop(0)


# ---------------------------------------------------------------------------
# Collectd plugin
# ---------------------------------------------------------------------------
class TestCollectdPlugin:
    """Verify collectd plugin files are present and syntactically valid."""

    def test_collectd_script_exists(self, python_binding_dir):
        script = os.path.join(python_binding_dir, "rdc_collectd.py")
        assert os.path.isfile(script), "rdc_collectd.py not found"

    def test_collectd_config_exists(self, python_binding_dir):
        conf = os.path.join(python_binding_dir, "rdc_collectd.conf")
        assert os.path.isfile(conf), "rdc_collectd.conf not found"

    def test_collectd_script_compiles(self, python_binding_dir):
        script = os.path.join(python_binding_dir, "rdc_collectd.py")
        with open(script) as f:
            compile(f.read(), script, "exec")
