"""Shared utilities for RCCL CI test scripts (JAX, PyTorch)."""

import json
import logging
import os
import smtplib
import sys
import urllib.request
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from pathlib import Path

log = logging.getLogger(__name__)

SMTP_SERVERS = ["smtp.amd.com", "aussmtp.amd.com", "mail.amd.com", "localhost"]


def find_rccl_library(artifact_dir: Path) -> Path:
    """Find librccl.so in the artifact directory tree."""
    matches = list(artifact_dir.rglob("librccl.so"))
    if not matches:
        so_files = list(artifact_dir.rglob("*.so"))[:20]
        log.error("librccl.so not found in %s", artifact_dir)
        log.error("Shared libraries found: %s", [str(f) for f in so_files])
        sys.exit(1)
    lib_path = matches[0].resolve()
    log.info("Found librccl.so at: %s", lib_path)
    return lib_path


def verify_rccl_override(rccl_lib_dir: Path) -> None:
    """Verify that the CI-built librccl.so exists on disk."""
    ci_rccl = rccl_lib_dir.resolve() / "librccl.so"
    if not ci_rccl.exists():
        log.error("CI-built librccl.so not found at %s", ci_rccl)
        sys.exit(1)
    log.info("CI-built RCCL: %s (%d bytes)", ci_rccl, ci_rccl.stat().st_size)


def parse_junit_xml(xml_path: Path) -> dict:
    """Parse JUnit XML and return structured results."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    passed_tests = []
    failed_tests = []
    error_details = []
    tests_run = 0
    failures = 0
    errors = 0

    for suite in root.iter("testsuite"):
        tests_run = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))

    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        time_s = tc.get("time", "")
        duration = f"{float(time_s):.2f}s" if time_s else ""

        failure = tc.find("failure")
        error = tc.find("error")
        if failure is not None:
            failed_tests.append(name)
            error_details.append(
                f"FAILED: {name}\n  {failure.get('message', '')}"
            )
        elif error is not None:
            failed_tests.append(name)
            error_details.append(
                f"ERROR: {name}\n  {error.get('message', '')}"
            )
        else:
            passed_tests.append((name, duration))

    return {
        "passed": passed_tests,
        "failed": failed_tests,
        "error_details": error_details,
        "tests_run": tests_run,
        "failures": failures,
        "errors": errors,
    }


def write_github_summary(report: str) -> None:
    """Write report to GITHUB_STEP_SUMMARY if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write("```\n")
            f.write(report)
            f.write("\n```\n")
        log.info("Summary written to GITHUB_STEP_SUMMARY")


def set_github_output(key: str, value: str) -> None:
    """Write a key=value pair to GITHUB_OUTPUT if available."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")


def send_email_report(
    report: str, recipient: str, status: str, subject_prefix: str
) -> None:
    """Send the summary report via email."""
    subject = f"{subject_prefix}: {status}"
    msg = MIMEText(report)
    msg["Subject"] = subject
    msg["From"] = "rccl-ci@amd.com"
    msg["To"] = recipient

    for server in SMTP_SERVERS:
        try:
            with smtplib.SMTP(server, timeout=10) as s:
                s.sendmail(msg["From"], [recipient], msg.as_string())
            log.info("Email sent to %s via %s", recipient, server)
            return
        except Exception as e:
            log.debug("SMTP %s failed: %s", server, e)
            continue
    log.warning(
        "Could not send email to %s (tried: %s)", recipient, ", ".join(SMTP_SERVERS)
    )


def send_teams_webhook(
    report: str, webhook_url: str, status: str, subject_prefix: str
) -> None:
    """Send the summary report to a Microsoft Teams channel via webhook."""
    color = "Good" if status == "PASSED" else "Attention"
    run_url = os.environ.get("GITHUB_SERVER_URL", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    actions_url = f"{run_url}/{repo}/actions/runs/{run_id}" if run_url else ""

    facts = [{"title": "Status", "value": status}]
    if actions_url:
        facts.append({"title": "Run", "value": f"[View]({actions_url})"})

    body = [
        {
            "type": "TextBlock",
            "text": f"{subject_prefix}: {status}",
            "weight": "Bolder",
            "size": "Medium",
            "color": color,
        },
        {"type": "FactSet", "facts": facts},
        {
            "type": "TextBlock",
            "text": report,
            "wrap": True,
            "fontType": "Monospace",
            "size": "Small",
        },
    ]

    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": body,
                },
            }
        ],
    }

    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            log.info("Teams webhook sent (HTTP %d)", resp.status)
    except Exception as e:
        log.warning("Failed to send Teams webhook: %s", e)
