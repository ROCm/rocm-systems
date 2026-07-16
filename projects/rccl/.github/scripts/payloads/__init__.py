"""Workload payload plugins. Register new workloads in REGISTRY."""

from .base import Payload
from .coverage import CoveragePayload

REGISTRY = {
    "coverage": CoveragePayload,
}

__all__ = ["Payload", "CoveragePayload", "REGISTRY"]
