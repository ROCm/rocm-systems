#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
###############################################################################
"""CPU-side duration and ALU/memory hint for LLVM-lowered AMDGPU kernels."""

from kernel_sched_hint.devices import Device, device, known_devices
from kernel_sched_hint.hint import predict_metadata, predict_text
from kernel_sched_hint.metadata import Argument, KernelMetadata
from kernel_sched_hint.model import Prediction

__all__ = [
    "Argument",
    "Device",
    "KernelMetadata",
    "Prediction",
    "device",
    "known_devices",
    "predict_metadata",
    "predict_text",
]
