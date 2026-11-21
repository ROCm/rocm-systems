# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests: no hardware, no driver.

Runs on import, so it holds under unit_tests.py, ``python -m unittest`` and pytest alike.
"""

from common.common import forbid_real_amdsmi_init

forbid_real_amdsmi_init()
