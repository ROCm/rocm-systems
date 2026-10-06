# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""DME <-> AMDSMI integration CI helpers.

These helpers replace the inline bash logic that previously lived in
``.github/workflows/amdsmi-dme-ci.yml``. Each module exposes a small,
testable surface so developers can reproduce CI steps locally:

    python3 -m dme_integration <subcommand> [options]

See ``__main__.py`` for the available subcommands.
"""
