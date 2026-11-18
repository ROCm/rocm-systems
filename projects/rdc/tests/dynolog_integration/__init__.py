# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""dynolog <-> RDC integration CI helpers.

The **dynolog** system-level telemetry daemon drives RDC through a thin C++
shim, ``RdcWrapper`` (``dynolog/src/gpumon/amd``), which embeds RDC in
process (no separate ``rdcd``). These helpers build that shim and its example
against a freshly-built RDC so CI catches RDC changes that would break the
downstream dynolog integration. Each phase is a subcommand::

    python3 -m dynolog_integration <subcommand> [options]

See ``__main__.py`` for the available subcommands.
"""
