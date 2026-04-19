# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared counter definitions for HW performance counter parsing.

This module is the single source of truth for three closely related concerns
that used to be hand-copied between :mod:`rocprof_compute_soc.soc_base` and
``tools/validate_sets_metric_ids.py``:

1. The set of hardware block names referenced by any arch's
   ``perfmon_config`` in ``src/utils/mi_gpu_spec.yaml`` — ``BLOCK_NAMES`` is
   derived from the YAML, so adding a new arch or block updates the regex
   automatically.

2. The aggregator-suffix grammar (``_sum``, ``_avr``, ``_max``, ``_min``) and
   the TCC channel-template marker produced when a formula references
   ``TCC_HIT[0]``/``TCC_HIT[1]``/... — downstream code in
   :func:`rocprof_compute_soc.soc_base.OmniSoC_Base._expand_tcc_template_counters`
   expands the template into per-channel counters.

3. The ``SQC``/``SP`` → ``SQ`` remap used by
   :meth:`rocprof_compute_soc.soc_base.CounterFile.add` when packing counters
   into PMC perfmon files.

Importing this module reads ``mi_gpu_spec.yaml`` once (cached). No logging.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    # For type-checking, treat yaml as the standard PyYAML shape (resolved
    # against types-pyyaml). At runtime, the vendored copy wins when the
    # profile-mode stdlib guard is active; otherwise we fall back to the
    # system yaml. Both code paths expose the same :func:`safe_load` surface.
    import yaml
else:  # pragma: no cover - resolved once at import time
    try:
        from vendored import yaml
    except ImportError:
        import yaml

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_GPU_SPEC_PATH: Final[Path] = _PROJECT_ROOT / "src" / "utils" / "mi_gpu_spec.yaml"

# ---------------------------------------------------------------------------
# Aggregator suffixes, synthetic counters, block remap
# ---------------------------------------------------------------------------

# Aggregator suffixes that can trail a counter name inside a formula.
AGGREGATOR_SUFFIXES: Final[tuple[str, ...]] = ("_sum", "_avr", "_max", "_min")

# Counters whose block name is keyed to another block for perfmon packing.
# ``SQC`` and ``SP`` counters share the ``SQ`` budget; see
# :meth:`rocprof_compute_soc.soc_base.CounterFile.add`.
BLOCK_REMAP: Final[dict[str, str]] = {"SQC": "SQ", "SP": "SQ"}

# ``SQ_ACCUM_PREV_HIRES`` is a synthetic counter injected separately for
# level counters (see :meth:`OmniSoC_Base.perfmon_filter`); not a real PMC
# counter and must be filtered out of extractor results.
SYNTHETIC_COUNTERS: Final[frozenset[str]] = frozenset({"SQ_ACCUM_PREV_HIRES"})


# ---------------------------------------------------------------------------
# Built-in variable and denominator definitions
# ---------------------------------------------------------------------------

SUPPORTED_DENOM: Final[dict[str, str]] = {
    "per_wave": "SQ_WAVES",
    "per_cycle": "$GRBM_GUI_ACTIVE_PER_XCD",
    "per_second": "((End_Timestamp - Start_Timestamp) / 1000000000)",
    "per_kernel": "1",
}

BUILD_IN_VARS: Final[dict[str, str]] = {
    "GRBM_GUI_ACTIVE_PER_XCD": "(GRBM_GUI_ACTIVE / $num_xcd)",
    "GRBM_COUNT_PER_XCD": "(GRBM_COUNT / $num_xcd)",
    "GRBM_SPI_BUSY_PER_XCD": "(GRBM_SPI_BUSY / $num_xcd)",
    "numActiveCUs": (
        "TO_INT(MIN(ROUND(SUM(4 * SQ_BUSY_CU_CYCLES) / "
        "SUM($GRBM_GUI_ACTIVE_PER_XCD), 0) / $max_waves_per_cu * 8 + "
        "MIN(MOD(ROUND(SUM(4 * SQ_BUSY_CU_CYCLES) / "
        "SUM($GRBM_GUI_ACTIVE_PER_XCD), 0), $max_waves_per_cu), 8), $cu_per_gpu))"
    ),
    "kernelBusyCycles": (
        "ROUND(AVG((((End_Timestamp - Start_Timestamp) / 1000) * $max_sclk)), 0)"
    ),
    "hbmBandwidth": "($max_mclk / 1000 * 32 * $num_hbm_channels)",
}


# ---------------------------------------------------------------------------
# YAML traversal helpers (shared between BLOCK_NAMES discovery and
# perfmon-config loading so the schema shape is documented once)
# ---------------------------------------------------------------------------


def _iter_perfmon_entries(
    path: Path,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(arch_name, perfmon_dict)`` pairs from *path*.

    ``yaml.safe_load`` returns ``Any``, so every level of the traversal is
    guarded with ``isinstance`` before it is used. A missing file yields
    nothing (callers that patch :data:`_GPU_SPEC_PATH` can start clean).
    """
    if not path.is_file():
        return
    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        return
    data = cast("dict[str, Any]", raw)
    series_list = data.get("mi_gpu_spec") or []
    if not isinstance(series_list, list):
        return
    for series in series_list:
        if not isinstance(series, dict):
            continue
        arch_list = series.get("gpu_archs") or []
        if not isinstance(arch_list, list):
            continue
        for arch_entry in arch_list:
            if not isinstance(arch_entry, dict):
                continue
            arch = arch_entry.get("gpu_arch")
            perfmon = arch_entry.get("perfmon_config") or {}
            if not isinstance(arch, str) or not isinstance(perfmon, dict):
                continue
            yield arch, cast("dict[str, Any]", perfmon)


@lru_cache(maxsize=1)
def _load_block_names(path: Path = _GPU_SPEC_PATH) -> frozenset[str]:
    """Union of counter-block names referenced by any arch's ``perfmon_config``.

    Non-block metadata entries (e.g. ``TCC_channels``) are filtered out: a
    valid block name has no underscore in it *and* its value is an integer
    counter cap. Returns an empty frozenset if the spec file is missing so
    tests that patch the path can start from a clean state.
    """
    blocks: set[str] = set()
    for _arch, perfmon in _iter_perfmon_entries(path):
        for key, value in perfmon.items():
            if isinstance(key, str) and isinstance(value, int) and "_" not in key:
                blocks.add(key)
    return frozenset(blocks)


def block_names() -> frozenset[str]:
    """Return the cached hardware block name set (see :func:`_load_block_names`)."""
    return _load_block_names()


def load_perfmon_configs(
    path: Path = _GPU_SPEC_PATH,
    metadata_keys: frozenset[str] = frozenset(),
) -> dict[str, dict[str, int]]:
    """Return ``{gpu_arch: {block: max_counters}}`` from *path*.

    Keys listed in *metadata_keys* are dropped (they appear alongside real
    block caps — e.g. ``TCC_channels`` — but are not per-block budgets).
    Non-integer values are ignored so partial/corrupt files degrade
    gracefully.
    """
    result: dict[str, dict[str, int]] = {}
    for arch, perfmon in _iter_perfmon_entries(path):
        result[arch] = {
            k: v
            for k, v in perfmon.items()
            if isinstance(k, str) and isinstance(v, int) and k not in metadata_keys
        }
    return result


# Module-level snapshot for callers that don't care about cache semantics.
BLOCK_NAMES: Final[frozenset[str]] = _load_block_names()


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _counter_regex(blocks: frozenset[str] = BLOCK_NAMES) -> re.Pattern[str]:
    """Build a counter-matching regex from *blocks*.

    Driven by the block set from ``mi_gpu_spec.yaml`` so adding a new block
    to any arch automatically extends the matcher. The block set is unioned
    with :data:`BLOCK_REMAP` keys so that aliased prefixes such as ``SQC_*``
    and ``SP_*`` — which share the ``SQ`` perfmon budget and do *not* appear
    as standalone entries in ``perfmon_config`` — still match. Alternatives
    are sorted longest-first so ``SQC`` wins over ``SQ``, ``GL1A`` over
    ``GL1``, etc.
    """
    prefixes = set(blocks) | set(BLOCK_REMAP)
    sorted_blocks = sorted(prefixes, key=len, reverse=True)
    if not sorted_blocks:
        # Nothing to match — produce a regex that matches nothing rather than
        # an empty alternation (which would be a syntax error).
        return re.compile(r"(?!)")
    block_alt = "|".join(re.escape(b) for b in sorted_blocks)
    suffix = r"(?:\[|_sum|_avr|_max|_min)*"
    return re.compile(rf"(?:{block_alt})_[0-9A-Z_]*[0-9A-Z]{suffix}")


_VARIABLE_RE: Final[re.Pattern[str]] = re.compile(r"\$([0-9A-Za-z_]*[0-9A-Za-z])")

# ---------------------------------------------------------------------------
# Legacy-compatible public constants. The string-literal regex pair is kept
# for callers that want a pre-built pattern without the block set as an
# argument. ``HW_COUNTER_RE`` is the data-driven pattern (includes ``GDS``
# and every other block in ``perfmon_config``).
# ---------------------------------------------------------------------------

HW_COUNTER_RE: Final[re.Pattern[str]] = _counter_regex()
VARIABLE_RE: Final[re.Pattern[str]] = _VARIABLE_RE


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------


def extract_variable_names(text: str) -> set[str]:
    """Return the variable names (without the leading ``$``) in *text*."""
    return set(_VARIABLE_RE.findall(text))


def extract_counter_tokens(
    text: str,
    blocks: frozenset[str] | None = None,
) -> set[str]:
    """Return every HW counter token in *text*.

    - TCC channel references in formulas (``TCC_HIT[0]``, ``TCC_HIT[1]``, ...)
      collapse into a single template token ending with ``[`` — downstream
      code (``_expand_tcc_template_counters``) detects this marker and
      expands the template to per-channel counters.
    - Aggregator-suffixed counters (``FOO_sum``, ``FOO_avr``, ...) are kept
      with the suffix attached.
    - ``$variables`` are excluded (fetched separately via
      :func:`extract_variable_names`).
    """
    block_set = BLOCK_NAMES if blocks is None else blocks
    pattern = _counter_regex(block_set)
    hw = set(pattern.findall(text))
    # The counter regex will match inside a $variable reference
    # (e.g. "$GRBM_GUI_ACTIVE_PER_XCD" yields a spurious counter token).
    # Strip variable bodies out so only real counters survive.
    return hw - extract_variable_names(text)


def strip_suffix(token: str) -> str:
    """Remove one trailing aggregator suffix from *token* if present."""
    for suffix in AGGREGATOR_SUFFIXES:
        if token.endswith(suffix):
            return token[: -len(suffix)]
    return token


_CHANNEL_INDEX_RE: Final[re.Pattern[str]] = re.compile(r"\[\d*\]?$")


def strip_channel(token: str) -> str:
    """Drop a trailing TCC channel index (``[0]``/``[15]``) or template ``[``.

    Handles both the fully expanded form (``TCC_HIT[3]``) and the
    template-marker form (``TCC_HIT[``) produced by
    :func:`extract_counter_tokens`.
    """
    return _CHANNEL_INDEX_RE.sub("", token)


def remap_block(counter: str) -> str:
    """Return the effective perfmon block for *counter*.

    ``SQC_*`` and ``SP_*`` counters are packed into the ``SQ`` budget; every
    other counter keeps its first-segment block.
    """
    block = counter.split("_", 1)[0]
    return BLOCK_REMAP.get(block, block)


# ---------------------------------------------------------------------------
# Public API (legacy-compatible names kept for existing callers)
# ---------------------------------------------------------------------------


def parse_counters_text(text: str) -> tuple[set[str], set[str]]:
    """Extract HW counter names and ``$variable`` names from formula *text*.

    Returns ``(hw_counters, variables)`` where *variables* are the names
    without the leading ``$``. Matches that look like variables are removed
    from the HW counter set.
    """
    return extract_counter_tokens(text), extract_variable_names(text)


def resolve_variables(
    text: str,
    build_in_vars: dict[str, str],
    extra_texts: list[str] | None = None,
    blocks: frozenset[str] | None = None,
) -> set[str]:
    """Extract HW counters from *text*, recursively expanding ``$`` variables.

    1. Pull counters + variable names from *text* and each string in
       *extra_texts* (typically the ``SUPPORTED_DENOM`` formulas).
    2. For every variable found, if it is defined in *build_in_vars*, parse
       its body and fold its counters/variables back in. Repeat until no new
       variables appear.

    Synthetic counters (``SQ_ACCUM_PREV_HIRES``) are filtered out.
    """
    block_set = BLOCK_NAMES if blocks is None else blocks
    hw: set[str] = set()
    variables: set[str] = set()

    for chunk in (text, *(extra_texts or [])):
        hw |= extract_counter_tokens(chunk, block_set)
        variables |= extract_variable_names(chunk)

    seen: set[str] = set()
    while variables - seen:
        for var in list(variables - seen):
            seen.add(var)
            body = build_in_vars.get(var)
            if body is None:
                continue
            hw |= extract_counter_tokens(body, block_set)
            variables |= extract_variable_names(body)

    return hw - SYNTHETIC_COUNTERS


def extract_counters(text: str) -> set[str]:
    """Return the full set of HW counters referenced by *text*.

    Resolves ``$variable`` references and supported denominators recursively
    so that all transitive counter dependencies are included. Synthetic
    counters (``SQ_ACCUM_PREV_HIRES``) are filtered out.
    """
    return resolve_variables(
        text,
        BUILD_IN_VARS,
        extra_texts=list(SUPPORTED_DENOM.values()),
    )


def counter_to_block(counter: str) -> str:
    """Map a counter name to its IP block, applying :data:`BLOCK_REMAP`."""
    return remap_block(counter)
