"""CI guardrail: AGENT_BUILDERS is the complete agent inventory.

Every other guardrail test (tool allowlist, schema field caps, fence size)
enumerates AGENT_BUILDERS. If an agent is missing from it, those tests pass
while silently skipping that agent — which is how the Trace-Diff specialist
went unchecked. These tests discover agent builders from the package itself
and assert the inventory matches.
"""

import importlib
import pkgutil

import perfxpert.agents as agents_pkg
from perfxpert.agents import AGENT_BUILDERS
from perfxpert.agents.framework import Agent


def _discover_agent_builders():
    """Every ``build_*`` in the agents package that returns an Agent."""
    discovered = {}
    for mod_info in pkgutil.iter_modules(agents_pkg.__path__):
        module = importlib.import_module(f"perfxpert.agents.{mod_info.name}")
        for attr in dir(module):
            if not attr.startswith("build_"):
                continue
            fn = getattr(module, attr)
            if not callable(fn):
                continue
            try:
                result = fn()
            except Exception:
                # Builders needing arguments are not plain agent factories.
                continue
            if isinstance(result, Agent):
                discovered[attr] = fn
    return discovered


def test_inventory_contains_every_discovered_builder():
    discovered = _discover_agent_builders()
    registered = {fn.__name__ for fn in AGENT_BUILDERS}
    missing = set(discovered) - registered
    assert not missing, (
        f"agent builder(s) missing from AGENT_BUILDERS: {sorted(missing)}. "
        f"Unregistered agents are skipped by every CI guardrail test."
    )


def test_inventory_has_no_stale_entries():
    discovered = _discover_agent_builders()
    registered = {fn.__name__ for fn in AGENT_BUILDERS}
    stale = registered - set(discovered)
    assert not stale, f"AGENT_BUILDERS references non-existent builder(s): {sorted(stale)}"


def test_inventory_is_exported():
    exported = set(agents_pkg.__all__)
    for fn in AGENT_BUILDERS:
        assert fn.__name__ in exported, (
            f"{fn.__name__} is in AGENT_BUILDERS but not in perfxpert.agents.__all__"
        )


def test_inventory_matches_documented_agent_count():
    """The package docstring advertises the hierarchy size; keep it honest."""
    assert len(AGENT_BUILDERS) == 8, (
        f"AGENT_BUILDERS has {len(AGENT_BUILDERS)} agents; update the "
        f"perfxpert.agents docstring if the hierarchy changed"
    )


def test_agent_names_are_unique():
    names = [b().name for b in AGENT_BUILDERS]
    assert len(names) == len(set(names)), f"duplicate agent names: {names}"
