"""Fallback-provider chain — transparent retry across providers.

Wraps a *chain* of providers so that a ``RateLimitError`` (and optionally
other taxonomy errors) from the primary provider cascades to the next
provider in the chain instead of either retrying or surfacing to the
caller. Intended for interactive sessions where the user has multiple
providers configured and just wants an answer.

Env var ``PERFXPERT_LLM_FALLBACK_CHAIN`` is a comma-separated list of
registered provider names, e.g. ``openai,anthropic,claude-code``. The
first name is tried first; each subsequent name is tried only on
``RateLimitError`` from the previous one.

See docs/superpowers/plans/2026-04-18-perfxpert-phase8-pr2-user-issues.md.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union

from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import (
    DryRunResponse as _DryRunResponseT,
    RateLimitError,
)
from perfxpert.providers.registry import get_provider

__all__ = [
    "FallbackProvider",
    "get_fallback_provider",
    "parse_chain_env",
    "ENV_FALLBACK_CHAIN",
]

ENV_FALLBACK_CHAIN = "PERFXPERT_LLM_FALLBACK_CHAIN"

_LOG = logging.getLogger("perfxpert.providers.fallback")


def parse_chain_env(value: Optional[str]) -> List[str]:
    """Parse the comma-separated fallback chain env var.

    Empty / None -> []. Whitespace and blank segments are trimmed.
    """
    if not value:
        return []
    return [seg.strip() for seg in value.split(",") if seg.strip()]


class FallbackProvider(Provider):
    """Cascades across providers on ``RateLimitError``.

    The constructor accepts either:

      - a list of already-instantiated Provider instances, OR
      - a list of provider-name strings that will be resolved lazily via
        ``get_provider(name)`` on first use.

    Non-RateLimitError exceptions propagate from the primary (the caller
    likely wants to see auth / timeout / validation errors).
    """

    def __init__(
        self,
        providers: Sequence[Union[Provider, str]],
        *,
        stop_on_auth_error: bool = True,
    ) -> None:
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        self._providers: List[Union[Provider, str]] = list(providers)
        self._stop_on_auth_error = stop_on_auth_error

    def _resolve(self, entry: Union[Provider, str]) -> Provider:
        if isinstance(entry, Provider):
            return entry
        return get_provider(entry)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        dry_run: bool = False,
    ) -> Union[ProviderResponse, _DryRunResponseT]:
        last_error: Optional[Exception] = None
        for idx, entry in enumerate(self._providers):
            try:
                provider = self._resolve(entry)
            except KeyError as e:
                # Unknown provider name — skip but remember.
                _LOG.warning("fallback: provider entry %r unresolvable: %s", entry, e)
                last_error = e
                continue

            try:
                return provider.complete(
                    messages,
                    system=system,
                    model=model,
                    max_tokens=max_tokens,
                    dry_run=dry_run,
                )
            except RateLimitError as e:
                _LOG.warning(
                    "fallback: provider %d/%d (%r) rate-limited; trying next",
                    idx + 1,
                    len(self._providers),
                    entry,
                )
                last_error = e
                continue
        assert last_error is not None  # loop must have run at least once
        raise last_error


def get_fallback_provider(
    chain: Optional[Sequence[str]] = None,
) -> Optional[FallbackProvider]:
    """Build a FallbackProvider from the env chain (or explicit arg).

    Returns None when no chain is configured — callers should then fall
    back to their usual single-provider ``get_provider()`` path.
    """
    if chain is None:
        chain = parse_chain_env(os.environ.get(ENV_FALLBACK_CHAIN))
    if not chain:
        return None
    return FallbackProvider(list(chain))
