"""Fallback-provider chain — transparent retry across providers.

Wraps a *chain* of providers so that a ``RateLimitError`` (and optionally
other taxonomy errors) from the primary provider cascades to the next
provider in the chain instead of either retrying or surfacing to the
caller. Intended for interactive sessions where the user has multiple
providers configured and just wants an answer.

Env var ``PERFXPERT_LLM_FALLBACK_CHAIN`` is a comma-separated list of
registered provider names, e.g. ``openai,anthropic``. The first name
is tried first; each subsequent name is tried only on
``RateLimitError`` from the previous one.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import (
    DryRunResponse as _DryRunResponseT,
    ProviderChainExhausted,
    RateLimitError,
    UnknownProvider,
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
        """Resolve a chain entry to a live Provider.

        Raises ``UnknownProvider`` (never a bare ``KeyError``) so callers
        and the cascade loop see only provider-taxonomy exceptions.
        """
        if isinstance(entry, Provider):
            return entry
        try:
            return get_provider(entry)
        except KeyError as exc:
            # Pull the known-list out of the message for richer telemetry.
            # registry.get_provider formats: "Unknown provider 'x'; known: a, b"
            known: List[str] = []
            msg = str(exc)
            if "known:" in msg:
                known_part = msg.rsplit("known:", 1)[1].strip().rstrip("'\"")
                if known_part and known_part != "<none registered>":
                    known = [k.strip() for k in known_part.split(",") if k.strip()]
            raise UnknownProvider(entry, known=known) from exc

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        dry_run: bool = False,
    ) -> Union[ProviderResponse, _DryRunResponseT]:
        # Defensive airgap guard — a session built wrong (e.g. airgap=True
        # but someone still passed this FallbackProvider to complete())
        # MUST NOT make network calls. build_session() already prevents
        # FallbackProvider construction in airgap mode, but if a caller
        # invokes complete() directly with PERFXPERT_AIRGAP=1 we refuse.
        # dry_run short-circuits BEFORE this guard so deterministic
        # cost-estimation paths still work inside airgap.
        if not dry_run and os.environ.get("PERFXPERT_AIRGAP", "0") == "1":
            raise RuntimeError(
                "FallbackProvider.complete() called while PERFXPERT_AIRGAP=1 "
                "(non-dry-run). Airgap mode forbids LLM calls; build the "
                "session with airgap=True and use dry_run=True or skip LLM "
                "invocation entirely."
            )
        attempts: List[Tuple[str, BaseException]] = []
        chain_names: List[str] = [
            entry if isinstance(entry, str) else type(entry).__name__
            for entry in self._providers
        ]
        for idx, entry in enumerate(self._providers):
            entry_name = entry if isinstance(entry, str) else type(entry).__name__
            try:
                provider = self._resolve(entry)
            except UnknownProvider as e:
                # Unknown provider name — record and move on.
                _LOG.warning(
                    "fallback: provider entry %r unresolvable: %s", entry, e
                )
                attempts.append((entry_name, e))
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
                attempts.append((entry_name, e))
                continue
        # Chain exhausted — surface a typed error. Never leak KeyError.
        last_exc: Optional[BaseException] = attempts[-1][1] if attempts else None
        err = ProviderChainExhausted(providers=chain_names, attempts=attempts)
        if last_exc is not None:
            raise err from last_exc
        raise err


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
