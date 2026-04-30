# Contributing: new LLM provider

## What you're adding

A backend adapter to a new LLM provider (OpenAI-compatible, Ollama,
custom, etc.). Providers implement PerfXpert's
`perfxpert.providers.Provider` interface and return a normalized
`ProviderResponse`. The live agent runtime still routes hosted
providers through the OpenAI Agents SDK facade in
`perfxpert/agents/framework.py`, so adding a new user-facing provider
usually requires both a provider module and runtime/CLI registration.

## File locations

- Implementation: `perfxpert/providers/<name>_provider.py`
- Unit tests: `tests/test_providers/test_<name>_provider.py`
- Registry: call `register("<name>", ProviderClass, "...")` from the
  provider module.
- Agent runtime: update `perfxpert/agents/framework.py` if the provider
  needs a new model prefix, default model, or `RunConfig` provider.
- CLI: update the `--llm` choices in `perfxpert/analyze.py`, the
  first-run provider detection in `perfxpert/cli/init_cmd.py`, and
  provider status rendering if the new provider needs env-var checks.

## Template

```python
# SKIP-SAMPLE — template: <name>/<Name>/<ENV_VAR> are placeholders
"""<name> — <description>.

Implements the PerfXpert Provider protocol.
"""

import os
from typing import Any, Dict, List, Optional, Union

from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import AuthError, DryRunResponse, ProviderError
from perfxpert.providers.registry import register


class NameProvider(Provider):
    """LLM provider adapter for <name>."""

    def __init__(self, api_key: Optional[str] = None, **_: Any) -> None:
        self.api_key = api_key or os.environ.get("<ENV_VAR>")
        if not self.api_key:
            raise AuthError("<name>", "no API key (set <ENV_VAR>)")

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        dry_run: bool = False,
    ) -> Union[ProviderResponse, object]:
        if dry_run:
            return DryRunResponse

        # Call the provider's chat/completions-style API here.
        try:
            content = "..."
        except Exception as exc:
            raise ProviderError(f"[<name>] {exc}") from exc

        return ProviderResponse(
            content=content,
            provider="<name>",
            model=model or "<default-model>",
            input_tokens=0,
            output_tokens=0,
        )


register("<name>", NameProvider, "<description for providers list>")
```

## Security requirements

- **Secrets:** read from environment variables only (e.g.,
  `PERFXPERT_LLM_<PROVIDER>_KEY` or the provider's canonical API key)
- Never log API keys, tokens, or request bodies containing secrets
- Normalize provider errors into `AuthError`, `RateLimitError`,
  `QuotaExceededError`, `TransientError`, `FatalError`, `TimeoutError`,
  or `ProviderError`
- No network calls outside the provider module or the runtime facade

## Schema constraints (CI-enforced)

- Implements `Provider.complete()` with the shared signature
- No new runtime dependencies (or vendored + approved)
- Secrets never appear in logs or test fixtures
- `dry_run=True` returns `DryRunResponse` without network I/O

## Tests you must add

Write `tests/test_providers/test_<name>_provider.py`:

- `test_<name>_initializes()` — constructor succeeds
- `test_<name>_complete_succeeds()` — happy-path call (mocked API)
- `test_<name>_raises_on_missing_secret()` — error handling
- `test_<name>_dry_run_has_no_network()` — dry-run invariant
- Provider status / registry test if the provider appears in
  `perfxpert providers list`

## Review requirements

- 1 security-focused reviewer
- Secrets handling audit
- CI green (protocol compliance + smoke tests)

## Common pitfalls

- Don't hardcode API keys or defaults; always read from env
- Don't assume the model exists or is available at test time (mock or skip)
- Error messages must not leak authentication details
- If the provider is rate-limited, raise `RateLimitError` so the
  fallback chain can try the next provider

## Related docs

- [Agentic mode guide](../../guides/agentic-mode.md) — provider selection,
  fallback chain, and air-gap behavior
- Existing providers in `perfxpert/providers/` as references
