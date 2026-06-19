# Contributing: new LLM provider

## What you're adding

A backend adapter to a new LLM provider (OpenAI-compatible, Ollama, custom,
etc.). Providers implement the Agents SDK `Model` protocol and handle auth,
request/response formatting, and error recovery.

## File locations

- Implementation: `perfxpert/providers/<name>_model.py`
- Smoke test: `tests/test_providers/test_<name>_smoke.py`
- Registry: `perfxpert/providers/__init__.py` (list of available providers)

## Template

```python
# SKIP-SAMPLE — template: <name>/<Name>/<ENV_VAR> are placeholders
"""<name> — <description>.

Implements the OpenAI Agents SDK Model protocol.
"""

import os
from typing import Optional

from openai_agents.models import Model


class NameModel(Model):
    """LLM provider adapter for <name>."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize.

        Args:
            api_key: optional override; if None, read from <ENV_VAR>
        """
        self.api_key = api_key or os.getenv("<ENV_VAR>")
        if not self.api_key:
            raise ValueError(f"<ENV_VAR> not set")

    async def agenerate(self, messages: list, **kwargs) -> str:
        """Call the model and return the response text."""
        # Implementation
        pass

    async def agenerate_with_tools(self, messages: list, tools: list, **kwargs) -> tuple:
        """Call the model with tool definitions; return (response, tool_calls)."""
        # Implementation
        pass

    def sanitize_output(self, text: str) -> str:
        """Strip unwanted prefixes/suffixes before tool invocation."""
        return text.strip()
```

## Security requirements

- **Secrets:** read from environment variables only (e.g., `<PROVIDER>_API_KEY`)
- Never log API keys, tokens, or request bodies containing secrets
- Sanitize LLM output before tool calls (strip control chars, injection payloads)
- No network calls outside the provider's official SDK

## Schema constraints (CI-enforced)

- Implements `Model` protocol (type-check passes)
- No new runtime dependencies (or vendored + approved)
- Secrets never appear in logs or test fixtures

## Tests you must add

Write `tests/test_providers/test_<name>_smoke.py`:

- `test_<name>_initializes()` — constructor succeeds
- `test_<name>_agenerate_succeeds()` — happy-path call (mocked API)
- `test_<name>_raises_on_missing_secret()` — error handling
- `test_<name>_sanitizes_output()` — injection prevention
- Nightly smoke test (registered in `.github/workflows/perfxpert-nightly.yml`)

## Review requirements

- 1 security-focused reviewer
- Secrets handling audit
- CI green (protocol compliance + smoke tests)

## Common pitfalls

- Don't hardcode API keys or defaults; always read from env
- Don't assume the model exists or is available at test time (mock or skip)
- Error messages must not leak authentication details
- If the provider is rate-limited, add backoff logic

## Related docs

- OpenAI Agents SDK Model protocol: https://github.com/openai/agents-sdk
- Nightly smoke workflow: `.github/workflows/perfxpert-nightly.yml`
- Existing providers in `perfxpert/providers/` as references
