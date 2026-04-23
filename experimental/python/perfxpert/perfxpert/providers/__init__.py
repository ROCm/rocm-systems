"""LLM provider abstraction layer.

All providers implement Provider.complete() with identical signature and
error taxonomy. dry_run=True always returns the DryRunResponse singleton
with zero network I/O.

Public API:
    get_provider(name, **kwargs) -> Provider
    list_providers() -> Dict[str, str]
    Provider, ProviderResponse         (base types, Task 2)
    DryRunResponse                     (singleton, Task 1)
    ProviderError, AuthError, RateLimitError, QuotaExceededError,
    TransientError, FatalError, TimeoutError, ProviderChainExhausted
                                    (taxonomy, Task 1)

Env var conventions (canonical):
    PERFXPERT_LLM_ANTHROPIC_KEY
    PERFXPERT_LLM_OPENAI_KEY
    PERFXPERT_LLM_LOCAL_URL         (ollama)
    PERFXPERT_LLM_PRIVATE_URL / _MODEL / _API_KEY / _HEADERS / _VERIFY_SSL
    PERFXPERT_OPENCODE_PATH
    PERFXPERT_IN_OPENCODE_SESSION   (recursion guard marker)

Back-compat env-var aliases (honored with DeprecationWarning):
    ROCPD_LLM_*  → PERFXPERT_LLM_*   (pre-rename alias)
"""

from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import (
    AuthError,
    DryRunResponse,
    FatalError,
    ProviderChainExhausted,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    TimeoutError,
    TransientError,
)
from perfxpert.providers.registry import get_provider, list_providers, register

__all__ = [
    "AuthError",
    "DryRunResponse",
    "FatalError",
    "ProviderChainExhausted",
    "ProviderError",
    "Provider",
    "ProviderResponse",
    "QuotaExceededError",
    "RateLimitError",
    "TimeoutError",
    "TransientError",
    "get_provider",
    "list_providers",
    "register",
]
