# ai_analysis/llm_conversation.py
"""Persistent multi-turn LLM conversation with streaming, compaction, and disk archive."""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .llm_analyzer import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL
from .exceptions import LLMAuthenticationError, LLMRateLimitError

_DEFAULT_LOCAL_URL = "http://localhost:11434/v1"
_DEFAULT_LOCAL_MODEL = "codellama:13b"


def _build_private_client(api_key: Optional[str], model_override: Optional[str]):
    """Build an OpenAI client for a private/enterprise LLM server.

    Reads configuration from environment variables:
        ROCINSIGHT_LLM_PRIVATE_URL        Base URL of the private server (required)
        ROCINSIGHT_LLM_PRIVATE_MODEL      Model name to use
        ROCINSIGHT_LLM_PRIVATE_API_KEY    API key (default: "dummy" for header-auth servers)
        ROCINSIGHT_LLM_PRIVATE_HEADERS    JSON object of extra request headers, e.g.
                                     '{"Ocp-Apim-Subscription-Key": "abc123"}'
                                     The "user" header is auto-set to os.getlogin()
                                     unless already present in ROCINSIGHT_LLM_PRIVATE_HEADERS.
        ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL Set to "0" or "false" to disable SSL certificate
                                     verification (e.g. for corporate proxies with
                                     self-signed certs). Requires httpx package.
    """
    try:
        import openai as _openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    base_url = os.environ.get("ROCINSIGHT_LLM_PRIVATE_URL", "")
    if not base_url:
        raise ValueError(
            "ROCINSIGHT_LLM_PRIVATE_URL is not set. "
            "Export it to point at your private LLM server, e.g.:\n"
            "  export ROCINSIGHT_LLM_PRIVATE_URL=https://my-apim.example.com/openai/deployments/gpt4"
        )
    key = api_key or os.environ.get("ROCINSIGHT_LLM_PRIVATE_API_KEY", "dummy")
    model = model_override or os.environ.get("ROCINSIGHT_LLM_PRIVATE_MODEL", "")

    # Build headers: start with user auto-header, then overlay env-var headers
    headers: Dict[str, str] = {}
    try:
        headers["user"] = os.getlogin()
    except OSError:
        pass  # getlogin() can fail in some CI/container environments
    raw_headers = os.environ.get("ROCINSIGHT_LLM_PRIVATE_HEADERS", "")
    if raw_headers:
        # Try strict JSON first; only normalize single-quotes as a fallback.
        # The replace-based normalization is intentionally not the first path
        # because it would corrupt values containing legitimate apostrophes
        # (e.g. Bearer tokens with embedded apostrophes).
        parsed_headers = None
        try:
            parsed_headers = json.loads(raw_headers)
        except json.JSONDecodeError:
            try:
                parsed_headers = json.loads(raw_headers.replace("'", '"'))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"ROCINSIGHT_LLM_PRIVATE_HEADERS is not valid JSON: {e}\n"
                    f'Use double-quoted JSON: \'{{"Ocp-Apim-Subscription-Key": "abc123"}}\'\n'
                    f"Value was: {raw_headers!r}"
                )
        if not isinstance(parsed_headers, dict):
            raise ValueError(
                f"ROCINSIGHT_LLM_PRIVATE_HEADERS must be a JSON object, got "
                f"{type(parsed_headers).__name__}.\n"
                f'Expected format: \'{{"Ocp-Apim-Subscription-Key": "abc123"}}\'\n'
                f"Value was: {raw_headers!r}"
            )
        headers.update(parsed_headers)

    # SSL verification — disabled when ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL=0/false
    verify_ssl_env = os.environ.get("ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL", "1").lower()
    verify_ssl = verify_ssl_env not in ("0", "false", "no")
    http_client = None
    if not verify_ssl:
        try:
            import httpx as _httpx

            http_client = _httpx.Client(verify=False)
        except ImportError:
            warnings.warn(
                "[LLMConversation] ROCINSIGHT_LLM_PRIVATE_VERIFY_SSL=0 requested but httpx is "
                "not installed. SSL verification will remain enabled. "
                "Run: pip install httpx",
                stacklevel=3,
            )

    kwargs: Dict[str, Any] = dict(api_key=key, base_url=base_url, default_headers=headers)
    if http_client is not None:
        kwargs["http_client"] = http_client

    client = _openai.OpenAI(**kwargs)
    return client, model, http_client


class LLMConversation:
    """Persistent multi-turn LLM session with streaming and LLM-based compaction.

    Usage:
        conv = LLMConversation(provider="anthropic", api_key="sk-ant-...")
        conv.initialize("You are an expert AMD GPU engineer.\\n\\n" + fence)
        response = conv.send("What is the bottleneck?", on_token=print_fn)
    """

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        compact_every: int = 10,
        keep_recent_turns: int = 6,
        history_path: Optional[Path] = None,
    ) -> None:
        valid = {"anthropic", "openai", "local", "private", "claude-code"}
        if provider not in valid:
            raise ValueError(
                f"Unknown provider: {provider!r}. Must be one of: {', '.join(sorted(valid))}"
            )
        self._provider = provider
        self._api_key = api_key
        self._model = model
        self._compact_every = max(1, compact_every)
        self._keep_recent_turns = max(0, keep_recent_turns)
        self._history_path = Path(history_path) if history_path else None
        self._system: str = ""
        self._messages: List[Dict[str, str]] = []
        self._turn_count: int = 0

    def initialize(self, system_prompt: str) -> None:
        """Set the system prompt (fence + role). Must be called before send()."""
        self._system = system_prompt

    def send(
        self,
        user_message: str,
        *,
        max_tokens: int = 4096,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Append user turn, stream response, increment turn_count, check compaction."""
        if not self._system:
            raise RuntimeError(
                "LLMConversation.send() called before initialize(). "
                "Call initialize(system_prompt) first."
            )
        self._messages.append({"role": "user", "content": user_message})
        result = self._stream_response(max_tokens=max_tokens, on_token=on_token)
        self._messages.append({"role": "assistant", "content": result})
        self._turn_count += 1
        if self._turn_count % self._compact_every == 0:
            self._compact()
        return result

    # ── Streaming ─────────────────────────────────────────────────────────────

    def _stream_response(
        self,
        max_tokens: int,
        on_token: Optional[Callable[[str], None]],
    ) -> str:
        if self._provider == "anthropic":
            return self._stream_anthropic(max_tokens=max_tokens, on_token=on_token)
        if self._provider == "claude-code":
            return self._stream_claude_code(max_tokens=max_tokens, on_token=on_token)
        return self._stream_openai(max_tokens=max_tokens, on_token=on_token)

    def _stream_anthropic(
        self,
        max_tokens: int,
        on_token: Optional[Callable[[str], None]],
    ) -> str:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMAuthenticationError(
                "No Anthropic API key. Set ANTHROPIC_API_KEY environment variable."
            )
        model = (
            self._model or os.environ.get("ROCINSIGHT_LLM_MODEL") or DEFAULT_ANTHROPIC_MODEL
        )
        client = _anthropic.Anthropic(api_key=api_key)
        chunks: List[str] = []
        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=self._system,
                messages=self._messages,
                timeout=120,
            ) as stream:
                for text in stream.text_stream:
                    if on_token:
                        on_token(text)
                    chunks.append(text)
        except _anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(f"Anthropic authentication failed: {e}")
        except _anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {e}")
        except (LLMAuthenticationError, LLMRateLimitError):
            raise
        except Exception as e:
            if chunks:
                warnings.warn(
                    f"[LLMConversation] Streaming error mid-response: {e}", stacklevel=3
                )
            else:
                raise
        return "".join(chunks)

    # Mapping from Claude Code CLI aliases → full Anthropic API model IDs (fallback path).
    _CLAUDE_CODE_ALIAS_MAP: Dict[str, str] = {
        "sonnet": "claude-sonnet-4-6",
        "opus": "claude-opus-4-6",
        "haiku": "claude-haiku-4-5",
    }

    def _stream_claude_code(
        self,
        max_tokens: int,
        on_token: Optional[Callable[[str], None]],
    ) -> str:
        """Two-tier call for the ``claude-code`` provider.

        Auth priority:
        1. ``ANTHROPIC_API_KEY`` via the ``anthropic`` SDK — direct API, no CLI.
        2. ``claude -p`` subprocess — stored Claude Code CLI OAuth credentials
           (activated only when no API key is available).

        The full conversation history is inlined into the user message so each
        invocation is stateless. Streaming is emulated: ``on_token`` is called
        once with the complete response text.
        """
        model = self._model or os.environ.get("ROCINSIGHT_LLM_MODEL") or "sonnet"

        # Build a transcript of prior turns to include as context
        history_lines: List[str] = []
        for msg in self._messages[:-1]:  # all but the last (current) user message
            role = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"[{role}]: {msg['content']}")

        current_user_msg = self._messages[-1]["content"] if self._messages else ""
        if history_lines:
            user_prompt = (
                "Previous conversation:\n"
                + "\n\n".join(history_lines)
                + f"\n\n[User]: {current_user_msg}"
            )
        else:
            user_prompt = current_user_msg

        result = self._call_claude_code_turn(user_prompt, model, max_tokens=max_tokens)
        if on_token and result:
            on_token(result)
        return result

    def _call_claude_code_turn(self, user_prompt: str, model: str, *, max_tokens: int = 4096) -> str:
        """Single-turn call for _stream_claude_code and _call_non_streaming.

        Priority:
        1. ``ANTHROPIC_API_KEY`` via the ``anthropic`` SDK — direct API, no CLI.
        2. ``claude -p`` subprocess — stored Claude Code CLI OAuth credentials.
        """
        # ── Tier 1: ANTHROPIC_API_KEY via anthropic SDK ───────────────────────
        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic as _anthropic
                api_model = self._CLAUDE_CODE_ALIAS_MAP.get(model, model)
                client = _anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=api_model,
                    max_tokens=max_tokens,
                    system=self._system,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return next(
                    (b.text for b in response.content if b.type == "text"), ""
                ).strip()
            except ImportError:
                pass  # anthropic package not installed — try CLI
            except Exception as _exc:
                # Re-raise auth/rate-limit errors; for anything else fall through to CLI
                if isinstance(_exc, (LLMAuthenticationError, LLMRateLimitError)):
                    raise
                try:
                    import anthropic as _a
                    if isinstance(_exc, _a.AuthenticationError):
                        raise LLMAuthenticationError("ANTHROPIC_API_KEY is invalid or expired.")
                    if isinstance(_exc, _a.RateLimitError):
                        raise LLMRateLimitError("Anthropic API rate limit reached.")
                except ImportError:
                    pass
                raise

        # ── Tier 2: claude -p subprocess (stored CLI credentials) ────────────
        try:
            result = self._call_claude_cli_subprocess(user_prompt, model)
            if result:
                return result
        except Exception as _cli_err:
            raise LLMAuthenticationError(
                f"Claude Code: no ANTHROPIC_API_KEY and CLI also failed ({_cli_err}). "
                "Set ANTHROPIC_API_KEY or ensure 'claude' CLI is authenticated."
            ) from _cli_err

        raise LLMAuthenticationError(
            "Claude Code: ANTHROPIC_API_KEY not set and 'claude' CLI returned empty. "
            "Set ANTHROPIC_API_KEY or ensure 'claude' CLI is working."
        )

    def _call_claude_cli_subprocess(self, user_prompt: str, model: str) -> str:
        """Call ``claude -p`` subprocess using Claude Code's stored credentials.

        Tier-2 fallback: used when no ``ANTHROPIC_API_KEY`` is available.  The
        system prompt is written as ``CLAUDE.md`` in a temporary directory so the
        CLI picks it up as project context (avoids argument-length limits and
        mirrors Claude Code's natural project-instruction mechanism).
        Raises on failure so the caller can propagate the error.
        """
        import subprocess as _subprocess
        import json as _json
        import tempfile as _tempfile
        import pathlib as _pathlib

        with _tempfile.TemporaryDirectory(prefix="rocinsight_claude_") as _tmpdir:
            (_pathlib.Path(_tmpdir) / "CLAUDE.md").write_text(
                self._system or "", encoding="utf-8"
            )

            cmd = [
                "claude",
                "-p", user_prompt,
                "--output-format", "json",
                "--no-session-persistence",
                "--model", model,
            ]
            try:
                proc = _subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=_tmpdir,
                )
            except FileNotFoundError:
                raise RuntimeError("claude CLI not found on PATH")
            except _subprocess.TimeoutExpired:
                raise RuntimeError("claude CLI timed out after 300 s")

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            raise RuntimeError(
                f"claude CLI exited with code {proc.returncode}"
                + (f": {detail}" if detail else "")
            )

        try:
            data = _json.loads(proc.stdout)
        except _json.JSONDecodeError:
            return proc.stdout.strip()

        if data.get("is_error"):
            raise RuntimeError(data.get("result") or "claude CLI reported an error")

        return (data.get("result") or "").strip()

    def _stream_openai(
        self,
        max_tokens: int,
        on_token: Optional[Callable[[str], None]],
    ) -> str:
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        _http_client = None
        if self._provider == "local":
            base_url = os.environ.get("ROCINSIGHT_LLM_LOCAL_URL", _DEFAULT_LOCAL_URL)
            model = self._model or os.environ.get(
                "ROCINSIGHT_LLM_LOCAL_MODEL", _DEFAULT_LOCAL_MODEL
            )
            client = _openai.OpenAI(api_key="ignored", base_url=base_url)
        elif self._provider == "private":
            client, model, _http_client = _build_private_client(
                self._api_key, self._model
            )
            if not model:
                if _http_client is not None:
                    _http_client.close()
                raise ValueError(
                    "No model specified for private provider. "
                    "Set ROCINSIGHT_LLM_PRIVATE_MODEL or pass --llm-private-model."
                )
        else:
            api_key = self._api_key or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise LLMAuthenticationError(
                    "No OpenAI API key. Set OPENAI_API_KEY environment variable."
                )
            model = (
                self._model or os.environ.get("ROCINSIGHT_LLM_MODEL") or DEFAULT_OPENAI_MODEL
            )
            client = _openai.OpenAI(api_key=api_key)
            _http_client = None

        messages_with_system = [
            {"role": "system", "content": self._system}
        ] + self._messages
        chunks: List[str] = []
        try:
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages_with_system,
                    max_completion_tokens=max_tokens,
                    stream=True,
                    timeout=120,
                )
            except _openai.BadRequestError as e:
                if "max_completion_tokens" in str(e):
                    stream = client.chat.completions.create(
                        model=model,
                        messages=messages_with_system,
                        max_tokens=max_tokens,
                        stream=True,
                        timeout=120,
                    )
                else:
                    raise
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    if on_token:
                        on_token(delta)
                    chunks.append(delta)
        except _openai.AuthenticationError as e:
            raise LLMAuthenticationError(f"OpenAI authentication failed: {e}")
        except _openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}")
        except (LLMAuthenticationError, LLMRateLimitError):
            raise
        except Exception as e:
            if chunks:
                warnings.warn(
                    f"[LLMConversation] Streaming error mid-response: {e}", stacklevel=3
                )
            else:
                raise
        finally:
            if _http_client is not None:
                _http_client.close()
        return "".join(chunks)

    # ── Compaction ────────────────────────────────────────────────────────────

    _COMPACTION_PROMPT = (
        "Summarize the key context from this session so far. Include:\n"
        "- What app is being profiled and its source files\n"
        "- Profiling runs done (trace types, counter sets collected)\n"
        "- Performance issues identified (bottlenecks, percentages)\n"
        "- Code optimizations applied and their observed effect\n"
        "- Current state of the application\n"
        "Be concise (max 300 words)."
    )

    def _compact(self) -> None:
        """LLM-summarize oldest messages; replace with summary block + recent turns."""
        keep = self._keep_recent_turns * 2
        if len(self._messages) <= keep:
            return
        old_messages = self._messages[:-keep] if keep > 0 else list(self._messages)
        recent_messages = self._messages[-keep:] if keep > 0 else []

        if self._history_path and old_messages:
            self._append_to_archive(old_messages)

        try:
            summary = self._call_non_streaming(
                messages=old_messages
                + [{"role": "user", "content": self._COMPACTION_PROMPT}],
                max_tokens=600,
            )
            summary_block = [
                {"role": "user", "content": "Summarize our session so far."},
                {"role": "assistant", "content": f"[Session summary] {summary}"},
            ]
            self._messages = summary_block + recent_messages
        except Exception as e:
            warnings.warn(
                f"[LLMConversation] Compaction failed, skipping: {e}", stacklevel=2
            )

    def _call_non_streaming(self, messages: List[Dict], max_tokens: int) -> str:
        """Non-streaming API call used for compaction. Does NOT increment _turn_count."""
        if self._provider == "claude-code":
            # Inline the messages as a single user prompt so the stateless
            # _call_claude_code_turn helper can handle both tiers.
            lines: List[str] = []
            for m in messages:
                role = "User" if m["role"] == "user" else "Assistant"
                lines.append(f"[{role}]: {m['content']}")
            prompt = "\n\n".join(lines)
            model = self._model or os.environ.get("ROCINSIGHT_LLM_MODEL") or "sonnet"
            return self._call_claude_code_turn(prompt, model, max_tokens=max_tokens)

        if self._provider == "anthropic":
            try:
                import anthropic as _anthropic
            except ImportError:
                raise ImportError("anthropic package not installed.")
            api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            model = (
                self._model
                or os.environ.get("ROCINSIGHT_LLM_MODEL")
                or DEFAULT_ANTHROPIC_MODEL
            )
            client = _anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self._system,
                messages=messages,
                timeout=120,
            )
            return resp.content[0].text if resp.content else ""

        # openai or local
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("openai package not installed.")
        _http_client = None
        if self._provider == "local":
            client = _openai.OpenAI(
                api_key="ignored",
                base_url=os.environ.get("ROCINSIGHT_LLM_LOCAL_URL", _DEFAULT_LOCAL_URL),
            )
            model = self._model or os.environ.get(
                "ROCINSIGHT_LLM_LOCAL_MODEL", _DEFAULT_LOCAL_MODEL
            )
        elif self._provider == "private":
            client, model, _http_client = _build_private_client(
                self._api_key, self._model
            )
            if not model:
                if _http_client is not None:
                    _http_client.close()
                raise ValueError(
                    "No model specified for private provider. "
                    "Set ROCINSIGHT_LLM_PRIVATE_MODEL or pass --llm-private-model."
                )
        else:
            client = _openai.OpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY", "")
            )
            model = (
                self._model or os.environ.get("ROCINSIGHT_LLM_MODEL") or DEFAULT_OPENAI_MODEL
            )
        full_messages = [{"role": "system", "content": self._system}] + messages
        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=full_messages,
                    max_completion_tokens=max_tokens,
                    timeout=120,
                )
            except _openai.BadRequestError as e:
                if "max_completion_tokens" in str(e):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=full_messages,
                        max_tokens=max_tokens,
                        timeout=120,
                    )
                else:
                    raise
            return resp.choices[0].message.content or ""
        finally:
            if _http_client is not None:
                _http_client.close()

    # ── Disk archive ──────────────────────────────────────────────────────────

    def _append_to_archive(self, messages: List[Dict]) -> None:
        """Append messages to JSONL archive (append-only)."""
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).isoformat()
            with self._history_path.open("a", encoding="utf-8") as f:
                for msg in messages:
                    entry = {
                        "role": msg["role"],
                        "content": msg["content"],
                        "turn": self._turn_count,
                        "ts": ts,
                    }
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            warnings.warn(
                f"[LLMConversation] Failed to write history archive: {e}", stacklevel=3
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for SessionData persistence. Does NOT include api_key."""
        return {
            "provider": self._provider,
            "model": self._model,
            "compact_every": self._compact_every,
            "keep_recent_turns": self._keep_recent_turns,
            "history_path": str(self._history_path) if self._history_path else None,
            "system": self._system,
            "messages": list(self._messages),
            "turn_count": self._turn_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **kwargs: Any) -> "LLMConversation":
        """Restore from serialized state.

        kwargs:
            api_key: override (api_key is never stored in to_dict())
            model:   override stored model
        """
        history_path = d.get("history_path")
        conv = cls(
            provider=d["provider"],
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model") or d.get("model"),
            compact_every=d.get("compact_every", 10),
            keep_recent_turns=d.get("keep_recent_turns", 6),
            history_path=Path(history_path) if history_path else None,
        )
        conv._system = d.get("system", "")
        conv._messages = list(d.get("messages", []))
        conv._turn_count = d.get("turn_count", 0)
        return conv

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def messages(self) -> List[Dict]:
        return list(self._messages)
