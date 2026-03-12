# LLM Persistent Conversation Design

**Date:** 2026-03-12
**Status:** Approved

---

## Goal

Replace the per-call `LLMAnalyzer` instantiation pattern in `InteractiveSession` with a single persistent multi-turn conversation that lives for the duration of the session. The LLM fence is set once as the system prompt; all subsequent turns are user/assistant message pairs. Streaming eliminates request-level timeouts. LLM-based compaction keeps the context window bounded. Full message history is archived to disk.

**Out of scope:**
- `LLMAnalyzer` — unchanged; continues to serve the standalone `analyze_with_llm()` API path
- File rewrite calls — both `_llm_rewrite_file` (WorkflowSession) and `_apply_suggestions_via_llm` (InteractiveSession) remain standalone `LLMAnalyzer` calls (see §File Rewrites)
- Anthropic extended thinking — not supported by `LLMConversation`; remains available only via `LLMAnalyzer`

---

## Architecture

One new file: `ai_analysis/llm_conversation.py` — the `LLMConversation` class.

Changes to `llm_analyzer.py`: expose `load_reference_guide() -> str` as a module-level function.

Changes to `interactive.py`:
- **`InteractiveSession`** — owns `_conv: LLMConversation`; LLM call sites in `_optimize_via_tier0`, `_request_optimization_suggestions`, and `_llm_annotate_profiling_plan` route through it; `SessionContext` and `_format_context_block` / `_update_ctx_*` machinery removed; `SessionData.context` replaced by `SessionData.conversation`; `_conv` persisted in `SessionData`
- **`WorkflowSession`** — does **not** own `_conv`; its only LLM call is `_llm_rewrite_file` which remains a standalone `LLMAnalyzer` call (see §File Rewrites)

---

## Data Model

### `LLMConversation`

```python
class LLMConversation:
    def __init__(
        self,
        provider: str,                        # "anthropic" | "openai" | "local"
        api_key: Optional[str] = None,        # read from env if None
        model: Optional[str] = None,          # provider default if None
        compact_every: int = 10,              # LLM-summarize after every N assistant turns
        keep_recent_turns: int = 6,           # full turns kept after compaction
        history_path: Optional[Path] = None,  # JSONL archive; None = no disk archive
    )
```

**Internal state:**
- `_system: str` — fence + role; set once by `initialize()`; sent on every API call
- `_messages: List[Dict]` — active context window (`[{"role": "user"|"assistant", "content": str}]`)
- `_turn_count: int` — total assistant turns completed; drives compaction trigger. **Compaction API calls do NOT increment `_turn_count`.**
- `_history_path: Optional[Path]` — `None` means no disk archive; compaction still runs but old messages are discarded rather than written

**Public interface:**

```python
def initialize(self, system_prompt: str) -> None:
    """Set system prompt once at session start (fence + role). Must be called before send()."""

def send(
    self,
    user_message: str,
    *,
    max_tokens: int = 4096,
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """Append user turn, stream response, return full response text.
    Callers must sanitize user_message before calling (no auto-sanitization).
    Checks compaction trigger after each assistant turn."""

def to_dict(self) -> Dict[str, Any]:
    """Serialize for SessionData persistence.
    Stores: system, messages, turn_count, compact_every, keep_recent_turns, history_path."""

@classmethod
def from_dict(cls, d: Dict[str, Any], **kwargs) -> "LLMConversation":
    """Restore from serialized state. _system is restored from d['system'] as-is.
    Note: if the fence file changed since the session was saved, the stored system
    prompt is used (no automatic detection). Users should start a fresh session
    to pick up fence changes."""

@property
def turn_count(self) -> int: ...

@property
def messages(self) -> List[Dict]: ...
```

**Data sanitization:** `send()` accepts raw strings. Callers are responsible for sanitizing before calling. This preserves existing behaviour: the same `_sanitize_data()` / `_sanitize_source_data()` calls made today before passing data to `LLMAnalyzer` continue to be made before calling `_conv.send()`.

### Modified: `SessionData` (InteractiveSession only)

Remove:
```python
context: Optional[Dict[str, Any]] = None  # was serialized SessionContext
```

Add:
```python
conversation: Optional[Dict[str, Any]] = None  # serialized LLMConversation
```

- `to_dict()` — include `conversation` key
- `from_dict()` — reconstruct from `d.get("conversation")`; `None` if absent (backward-compatible)

---

## Streaming

`send()` always uses streaming. Streaming eliminates request-level timeouts: the connection stays alive as long as tokens arrive.

```python
# Anthropic
with client.messages.stream(
    model=model, max_tokens=max_tokens,
    system=self._system,
    messages=self._messages,
) as stream:
    for text in stream.text_stream:
        if on_token:
            on_token(text)
        result += text

# OpenAI and local (OpenAI-compatible endpoint)
# Try max_completion_tokens first (gpt-5, o1, o3, gpt-4o-2024-11-20+);
# fall back to max_tokens on BadRequestError (older models).
try:
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": self._system}] + self._messages,
        max_completion_tokens=max_tokens,
        stream=True,
    )
except openai.BadRequestError as e:
    if "max_completion_tokens" in str(e):
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": self._system}] + self._messages,
            max_tokens=max_tokens,
            stream=True,
        )
    else:
        raise
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        if on_token:
            on_token(delta)
        result += delta
```

**Local provider** uses the same OpenAI streaming path with:
- `base_url`: `ROCPD_LLM_LOCAL_URL` env var (default `http://localhost:11434/v1`)
- `model`: `ROCPD_LLM_LOCAL_MODEL` env var (default `codellama:13b`)
- `api_key`: `"ignored"`

**`on_token` callback:**

A module-level helper in `interactive.py`:
```python
def _print_token(t: str) -> None:
    print(t, end="", flush=True)
```

| Call site | `on_token` |
|---|---|
| `_optimize_via_tier0` | `_print_token` — live stream to terminal |
| `_request_optimization_suggestions` | `_print_token` — live stream to terminal |
| `_llm_annotate_profiling_plan` | `_print_token` — live stream to terminal |
| Post-rewrite summary turn | `None` — silent, no display needed |
| Compaction summary call | `None` — internal, not shown |
| File rewrites | N/A — handled by standalone `LLMAnalyzer`, not `_conv` |

---

## Compaction

**Trigger:** `_turn_count % compact_every == 0` checked at the end of `send()`, after `_turn_count` is incremented.

**What gets compacted:** all messages except the most recent `keep_recent_turns * 2` entries (default: keep last 12 messages = 6 full turns).

**Compaction call** — non-streaming, separate provider call. Does NOT increment `_turn_count`. Uses:
- system: `self._system`
- messages: `old_messages + [{"role": "user", "content": compaction_prompt}]`
- max_tokens: 600

```python
compaction_prompt = (
    "Summarize the key context from this session so far. Include:\n"
    "- What app is being profiled and its source files\n"
    "- Profiling runs done (trace types, counter sets collected)\n"
    "- Performance issues identified (bottlenecks, percentages)\n"
    "- Code optimizations applied and their observed effect\n"
    "- Current state of the application\n"
    "Be concise (max 300 words)."
)
```

**Result:** old messages replaced by a two-message summary block at position 0:
```python
{"role": "user",      "content": "Summarize our session so far."}
{"role": "assistant", "content": f"[Session summary] {summary}"}
```
Followed by the `keep_recent_turns * 2` recent messages.

**Archived messages:** if `_history_path` is not `None`, old messages are appended to the JSONL archive before being dropped. If `None`, they are discarded.

**Compaction failure:** log warning, skip this cycle, retry at next trigger. Session never crashes.

**User-configurable:** `compact_every` via `--llm-compact-every N` CLI flag; registered in `analyze.py:add_args()`, passed through to session constructors.

---

## Disk Archive

**Path:** explicitly passed by `InteractiveSession` as `~/.rocpd/sessions/<session_id>_history.jsonl`. `WorkflowSession` passes `history_path=None` (no archive).

**Format:**
```json
{"role": "user", "content": "...", "turn": 3, "ts": "2026-03-12T02:15:00"}
{"role": "assistant", "content": "...", "turn": 3, "ts": "2026-03-12T02:15:01"}
```

**Write pattern:** append-only; messages written when compacted out of `_messages`.

**Read pattern:** never loaded back into active context. Exists for human audit and future tooling.

---

## File Rewrites

Both `_llm_rewrite_file` (WorkflowSession, line ~2179) and `_apply_suggestions_via_llm` (InteractiveSession, line ~1245) are **kept as standalone `LLMAnalyzer` calls** — not routed through `_conv`. Reasons:

1. File rewrites require a specialized system prompt ("Return ONLY the complete rewritten file — no markdown") incompatible with the fence-as-system-prompt model.
2. The full rewritten file bloats the conversation without adding useful context for subsequent turns.

**Post-rewrite summary:** after a successful rewrite in `InteractiveSession._apply_suggestions_via_llm`, a brief turn is appended to `_conv` so the LLM is aware of what changed. `WorkflowSession._llm_rewrite_file` does not do this (no `_conv`):
```python
# InteractiveSession._apply_suggestions_via_llm only
self._conv.send(
    f"File `{file_path.name}` was rewritten applying the above optimizations. "
    f"Compilation: {'successful' if compiled else 'pending'}.",
    on_token=None,
)
```

---

## Session Integration

### System prompt construction

`llm_analyzer.py` exposes:
```python
def load_reference_guide() -> str:
    """Load the LLM fence document. Same path lookup order as LLMAnalyzer._load_reference_guide():
    ROCPD_LLM_REFERENCE_GUIDE env var → module share/ dir → /opt/rocm/share/..."""
```

`InteractiveSession` calls it once at init:
```python
fence = load_reference_guide()
self._conv.initialize(
    "You are an expert AMD GPU performance engineer "
    "helping optimize a HIP/ROCm application.\n\n" + fence
)
```

### LLM call site mapping

Specialized format requirements previously encoded in per-call system prompts are moved into the user message. Routing all calls through `_conv` means the fence is always present as system context — this is intentional and improves response quality even for calls (like `annotate_profiling_plan`) that previously used a short focused system prompt.

| Old call | New `_conv.send()` user message |
|---|---|
| `analyzer.annotate_profiling_plan(metadata)` | `"Annotate this profiling plan (max 200 words, plain text only — no markdown): {json.dumps(metadata)}"` |
| `analyzer.suggest_optimizations(summaries, custom_prompt)` | `"{custom_prompt}\n\nAnalyze these AMD GPU source files and provide concrete, actionable optimization suggestions. Focus on: memory coalescing, wave occupancy, unnecessary hipDeviceSynchronize, blocking hipMemcpy, MFMA usage, LDS utilization, loop structure, kernel launch parameters. Be specific — reference actual patterns visible in the code. Use plain text only — no markdown headers. Start each file section with exactly: FILE: <filename>\n\n{formatted_summaries}"` |
| `analyzer._call_*/send()` in `_optimize_via_tier0` | `"Based on these detected GPU source patterns, provide concrete optimization recommendations (max 300 words, plain text only — no markdown headers):\n{json.dumps(metadata)}"` |

### Exception handling at call sites

Existing `try/except` wrappers at each call site are unchanged. `LLMAuthenticationError` and `LLMRateLimitError` raised by `send()` propagate through to the existing handlers.

### What is removed from `interactive.py`

- `SessionContext` dataclass
- `_format_context_block()`
- `_update_ctx_analysis()`
- `_update_ctx_suggestion()`
- `_update_ctx_command()`
- Per-call `LLMAnalyzer` instantiation at `_optimize_via_tier0`, `_request_optimization_suggestions`, `_llm_annotate_profiling_plan`

### Persistence pattern (InteractiveSession only)

**Save** (both `[s]` branch and `_save_and_quit`):
```python
if self._conv:
    self._session.conversation = self._conv.to_dict()
self._store.save(self._session)
```

**Restore** (in `_init_session`):
```python
raw_conv = loaded.conversation
history_path = Path(f"~/.rocpd/sessions/{self._session.session_id}_history.jsonl").expanduser()
if raw_conv:
    self._conv = LLMConversation.from_dict(
        raw_conv, api_key=self._llm_api_key, model=self._llm_model
    )
else:
    self._conv = LLMConversation(
        provider=self._llm_provider, api_key=self._llm_api_key,
        model=self._llm_model, compact_every=self._compact_every,
        history_path=history_path,
    )
    self._conv.initialize(
        "You are an expert AMD GPU performance engineer "
        "helping optimize a HIP/ROCm application.\n\n" + load_reference_guide()
    )
```

**WorkflowSession:** does not own `_conv` — no session-level LLM calls to route through a conversation.

---

## Error Handling

- Streaming errors mid-response: catch exception, return whatever was accumulated. Print warning. Session continues.
- Compaction failure: log warning, skip this cycle, retry next trigger.
- `LLMAuthenticationError` / `LLMRateLimitError`: propagate; existing call-site handlers catch them.
- `_conv is None`: all call sites guard with `if self._conv is None` and fall back to local analysis.

---

## Token Budget

Active context per turn (approximate):
- System prompt (fence): ~3 000 tokens
- Compaction summary: ~400 tokens
- `keep_recent_turns=6` recent turns: ~6 000 tokens
- **Total per call: ~9 400 tokens input** — well within 200 K token context windows

---

## Testing

New file: `ai_analysis/tests/test_llm_conversation.py`

- `TestLLMConversation`: mock provider; `_messages` grows correctly; `_system` set once; `send()` appends user + assistant turns; `_turn_count` increments
- `TestStreaming`: `on_token` called per chunk; silent collection when `None`; all three providers (anthropic, openai, local)
- `TestOpenAIFallback`: `max_completion_tokens` → `max_tokens` on `BadRequestError`
- `TestCompaction`: trigger at turn N; `_turn_count` NOT incremented by compaction call; old messages flushed to JSONL archive; summary block at position 0; recent turns preserved
- `TestDiskArchive`: JSONL append-only; `history_path=None` → no file written
- `TestPersistence`: `to_dict()` / `from_dict()` round-trip; `_messages`, `_turn_count`, `_system` restored correctly
- Integration: mock `LLMConversation` in `InteractiveSession`; verify `_conv.send()` called for `_llm_annotate_profiling_plan`, `_optimize_via_tier0`, `_request_optimization_suggestions`; verify `SessionContext` no longer referenced in `InteractiveSession`; verify post-rewrite summary turn appended to `_conv`; verify `WorkflowSession` has no `_conv` attribute

Existing `ai_analysis/tests/test_interactive_context.py` (27 tests) — **removed**.

---

## Files Changed

| File | Change |
|---|---|
| `ai_analysis/llm_conversation.py` | **NEW** — `LLMConversation` class |
| `ai_analysis/llm_analyzer.py` | Add module-level `load_reference_guide()` |
| `ai_analysis/interactive.py` | `InteractiveSession` owns `_conv`; `WorkflowSession` unchanged (no `_conv`); remove `SessionContext` + `_format_context_block` + `_update_ctx_*` from `InteractiveSession`; update `SessionData` (`conversation` replaces `context`); streaming display; post-rewrite summary turns; `_print_token` helper |
| `ai_analysis/__init__.py` | Export `LLMConversation` |
| `ai_analysis/tests/test_llm_conversation.py` | **NEW** — unit + integration tests |
| `ai_analysis/tests/test_interactive_context.py` | **REMOVED** |
| `analyze.py` | Register `--llm-compact-every N` in `add_args()`; pass to session constructors |
