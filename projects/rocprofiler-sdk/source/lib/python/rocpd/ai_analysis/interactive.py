"""Interactive session for rocpd analyze --interactive."""
from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


# ── Session data ─────────────────────────────────────────────────────────────

@dataclass
class PersistentMenuItem:
    """A recommendation promoted to the main menu from a previous analysis."""
    id: str
    title: str
    priority: str               # "HIGH" | "MEDIUM" | "LOW"
    source: str                 # "profiling_analysis" | "code_change_analysis"
    added_at: str               # ISO-8601
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoryEntry:
    type: str                   # "profiling_run" | "code_change"
    timestamp: str
    db_path: str = ""
    files_modified: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class SessionData:
    session_id: str
    source_dir: str
    created_at: str
    last_updated: str
    history: List[HistoryEntry] = field(default_factory=list)
    persistent_menu_items: List[PersistentMenuItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionData":
        history = [HistoryEntry(**h) for h in d.get("history", [])]
        items = [PersistentMenuItem(**m) for m in d.get("persistent_menu_items", [])]
        return cls(
            session_id=d["session_id"],
            source_dir=d["source_dir"],
            created_at=d["created_at"],
            last_updated=d["last_updated"],
            history=history,
            persistent_menu_items=items,
        )


# ── SessionStore ──────────────────────────────────────────────────────────────

_DEFAULT_SESSIONS_DIR = pathlib.Path.home() / ".rocpd" / "sessions"


class SessionStore:
    """Handles session file I/O under sessions_dir."""

    def __init__(self, sessions_dir: Optional[Union[str, pathlib.Path]] = None) -> None:
        self._dir = pathlib.Path(sessions_dir) if sessions_dir else _DEFAULT_SESSIONS_DIR

    def _path_for(self, session_id: str) -> pathlib.Path:
        return self._dir / f"{session_id}.json"

    def save(self, data: SessionData) -> pathlib.Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        p = self._path_for(data.session_id)
        p.write_text(json.dumps(data.to_dict(), indent=2))
        return p

    def load(self, id_or_path: str) -> Optional[SessionData]:
        """Load by session ID or by absolute/relative file path."""
        try:
            candidate = pathlib.Path(id_or_path)
            if candidate.exists():
                raw = json.loads(candidate.read_text())
                return SessionData.from_dict(raw)
            p = self._path_for(id_or_path)
            if p.exists():
                raw = json.loads(p.read_text())
                return SessionData.from_dict(raw)
            return None
        except Exception as exc:
            warnings.warn(f"Failed to load session {id_or_path!r}: {exc}", stacklevel=2)
            return None

    def find_by_source_dir(self, source_dir: str) -> List[SessionData]:
        """Return all sessions whose source_dir matches, newest first."""
        if not self._dir.exists():
            return []
        results: List[SessionData] = []
        for f in self._dir.glob("*.json"):
            try:
                raw = json.loads(f.read_text())
                if raw.get("source_dir") == source_dir:
                    results.append(SessionData.from_dict(raw))
            except Exception:
                pass
        def _safe_dt(s):
            try:
                return datetime.fromisoformat(s.created_at)
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)
        return sorted(results, key=_safe_dt, reverse=True)

    @staticmethod
    def make_session_id(source_dir: str) -> str:
        slug = pathlib.Path(source_dir).name.replace(" ", "_")[:24] or "session"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        return f"{ts}_{slug}"


# ── Rendering helpers ─────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    _RICH = True
except ImportError:
    _RICH = False

_console = Console() if _RICH else None

_PRI_STYLE = {"HIGH": "bold red", "MEDIUM": "yellow", "LOW": "green", "INFO": "blue"}


def _print(msg: str = "", style: str = "") -> None:
    if _RICH and _console:
        _console.print(msg, style=style or None)
    else:
        print(msg)


def _input(prompt: str) -> str:
    return input(prompt)


# ── InteractiveSession ────────────────────────────────────────────────────────

class InteractiveSession:
    """Top-level interactive menu for rocpd analyze --interactive."""

    def __init__(
        self,
        source_dir: str,
        tier0_result: Optional[Any],
        recommendations: List[Dict[str, Any]],
        database_path: str,
        llm_provider: Optional[str],
        llm_api_key: Optional[str],
        llm_model: Optional[str],
        llm_local: Optional[str] = None,
        llm_local_model: Optional[str] = None,
        session_store: Optional[SessionStore] = None,
        resume_session_id: Optional[str] = None,
    ) -> None:
        self._source_dir = source_dir
        self._tier0 = tier0_result
        self._recs = recommendations or []
        self._db_path = database_path
        self._llm_provider = llm_provider
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model
        self._llm_local = llm_local
        self._llm_local_model = llm_local_model
        self._store = session_store or SessionStore()
        self._session = self._init_session(resume_session_id)

    @property
    def session(self) -> SessionData:
        return self._session

    def _init_session(self, resume_id: Optional[str]) -> SessionData:
        # Explicit resume
        if resume_id:
            loaded = self._store.load(resume_id)
            if loaded:
                return loaded

        # Auto-detect
        existing = self._store.find_by_source_dir(self._source_dir)
        if existing:
            chosen = self._prompt_resume(existing)
            if chosen:
                return chosen

        # New session
        now = datetime.now(timezone.utc).isoformat()
        return SessionData(
            session_id=SessionStore.make_session_id(self._source_dir),
            source_dir=self._source_dir,
            created_at=now,
            last_updated=now,
        )

    def _prompt_resume(self, existing: List[SessionData]) -> Optional[SessionData]:
        _print()
        _print(f"Found {len(existing)} previous session(s) for {self._source_dir}:",
               style="cyan")
        for i, s in enumerate(existing, 1):
            n_runs   = sum(1 for h in s.history if h.type == "profiling_run")
            n_change = sum(1 for h in s.history if h.type == "code_change")
            n_items  = len(s.persistent_menu_items)
            _print(f"  [{i}]  {s.session_id}  "
                   f"({n_runs} profiling run(s), {n_change} code change(s), "
                   f"{n_items} saved recommendation(s))")
        _print("  [n]  Start new session")
        _print()
        choice = _input("  > ").strip().lower()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(existing):
                return existing[idx]
            _print(f"  Invalid selection — starting new session.", style="dim")
        elif choice != "n":
            _print(f"  Unrecognized input — starting new session.", style="dim")
        return None

    def _render_main_menu(self) -> None:
        src_label = pathlib.Path(self._source_dir).name
        sid_label = self._session.session_id
        if _RICH and _console:
            _console.print(Panel(
                f"[bold]Source:[/bold] {src_label}   "
                f"[bold]Session:[/bold] {sid_label}   "
                f"[dim]\\[s] save  \\[q] quit[/dim]",
                title="[bold cyan]rocpd Interactive Analysis[/bold cyan]",
                border_style="blue",
            ))
        else:
            w = 70
            print("=" * w)
            print(f"  rocpd Interactive Analysis | {src_label}")
            print(f"  Session: {sid_label}  [s] save  [q] quit")
            print("=" * w)

        _print()
        _print("  [p]  Get profiling commands", style="white")
        _print("  [o]  Optimize source code  (AI — uploads source summary)",
               style="white")

        if self._session.persistent_menu_items:
            _print()
            _print("  ── From previous analysis " + "─" * 38, style="dim")
            for i, item in enumerate(self._session.persistent_menu_items, 1):
                pri_style = _PRI_STYLE.get(item.priority, "white")
                badge = f"[{item.priority}]"
                src_tag = " [code]" if item.source == "code_change_analysis" else ""
                if _RICH and _console:
                    _console.print(
                        f"  [cyan bold]\\[{i}][/cyan bold]  "
                        f"[cyan]✦[/cyan] {item.title}  "
                        f"[{pri_style}]{badge}[/{pri_style}]{src_tag}"
                    )
                else:
                    print(f"  [{i}]  ✦ {item.title}  {badge}{src_tag}")
        _print()

    def run(self) -> None:
        """Main event loop."""
        while True:
            self._render_main_menu()
            try:
                choice = _input("  > ").strip().lower()
            except EOFError:
                self._save_and_quit()
                break

            if choice == "q":
                self._save_and_quit()
                break
            elif choice == "s":
                self._store.save(self._session)
                _print("  Session saved.", style="green")
            elif choice == "p":
                self._path_profiling()
            elif choice == "o":
                self._path_optimize()
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(self._session.persistent_menu_items):
                    self._pursue_recommendation(
                        self._session.persistent_menu_items[idx]
                    )
            else:
                _print("  Unknown choice. Try p, o, 1–N, s, or q.", style="dim")

    def _save_and_quit(self) -> None:
        self._session.last_updated = datetime.now(timezone.utc).isoformat()
        self._store.save(self._session)
        _print("  Session saved. Goodbye.", style="cyan")

    def _ingest_recommendations(
        self, new_recs: List[Dict[str, Any]], source: str = "profiling_analysis"
    ) -> int:
        """Add unique recommendations to persistent_menu_items. Returns count added."""
        now = datetime.now(timezone.utc).isoformat()
        existing_ids = {m.id for m in self._session.persistent_menu_items}
        added = 0
        for rec in new_recs:
            rid = rec.get("id", rec.get("category", ""))
            if rid and rid not in existing_ids:
                self._session.persistent_menu_items.append(PersistentMenuItem(
                    id=rid,
                    title=rec.get("issue", rec.get("category", rid)),
                    priority=rec.get("priority", "INFO"),
                    source=source,
                    added_at=now,
                    detail=rec,
                ))
                existing_ids.add(rid)
                added += 1
        return added

    def _path_profiling(self, _source: str = "profiling_analysis") -> None:
        """Show profiling commands; optionally annotate with LLM; intake .db file."""
        _print()
        _print("  ── Profiling Commands ──────────────────────────────────", style="cyan")
        _print()

        cmds = self._collect_profiling_commands()

        # Optional LLM annotation on ProfilingPlan metadata (no source text)
        if self._llm_provider and self._tier0:
            cmds = self._llm_annotate_profiling_plan(cmds)

        if not cmds:
            _print("  (no profiling commands available)", style="dim")
        else:
            for i, (label, cmd) in enumerate(cmds, 1):
                _print(f"  [{i}]  {label}", style="white")
                _print(f"       $ {cmd}", style="dim")
                _print()

        _print("  Enter path to .db file when ready (or Enter to skip):", style="cyan")
        try:
            db_input = _input("  > ").strip()
        except EOFError:
            return

        if not db_input:
            return

        db_path = pathlib.Path(db_input).expanduser()
        if not db_path.exists():
            _print(f"  File not found: {db_path}", style="red")
            return

        _print("  Running Tier 1/2 analysis...", style="dim")
        new_recs = self._run_tier1_analysis(str(db_path))

        added = self._ingest_recommendations(new_recs, source=_source)
        now = datetime.now(timezone.utc).isoformat()
        self._session.history.append(HistoryEntry(
            type="profiling_run",
            timestamp=now,
            db_path=str(db_path),
        ))
        self._db_path = str(db_path)
        _print(f"  ✓ {added} recommendation(s) added to main menu.", style="green")

    def _collect_profiling_commands(self) -> List[tuple]:
        """Collect (label, full_command) pairs from tier0 and existing recommendations."""
        cmds: List[tuple] = []
        seen: set = set()

        def _add(label: str, cmd: str) -> None:
            if cmd and cmd not in seen:
                seen.add(cmd)
                cmds.append((label, cmd))

        if self._tier0:
            fc = getattr(self._tier0, "suggested_first_command", None)
            if fc:
                _add("Start Here — suggested first profiling command", fc)

        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
        for rec in sorted(self._recs,
                          key=lambda r: priority_order.get(r.get("priority", "INFO"), 4)):
            for cmd in rec.get("commands", []):
                fc = cmd.get("full_command", "")
                label = (f"[{rec.get('priority','INFO')}] {rec.get('category','')} — "
                         f"{cmd.get('tool','')}: {cmd.get('description','')}")
                _add(label, fc)

        return cmds

    def _llm_annotate_profiling_plan(self, cmds: List[tuple]) -> List[tuple]:
        """Send ProfilingPlan metadata (NOT source text) to online LLM for annotation."""
        try:
            from rocpd.ai_analysis.llm_analyzer import LLMAnalyzer
            plan = getattr(self._tier0, "profiling_plan", None)
            if plan is None:
                return cmds
            metadata = {
                "programming_model": getattr(plan, "programming_model", "HIP"),
                "kernel_count":      getattr(plan, "kernel_count", 0),
                "suggested_counters": getattr(plan, "suggested_counters", []),
                "risk_areas":         getattr(plan, "risk_areas", []),
                "detected_patterns":  [
                    {"id": p.pattern_id, "severity": p.severity,
                     "description": p.description}
                    for p in getattr(plan, "detected_patterns", [])
                ],
                "suggested_commands": [cmd for _, cmd in cmds],
            }
            model = self._llm_local_model if self._llm_provider == "local" else self._llm_model
            analyzer = LLMAnalyzer(
                provider=self._llm_provider,
                api_key=self._llm_api_key,
                model=model,
            )
            note = analyzer.annotate_profiling_plan(metadata)
            if note:
                _print()
                _print("  ── LLM Profiling Advice ────────────────────────────", style="cyan")
                _print(note)
                _print()
        except Exception as exc:
            _print(f"  (LLM annotation skipped: {exc})", style="dim")
        return cmds  # commands unchanged; LLM output is advisory only

    def _run_tier1_analysis(self, db_path: str) -> List[Dict[str, Any]]:
        """Run Tier 1/2 analysis on db_path; return recommendation list."""
        import io
        import contextlib
        try:
            from rocpd.rocpd import RocpdImportData
            from rocpd.analyze import analyze_performance
            result_store: Dict[str, Any] = {}
            _null = io.StringIO()
            with contextlib.redirect_stdout(_null):
                analyze_performance(
                    connection=RocpdImportData([db_path]),
                    database_path=db_path,
                    _collect_result=result_store,
                )
            return result_store.get("recommendations", [])
        except Exception as exc:
            _print(f"  (Tier 1 analysis failed: {exc})", style="red")
            return []

    _TOKEN_BUDGET = 60_000  # characters (approximate token proxy)

    def _select_hot_files(self, budget: int = _TOKEN_BUDGET) -> List[tuple]:
        """Return [(abs_path, content)] for files with detected kernels, within budget."""
        if not self._tier0:
            return []
        # Support both SourceAnalysisResult (detected_kernels directly on tier0)
        # and any future wrapper that exposes a .profiling_plan child object.
        plan = getattr(self._tier0, "profiling_plan", None) or self._tier0

        rel_paths: List[str] = []
        seen: set = set()
        for k in getattr(plan, "detected_kernels", []):
            # kernels may be dicts {"file": ...} or dataclass objects with .file
            rp = k.get("file", "") if isinstance(k, dict) else getattr(k, "file", "")
            if rp and rp not in seen:
                seen.add(rp)
                rel_paths.append(rp)

        result: List[tuple] = []
        used = 0
        base = pathlib.Path(self._source_dir)
        for rp in rel_paths:
            full = base / rp
            if not full.exists():
                continue
            content = full.read_text(errors="replace")
            if used + len(content) > budget:
                content = content[: budget - used]
                result.append((str(full), content))
                break
            result.append((str(full), content))
            used += len(content)

        return result

    def _path_optimize(self) -> None:
        """Optimize source code via two-stage LLM pipeline, then optionally profile."""
        hot_files = self._select_hot_files()
        if not hot_files:
            _print("  No kernel-containing files detected. "
                   "Run with --source-dir pointing at your source.", style="yellow")
            return

        _print()
        _print(f"  Hot files selected ({len(hot_files)}):", style="cyan")
        for path, _ in hot_files:
            _print(f"    · {pathlib.Path(path).name}", style="dim")
        _print()

        # Stage 1: Local LLM summarization (if configured)
        summaries: List[tuple] = []  # [(filename, summary_or_content)]
        if self._llm_local:
            _print("  Stage 1: Summarizing files with local LLM...", style="dim")
            try:
                from rocpd.ai_analysis.llm_analyzer import LLMAnalyzer
                local = LLMAnalyzer(
                    provider="local",
                    model=self._llm_local_model or "codellama:13b",
                    api_key="ignored",
                )
                for path, content in hot_files:
                    name = pathlib.Path(path).name
                    _print(f"    Summarizing {name}...", style="dim")
                    summary = local.summarize_source_file(name, content)
                    summaries.append((name, summary))
                _print("  Stage 1 complete.", style="green")
            except Exception as exc:
                _print(f"  (local LLM failed, sending files directly: {exc})", style="yellow")
                summaries = [(pathlib.Path(p).name, c) for p, c in hot_files]
        else:
            summaries = [(pathlib.Path(p).name, c) for p, c in hot_files]

        # Stage 2: Online LLM for optimization suggestions
        if not self._llm_provider:
            _print("  No online LLM configured (--llm). Showing rule-based suggestions only.",
                   style="yellow")
            return

        _print("  Stage 2: Requesting optimization suggestions from online LLM...", style="dim")
        suggestions = self._request_optimization_suggestions(summaries)
        if not suggestions:
            return

        # Present and apply file by file
        modified: List[str] = []
        for path, original_content in hot_files:
            name = pathlib.Path(path).name
            file_sugg = suggestions.get(name)
            if not file_sugg:
                continue
            modified_content = self._present_and_apply(path, original_content, file_sugg)
            if modified_content is not None:
                p = pathlib.Path(path)
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=p.parent, delete=False, suffix=".tmp"
                ) as tmp:
                    tmp.write(modified_content)
                os.replace(tmp.name, str(p))
                modified.append(name)

        if modified:
            now = datetime.now(timezone.utc).isoformat()
            self._session.history.append(HistoryEntry(
                type="code_change",
                timestamp=now,
                files_modified=modified,
                summary=f"Optimized {len(modified)} file(s) via LLM suggestions",
            ))
            _print(f"  ✓ Modified: {', '.join(modified)}", style="green")
            _print()
            try:
                ans = _input("  Run profiling commands now? [y/N]  ").strip().lower()
            except EOFError:
                ans = ""
            if ans == "y":
                self._path_profiling(_source="code_change_analysis")

    def _request_optimization_suggestions(
        self, summaries: List[tuple]
    ) -> Dict[str, str]:
        """Send summaries to online LLM; return {filename: suggestion_text}."""
        try:
            from rocpd.ai_analysis.llm_analyzer import LLMAnalyzer
            analyzer = LLMAnalyzer(
                provider=self._llm_provider,
                api_key=self._llm_api_key,
                model=self._llm_model,
            )
            combined = "\n\n".join(
                f"=== {name} ===\n{content}" for name, content in summaries
            )
            system = (
                "You are an expert AMD GPU performance engineer. "
                "Given source file summaries, provide concrete optimization suggestions "
                "per file. Format your response as:\n"
                "FILE: <filename>\n<suggestions>\n\n"
                "Be specific and actionable. Focus on memory coalescing, occupancy, "
                "MFMA usage, and unnecessary synchronization."
            )
            raw = analyzer.analyze_with_llm(
                {"source_summaries": combined},
                custom_prompt="Provide file-by-file optimization suggestions.",
            )
            result: Dict[str, str] = {}
            # Normalize: if response starts with FILE:, add a leading newline
            if raw.lstrip().startswith("FILE:"):
                raw = "\n" + raw.lstrip()
            for block in re.split(r"\nFILE:\s*", raw):
                block = block.strip()
                if not block:
                    continue
                lines = block.split("\n", 1)
                if len(lines) == 2:
                    result[lines[0].strip()] = lines[1].strip()
            return result
        except Exception as exc:
            _print(f"  (online LLM failed: {exc})", style="red")
            return {}

    def _present_and_apply(
        self, path: str, original: str, suggestion: str
    ) -> Optional[str]:
        """Show suggestion, optionally show diff, ask for confirmation. Return new content or None."""
        name = pathlib.Path(path).name
        _print()
        _print(f"  ── Suggestions for {name} ──────────────────────────────", style="cyan")
        _print(suggestion[:2000] + ("…" if len(suggestion) > 2000 else ""))
        _print()
        try:
            ans = _input(f"  Append LLM suggestions as comments to {name}? [y/N/diff]  ").strip().lower()
        except EOFError:
            return None
        if ans == "diff":
            _print("  (Diff view: LLM suggestions are advisory — showing suggestion text)",
                   style="dim")
            _print(suggestion, style="dim")
            try:
                ans = _input(f"  Append LLM suggestions as comments to {name}? [y/N]  ").strip().lower()
            except EOFError:
                return None
        if ans == "y":
            separator = "\n" + "=" * 72 + "\n"
            return (original + separator +
                    "// LLM OPTIMIZATION SUGGESTIONS:\n// " +
                    "\n// ".join(suggestion.splitlines()) + "\n")
        return None

    def _pursue_recommendation(self, item: PersistentMenuItem) -> None:
        """Show full recommendation and sub-menu: [r] run command, [m] back to main menu."""
        _print()
        _print(f"  ── {item.title} [{item.priority}] ──────────────────────────────",
               style="cyan")
        detail = item.detail
        if detail.get("issue"):
            _print(f"  Issue:  {detail['issue']}")
        if detail.get("suggestion"):
            _print(f"  Why:    {detail['suggestion']}")
        if detail.get("estimated_impact"):
            _print(f"  Impact: {detail['estimated_impact']}")

        cmds = [c.get("full_command", "") for c in detail.get("commands", [])
                if c.get("full_command")]
        if cmds:
            _print()
            _print("  Suggested commands:", style="cyan")
            for i, cmd in enumerate(cmds, 1):
                _print(f"    [{i}]  $ {cmd}", style="dim")

        _print()
        _print("  [r]  Run suggested command")
        _print("  [m]  Back to main menu")
        _print()
        try:
            choice = _input("  > ").strip().lower()
        except EOFError:
            return

        if choice == "r" and not cmds:
            _print("  No suggested commands available for this recommendation.", style="yellow")
        elif choice == "r" and cmds:
            cmd = cmds[0]
            _print(f"  Running: $ {cmd}", style="dim")
            try:
                args = shlex.split(cmd)
            except ValueError:
                args = [cmd]
            try:
                subprocess.run(args, shell=False, check=False)
            except Exception as exc:
                _print(f"  Command failed: {exc}", style="red")
            try:
                db_input = _input(
                    "  Enter path to .db file from this run (or Enter to skip): "
                ).strip()
            except EOFError:
                db_input = ""
            if db_input:
                db_path = pathlib.Path(db_input).expanduser()
                if db_path.exists():
                    new_recs = self._run_tier1_analysis(str(db_path))
                    added = self._ingest_recommendations(new_recs)
                    now = datetime.now(timezone.utc).isoformat()
                    self._session.history.append(HistoryEntry(
                        type="profiling_run", timestamp=now, db_path=str(db_path)
                    ))
                    _print(f"  ✓ {added} new recommendation(s) added.", style="green")
                else:
                    _print(f"  File not found: {db_path}", style="red")
        # [m] or any other input → return to main menu (item stays in list)
