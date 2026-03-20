# AI Agent Guidelines Plan for rocprofiler-compute

## Goal
Create AI coding guidelines and code review skills for Claude, Copilot, and Cursor while avoiding redundancy with existing docs.

**Principle:** Don't duplicate. Reference CONTRIBUTING.md, pyproject.toml, PR templates.

---

## Files to Create

```
projects/rocprofiler-compute/
  AGENTS.md                          # Main guidelines
  CLAUDE.md                          # Claude adapter

  ai/skills/
    code-review.md                   # Shared review skill

  .claude/skills/code-reviewer/
    SKILL.md                         # Claude skill wrapper (enables /code-reviewer)

  .cursor/rules/
    global-guidelines.mdc            # Cursor adapter

  .github/
    copilot-instructions.md          # Copilot adapter
```

---

## Why `.claude/skills/` AND `ai/skills/`?

**Two-tier approach:**

1. **`ai/skills/code-review.md`** - Shared documentation
   - Used by all AI agents (Claude, Copilot, Cursor)
   - Pure markdown, no YAML

2. **`.claude/skills/code-reviewer/SKILL.md`** - Claude-specific wrapper
   - Has YAML frontmatter (required by Claude)
   - Enables `/code-reviewer` slash command
   - Just references `ai/skills/code-review.md`

**Why both?**
- Claude requires `.claude/skills/` structure for slash commands
- Other tools (Copilot, Cursor) read plain markdown from `ai/skills/`
- Content lives in ONE place (`ai/skills/`), Claude wrapper just points to it

---

## File Content (Brief Examples)

### 1. AGENTS.md (~150 lines)

**Sections:**
- **Tech Stack** - Python 3.9+, Ruff, CMake, pytest
- **Architecture** - Profiler backends, analysis engines, GPU support (GFX908-950)
- **Code Style** - ⚠️ Reference `pyproject.toml`, NOT duplicate
- **Module Rules** - Profile code can't import analysis modules
- **Testing** - Mirror `src/` in `tests/`, use pytest markers
- **GPU Archs** - MI100-MI350 support, config files
- **Experimental** - ExperimentalAction workflow (see CONTRIBUTING.md)

**Example snippet:**
```markdown
## Code Style
**Reference `pyproject.toml` [tool.ruff.lint]:**
- Type annotations required (ANN)
- F-strings only (UP)
- pathlib only (PTH)
- 88 char line limit

**Pre-commit:** See CONTRIBUTING.md
```

---

### 2. ai/skills/code-review.md (~100 lines)

**Sections:**
- **Review Priority Order:**
  1. Correctness & GPU-specific bugs
  2. Security
  3. Performance (profile mode overhead)
  4. Style (Ruff rules)
  5. Architecture (module boundaries)
  6. Tests & docs

- **Comment Types:** Issue / Suggestion / Question / Nit

- **Domain Checks:**
  - Profiler: Counter formulas, API usage
  - Analysis: Metric calculations, pandas ops
  - GPU: Architecture compatibility (GFX908-950)
  - TUI/WebUI: Responsive design, data refresh

**Example snippet:**
```markdown
## Review Priorities
1. **Correctness** - Bugs, GPU counter formulas, arch compatibility
2. **Security** - No hardcoded credentials, validate shell commands
3. **Performance** - Profile mode: minimize imports, lazy loading
4. **Style** - Check against pyproject.toml (ANN, UP, PTH)
5. **Architecture** - Profile can't import analysis modules
6. **Tests** - New features need happy-path + failure tests
```

---

### 3. .claude/skills/code-reviewer/SKILL.md (~30 lines)

**Purpose:** Claude wrapper to enable `/code-reviewer` command

```markdown
***
name: code-reviewer
description: Review diffs for correctness, security, GPU-specific issues
context: repo
***

# Code Review Skill

When invoked, read and follow:
1. `AGENTS.md` - Project architecture and patterns
2. `ai/skills/code-review.md` - Review priorities and process

Focus on:
- GPU profiling correctness (counter formulas, arch compatibility)
- Module boundaries (profile vs analysis)
- Performance (minimize profile mode overhead)

Output structured review: Summary → Issues → Suggestions → Questions
```

---

### 4. CLAUDE.md (~50 lines)

```markdown
# Claude Code Guidelines

**Read:** `AGENTS.md` for architecture, GPU support, module rules

**Code Style:** Enforced by Ruff
- Config: `pyproject.toml`
- Pre-commit: CONTRIBUTING.md

**Review:** Follow `ai/skills/code-review.md`

**Key Context:**
- GPU archs: GFX908 (MI100) → GFX950 (MI350)
- Profiler backends: rocprof-v3, rocprofiler-sdk
- Analysis modes: CLI, WebUI, Database, TUI
- Profile code must NOT import analysis modules

**PRs:** Use `.github/pull_request_template.md`
```

---

### 5. .cursor/rules/global-guidelines.mdc (~30 lines)

```markdown
# title: rocprofiler-compute guidelines
# appliesTo: "**/*"
# mode: auto

Follow `AGENTS.md` for:
- Architecture (profiler/analyzer separation)
- GPU support (GFX908-GFX950)
- Module rules, testing

Code style via Ruff (see pyproject.toml):
- Type annotations required
- F-strings only
- pathlib only

Review: Use `ai/skills/code-review.md` priorities
```

---

### 6. .github/copilot-instructions.md (~40 lines)

```markdown
# Copilot Instructions

**Read:** `AGENTS.md` for architecture, GPU patterns

**Style:** Automated by Ruff (see pyproject.toml, CONTRIBUTING.md)
- Type annotations required
- F-strings only
- pathlib only

**Review:** Follow `ai/skills/code-review.md`
1. Correctness & GPU issues
2. Security & performance
3. Architecture adherence

**PRs:** Use `.github/pull_request_template.md`

**Don't:**
- ❌ Duplicate Ruff config
- ❌ Import analysis in profile code
```

---

## Debate: How Claude Skills Work

**Your Confusion:** "No skill-wise folder?"

**My Answer:** There IS a skill folder (`.claude/skills/code-reviewer/`)

**How it works:**
1. User types `/code-reviewer` in Claude
2. Claude loads `.claude/skills/code-reviewer/SKILL.md`
3. SKILL.md says "read `ai/skills/code-review.md`"
4. Claude follows that shared documentation

**Why two locations?**
- `.claude/skills/` = Claude-specific (YAML, slash commands)
- `ai/skills/` = Shared by all agents (pure markdown)

**Alternative:** Put everything in `.claude/skills/` only
- ❌ Con: Copilot/Cursor can't use it easily
- ✅ Pro: Simpler if you only care about Claude

**Your call:** Keep both (max compatibility) or Claude-only (simpler)?
