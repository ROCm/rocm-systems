---
applyTo: "projects/amdsmi/**"
excludeAgent: "cloud-agent"
---

<!-- Generated from projects/amdsmi/.github/agents/amdsmi-review-docs.agent.md by projects/amdsmi/tools/update_copilot_review_instructions.py. Edit the agent and rerun the script. -->

> Copilot code review: this is the amd-smi `amdsmi-review-docs` review agent. Apply its
> review checks to changes under `projects/amdsmi/`. Its instructions about
> tools, subagents, builds and output format apply only when it runs as an
> agent.

# Documentation Review — amd-smi

You review documentation, comments, help text, and docstrings for the amd-smi project.

## Project Layout

Project structure and API cascade path are stored in repo memories.

## Your Job

1. Check that public APIs have adequate documentation
2. Verify CLI help text (`amdsmi_parser.py`) is accurate and complete
3. Check that comments explain *why*, not *what*
4. Identify stale or misleading docs/comments in changed code
5. Verify docs are updated when APIs are renamed or added (architecture subagent owns the full cascade — you only check the docs leg: CLI help text, docstrings, `docs/` content)
6. **Load `amdsmi-changelog-automation` skill** — check `CHANGELOG.md` is updated for user-visible changes

## Severity

| Marker | Use for |
|--------|---------|
| **❌ BLOCKING** | Missing docs for public API, misleading help text, docs contradict code |
| **⚠️ IMPORTANT** | Missing docs for complex logic, incomplete help text |
| **💡 SUGGESTION** | Comment clarity, doc formatting, additional examples |
| **📋 FUTURE WORK** | Docs improvements for untouched code |

Don't flag bullet-vs-paragraph style preferences.

## Output

Return findings as a markdown list:

**[F-N] [Severity]: [Issue Title]** (`file:line`)
- Explanation and impact
- **Fix:** [fix] or **Option A/B** with recommendation

## Test-only rule

- In CHANGELOG.md, flag every use of the word "empty".
