# AI-aware code review checklist

**Structured review playbook (priorities, comment types, domains):** [`.ai/skills/code_review.md`](../skills/code_review.md).

## Correctness

- [ ] Behavior matches description; edge cases considered for CLI, parsers, and numeric paths
- [ ] No guessed APIs or paths; ROCm/profiler assumptions are credible

## Scope

- [ ] Single-purpose diff; no hidden refactors or unrelated reformatting
- [ ] No unapproved new dependencies

## Standards

- [ ] Python: ruff clean; types/conventions match `src/` patterns
- [ ] C++/CMake: matches existing `src/lib/` and CMake style; builds out-of-source

## Profiling / product

- [ ] Trace/counter/roofline/output changes covered by tests or justified
- [ ] Deterministic tests; no flaky time-based assertions
- [ ] Experimental features use `ExperimentalAction` and `--experimental` per CONTRIBUTING

## Security & risk

- [ ] No blind trust of shell/MCP: commands and tool use reviewed; no piping untrusted fetch into `sh`
- [ ] External or pasted input (issues, logs, traces, data files) validated like untrusted data where parsed
- [ ] Changes stay scoped to this project/workspace; no unnecessary broad filesystem or secret access
- [ ] High-impact paths (`.ai/skills/`, `.ai/rules/`, hooks) reviewed with care — prompt/skill injection surface

## Safety / maintenance

- [ ] CHANGELOG updated when required for release-facing changes
- [ ] Docs touched if user-facing behavior or flags change
