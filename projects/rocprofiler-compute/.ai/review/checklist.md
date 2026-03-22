# AI-aware code review checklist

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

## Safety / maintenance

- [ ] CHANGELOG updated when required for release-facing changes
- [ ] Docs touched if user-facing behavior or flags change
