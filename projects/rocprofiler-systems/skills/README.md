# rocprofiler-systems agent skills

User-facing Agent Skills that teach an AI agent to drive the ROCm Systems Profiler
(`rocprof-sys-run`, `rocprof-sys-sample`, `rocprof-sys-avail`, `rocprof-sys-instrument`).
Each skill covers one task end to end, so an agent loads only what the question needs.

| Skill | Use it for |
| --- | --- |
| [choosing-a-profiling-preset](choosing-a-profiling-preset/SKILL.md) | Start here. Pick the `--preset` that fits a workload (general, GPU, HPC and MPI, OpenMP, API tracing, production) and get the ready-to-run command |
| [discovering-profiling-options](discovering-profiling-options/SKILL.md) | Find out what is available and how to use it: settings, hardware counters, ROCm tracing domains, components, GPU metrics, `--help` topics, and saving a configuration file |
| [instrumenting-binaries](instrumenting-binaries/SKILL.md) | Runtime instrumentation vs binary rewrite, including and excluding functions and modules, dry runs, granularity tuning, why a function is missing, and rewriting libraries |

`discovering-profiling-options` and `instrumenting-binaries` name the neighbouring
skills in their descriptions, so a question about presets goes to the preset skill.
Attaching to an already running process is not covered here; use `rocprof-sys-attach`.

## Install

An installed ROCm Systems Profiler ships these skills under
`<install-prefix>/share/rocprofiler-systems/skills/`. Agents do not look there on
their own, so run the install script from that folder for your agent:

```bash
<install-prefix>/share/rocprofiler-systems/skills/install-skills.sh --agent claude
```

| `--agent` | Installs into |
| --- | --- |
| `claude` | `~/.claude/skills/` |
| `codex` | `~/.agents/skills/` |
| `cursor` | `~/.cursor/skills/` |

From a source checkout, run `skills/install-skills.sh` the same way.

The script installs every folder next to it that contains a `SKILL.md`, replaces a
skill of the same name that is already in the target folder, and leaves other skills
alone. Run it again after upgrading to refresh the copies. Cursor also reads the
Claude and Codex folders, so install for one agent only if you use Cursor alongside
them.

## Uninstall

```bash
<install-prefix>/share/rocprofiler-systems/skills/install-skills.sh --agent claude --uninstall
```

This removes only the skills listed above from that folder.

## Structure

Every skill directory holds:

```text
<skill-name>/
├── SKILL.md            # frontmatter (name, description) plus the instructions
├── skill-card.md       # human-facing card: description, owner, license
└── evals/evals.json    # prompts that must and must not route to this skill
```

`SKILL.md` states which command to run, when, and where the boundaries are. The skills
are self-contained: they describe behaviour that `--help` does not state and point to
`<tool> --help` for the full, current flag list instead of copying it, so they do not
go stale when options are added. The `evals/` directories are for local testing only:
neither the CMake install nor the install script copies them.

## Tests

The skills are tested with [skillscope](https://github.com/amd/skillscope), the same
harness the `amd/skills` repository runs on every pull request. See its
[README](https://github.com/amd/skillscope#readme) for the full reference. Skillscope
reads one dataset per skill, `<skill>/evals/evals.json`, and runs three kinds of test:

- **Structural**: the skill is well formed (expected files, hierarchy, references).
- **Routing**: the skill loads when it should and stays quiet when it should not.
- **Behavioral**: the agent runs end to end and an LLM judge checks the answer against
  the case's `expected_behavior` and `unexpected_behavior`.

### Quick start

It needs Python 3.10 or newer and, for routing and behavioral tests, an authenticated
`claude` CLI. Run it from this `skills/` folder:

```bash
uv tool install --system-certs git+https://github.com/amd/skillscope

skillscope structural                        # no agent, no tokens
skillscope structural --external             # the same, plus checking external URLs
skillscope behavioral --skill instrumenting-binaries   # needs an authenticated `claude` CLI
skillscope routing --routing-room instrumenting-binaries
skillscope routing --routing-room choosing-a-profiling-preset
```

CI uses a pinned release, so install that one to reproduce a CI result: append
`@v0.1.3` to the URL above, or whatever tag the `amd/skills` workflow names.

| Command | Does |
| --- | --- |
| `skillscope structural` | Skill folders, datasets, and markdown references. |
| `skillscope routing` | Which skill fires, with several installed together. |
| `skillscope behavioral` | What a skill does once it has fired. |

`--help` on any command is the reference. Reports go to stdout as markdown and to JSON
under `.skillscope/runs/` in the repository, which is worth adding to `.gitignore`.
How to write a dataset is in skillscope's
[docs/authoring-evals.md](https://github.com/amd/skillscope/blob/main/docs/authoring-evals.md)
and every flag is in its
[docs/usage.md](https://github.com/amd/skillscope/blob/main/docs/usage.md).

### Writing cases that pass

The behavioral judge sees the files in the workspace, the tools the agent used, the
**first 1500 characters** of the agent's final message, and the first 4000 characters
of tool input and output. It does not see the skill text. A point the agent makes later
in a long answer is invisible, so:

- Write an `expected_behavior` about something the prompt makes the agent say early.
- Keep `SKILL.md` guidance short and command-first, so answers stay inside the limit.
- If a claim fails but the answer looks right, check whether the point came after the
  first 1500 characters.

The agent is a language model, so a result can differ between runs. Re-run a failing
case before changing a skill or its dataset. Keep case ids unique across the whole
repository:

```bash
grep -rhoE --include=evals.json '"id": *"[^"]+"' ../.. | sed -E 's/"id": *//' | sort | uniq -d   # must print nothing
```

## Keeping skills correct

Nothing catches drift automatically, so it falls to each change. A PR that alters
`rocprof-sys-instrument` options or defaults, the built-in presets, the
`rocprof-sys-avail` output, or the `--help` topics updates the matching skill in the
same PR, exactly as it updates docs and tests. The skills record what was observed on
a real build, so re-run the commands a skill quotes before changing it.
