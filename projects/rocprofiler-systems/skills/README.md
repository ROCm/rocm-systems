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
are self-contained: they describe behavior that `--help` does not state and point to
`<tool> --help` for the full, current flag list instead of copying it, so they do not
go stale when options are added. The install script leaves `evals/` out of the copy
it makes for the agent.

## Keeping skills correct

Nothing catches drift automatically, so it falls to each change. A PR that alters
`rocprof-sys-instrument` options or defaults, the built-in presets, the
`rocprof-sys-avail` output, or the `--help` topics updates the matching skill in the
same PR, exactly as it updates docs and tests. The skills record what was observed on
a real build, so re-run the commands a skill quotes before changing it.
