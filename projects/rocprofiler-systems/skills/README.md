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

## Tests

There is no CI for skills. `run_evals.py` is the gate, and it is meant to be run by
hand before a release:

```bash
./skills/run_evals.py --list                                # validate the datasets
./skills/run_evals.py                                       # does the right skill load?
./skills/run_evals.py --mode behavioral                     # also grade the agent
./skills/run_evals.py --skill instrumenting-binaries        # one skill
./skills/run_evals.py --model <model>                       # pick the model under test
./skills/run_evals.py --save-transcripts /tmp/transcripts   # keep transcripts to debug
```

It needs Python 3.9 or newer and the `claude` CLI, and it also runs from the installed
`skills/` folder because the datasets are installed with the skills. It exits 0 when
every case passes, 1 on a failure, and 77 when `claude` is not installed, so a caller
can treat that as skipped rather than failed. `--list` needs no CLI.

Routing mode denies the agent its tools and only checks which skill it chose, so it is
quick (about a minute for all cases). Behavioral mode lets the agent work and grades
`logs_contain`, `files_exist`, and the plain-language `expected_behavior` and
`unexpected_behavior` claims with a second agent, so it takes longer and uses more
quota; the agent may run the profiler tools if they are on `PATH`.

How a case is graded:

- A skill counts as loaded only when the agent calls the `Skill` tool. Reading a
  `SKILL.md` does not count: on a vague prompt the agent explores the workspace, finds
  the installed skills and reads them, which would fail a case that should not route.
- Every case gets its own scratch workspace with the skills installed by
  `install-skills.sh`, and runs with `--setting-sources project` so the agent does not
  also see your personal skills, which can answer a prompt instead of the skill under
  test. `--keep-user-settings` turns that off, and the report then warns about any
  personal skill that was visible.
- The agent is a language model, so a result can differ between runs. Re-run a failing
  case before changing a skill or its dataset, and use `--save-transcripts` to see what
  the agent actually did.

Each dataset needs at least three cases that should trigger the skill, two that should
not, and one triggering case with judged behavior. Negative cases carry only an id, a
prompt, and an optional note. The runner also rejects a case id that two skills share;
uniqueness across the whole repository is still a manual check:

```bash
grep -rhoE --include=evals.json '"id": *"[^"]+"' ../.. | sed -E 's/"id": *//' | sort | uniq -d   # must print nothing
```

The install script has no automated test of its own; `run_evals.py` exercises it for
every case, and you can also try it by hand with a throwaway `HOME`:

```bash
HOME=$(mktemp -d) ./install-skills.sh --agent claude
```

## Keeping skills correct

Nothing catches drift automatically, so it falls to each change. A PR that alters
`rocprof-sys-instrument` options or defaults, the built-in presets, the
`rocprof-sys-avail` output, or the `--help` topics updates the matching skill in the
same PR, exactly as it updates docs and tests. The skills record what was observed on
a real build, so re-run the commands a skill quotes before changing it.
