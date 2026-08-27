# Timing

`mirage run --timing` asks the emulator to model how long the emulated
device would have taken, and to make the device's clock say so, rather
than computing the right answer as fast as the host can and having no
opinion about the time.

```sh
mirage run --profile mi350x --timing -- ./my_benchmark
mirage run --profile mi350x --timing-tuning ~/private/mi355x.json -- ./my_benchmark
mirage run --profile mi350x --no-timing -- ./my_benchmark
```

## The numbers are baked in, not referenced

The emulator is given exactly one architecture config, and the `timing`
block inside it is the whole description of how fast the part is. There
is no second file, no search path, no model to name and no `.so` to
load. Mirage's job is to put the numbers *into* that one file.

That is the point of baking rather than referencing. A run's timing is
reproducible from the single artefact it was handed, and that artefact
can be given to somebody else without them needing anything more — no
tuning file to ship alongside it, no version of mirage that happens to
carry the same table, no environment variable set the same way.

The config lands in the session's scratch directory under
`$XDG_RUNTIME_DIR` (`mirage paths` shows where), which is outside this
repository and is removed with the session. To keep a copy, take it from
there while the run is up.

The block looks like this:

```json
"timing": {
  "enabled": true,
  "clock_mhz": 2700,
  "machine": {
    "compute_units": 256,
    "xcds": 8,
    "dram.latency_cycles": 1026,
    "dram.bytes_per_cycle": 2962.963,
    "matrix_multiply.macs_per_cycle.f16": 2048,
    "…": 0
  }
}
```

`machine` is a **flat** map of dotted keys to numbers. Flat, not nested,
because the key space is not a tree: the timing plane reads both
`matrix_multiply.macs_per_cycle` and
`matrix_multiply.macs_per_cycle.f16`, and no nested object can hold a
number and an object under one name.

## The three layers

Mirage assembles the block from three layers, applied in this order and
reported by `mirage run` before it starts anything:

1. **The agent's built-in table.** It lives beside the geometry mirage
   already knows for that device — see `builtin/src/timing.rs` — because
   it describes the same device. A custom agent carries its own in the
   `timing` object of its agent JSON.
2. **The profile's stored overrides**, as `timing.machine.<key>` emulator
   options on the profile, alongside a `timing` switch:

   ```json
   "emulator": {
     "options": {
       "timing": true,
       "timing.machine.dram.latency_cycles": "900",
       "timing.machine.fabric.bytes_per_cycle": "7000.0"
     }
   }
   ```

   Written by hand into `<config>/mirage/profile/<name>.json`. Values may
   be quoted or bare; they are numbers either way, and quoted is what
   mirage writes because a `SimpleValue` number is an integer and half
   these keys are rates. `-o timing=true` will not do it: `rocjitsu`
   declares no option schema and refuses every `-o` key, which is what
   keeps a typo'd option from being silently ignored.
3. **`--timing-tuning PATH`**, a JSON file merged over both.

```text
mirage: timing on at 2700 MHz: 85 numbers baked in from agent `mi350x`
mirage: timing: 2 replaced by the profile (lds.banks, xcds)
mirage: timing: 3 replaced by /home/you/private/mi355x.json (dram.bytes_per_cycle, dram.latency_cycles, fabric.bytes_per_cycle)
```

`--timing-tuning` on its own turns timing on: it is only ever typed by
somebody who wants those numbers used, and reading a tuning file into a
functional run would be a silent no-op.

Only a backend that emulates the device in software has device time to
model. `rocjitsu` does; a backend that runs the workload on real
hardware, or that only translates it, already has somebody else's
timing, and asking one of those for `--timing` is refused by name rather
than reported as a timing model that will not run. `mirage emulators -l`
lists what this build has.

### A key nobody has heard of is an error

A layer may only **replace** a key the layer below already has. A tuning
file that names a key the agent's table does not have is refused, and the
error prints the keys sharing its first path segment:

```text
/home/you/t.json names a key the agent's timing table does not have:
dram.latency_cyles. A tuning file may only change a number the device
already has, never add one — a mistyped key that silently became a new
parameter is how a config ends up describing a machine nobody built.
  under `dram` it has: dram.bytes_per_cycle, dram.latency_cycles
```

The same check runs again in the backend, on the way into the config,
because a profile is a document a person can edit by hand and that is the
last place a mistyped key can be told from a new parameter. After that it
is a number in a config file, indistinguishable from one the device
really has.

## Writing a tuning file

An object of numbers. Keys may be spelled dotted or nested, and mixed:

```json
{
  "dram.latency_cycles": 900,
  "dram": { "bytes_per_cycle": 3100.0 },
  "l2": { "hit_cycles": 300, "lines_per_cycle": 32.0 }
}
```

A whole `machine` block from a baked config is also accepted as-is, so a
config can be pulled out of a session's scratch directory, edited, and
fed straight back in with `--timing-tuning`.

Only numbers. A quoted number is refused rather than parsed: in a tuning
file it is a typo often enough that accepting it costs more than it
saves. Integers stay integers and rates stay rates through the whole
merge, because a `sets` of `2048.0` is a different JSON type and the
emulator's parser may refuse it.

`mirage agent show <name>` prints the whole vocabulary a device accepts,
which is the same list as the keys in its `timing` object.

## Where a private table goes, and where it must not

**A measured tuning table must not be committed to this repository.**

The built-in tables in `builtin/src/timing.rs` contain only values whose
provenance is stated at the point of definition, and every one of them is
public-derived or assumed. **None of them is measured.** Three kinds
appear, each marked where it is defined:

* *Derived* — computed from something the agent already carries (its
  clock, memory width, preset per-CU limits, component tree) or from a
  published headline figure, by an arithmetic written out in the code.
* *Published* — a figure AMD has stated for the part.
* *Assumed* — a plausible value for a parameter AMD has not published in
  this detail.

Bandwidths in the built-in tables are **datasheet peaks**, not achievable
rates. They are a ceiling; the shortfall a real kernel sees is supposed
to come out of the model's queueing, not out of the table. A table that
carries an achievable rate is a different kind of table and has to say
so.

A private table lives as a JSON tuning file **outside the repository** —
somewhere like `~/.config/mirage-private/` or a directory your team
controls — and reaches a run through `--timing-tuning`. Mirage reads it,
merges it, and does nothing else with it: it is never copied into
mirage's configuration directory, never written back to a profile on
disk, and never placed in this tree. The only file mirage writes it into
is the session's own config under `$XDG_RUNTIME_DIR`, which is outside
the repository and is removed when the session ends.

If a run's results need to be called *measured*, they need a table that
was measured, and the run's own start-up report is what records which
file it came from.

## Turning it off

`--no-timing` overrides a profile that enables timing, and clears both
the switch and the stored overrides for that run — otherwise a later
`--timing` would silently inherit overrides for a table nobody looked at
in between.

A drop-in `--config PATH` is the whole machine description, timing block
included, so mirage adds nothing to it. The `--timing` flags are refused
alongside `--config` rather than ignored.

## Agents from before mirage had a table

An agent JSON with no `timing` object still loads and still runs
functionally. Asking to time it is refused by name rather than answered
with fallbacks — a table of missing numbers resolves to the slowest
reasonable value for each, which reads slow and looks exactly like a
measurement of a slow machine.

To get the table for a builtin agent, run `mirage agent delete <name>`:
mirage puts the shipped agent — timing table and all — straight back in
its place in the same breath.

`mirage state builtins` will not do it on its own, and that is
deliberate. An agent from before mirage shipped timing tables differs
from the one mirage now ships, which is indistinguishable from an agent
you edited yourself, and `state builtins` leaves those alone rather than
discarding the only copy of somebody's work. It names them, so you can
see which.
