# RocJITsu CLI

The rocjitsu CLI

## rocjitsu daemon

### rocjitsu daemon start
start the daemon

### rocjitsu daemon kill
kill the daemon

## rocjitsu workload


### rocjitsu workload create
--name filename.json

prompts interactively to help create a workload
has a bunch of arguments to prefill things you could want
combos the profile and exec creation prompts

### rocjitsu workload run

```
rocjitsu workload [--attach] [--keep] run [workload.json]
```

runs a workload
if --keep, don't delete on completeion
if --attach, it forwards the stdout of all the nodes to the stdout here, and the exit code is same as the head's exit code.
if not --attach, it
returns a session id of the running workload

## rocjitsu profile
### rocjitsu profile list
list all profiles

### rocjitsu profile rm
delete a profile

### rocjitsu profile create
create a profile
has lots of arguments for everything you could want to set for a profile
prompts for any questions that were not specified

### rocjitsu profile import
import a profile from a json file

### rocjitsu profile export
export a profile to a json file

## rocjitsu emulator

### rocjitsu emulator list
list the installed emulators

### rocjitsu emulator status
get the status of an emulator

## rocjitsu state

### rocjitsu state builtins
(re)write the builtin topologies to `$ROCJITSU_CLI_CONFIG_DIR/topology/`,
overwriting any existing files. RocJITsu writes any *missing*
builtins automatically on every run; use this command after
upgrading rocjitsu to refresh the bundled set.

### rocjitsu state purge
completely stop and purge all rocjitsu processes and state.
Removes the runtime and state directories; pass `--all`
to also delete the config directory (profiles + topologies).
Refuses while any `rocjitsu run` is live.

## rocjitsu cleanup

reclaim what a run that died abruptly could not: its containers
and network, its stranded workload processes, and its session
scratch directory. Skips sessions whose run still answers, so it
is safe to run at any time; `--dry-run` previews it.

## rocjitsu session

Implemented: `start`, `list`, `stop`.

### rocjitsu session start
takes every override `rocjitsu run` does (`--profile`, `--num-nodes`, …)

brings a session up with no workload, prints its id, and holds it
open so other terminals can `rocjitsu exec --session <id>` into it.
The session lives exactly as long as this process: background it,
or leave it in a terminal of its own. Ends on Ctrl-C, or when
`rocjitsu session stop` asks it to.

### rocjitsu session list
list the sessions whose owning process still answers

### rocjitsu session stop
--session (optional; the id, when more than one is up)

ask a session to tear itself down. Returns once it has accepted,
not once teardown has finished.

### rocjitsu session shell
--session
--node

start an interactive shell

### rocjitsu session exec
--session
--node
--env (set env var)
--workdir (set workingdir)
program to run

### rocjitsu session attach
--session str
--node 0
--exec u32

attach to relevent places


## rocjitsu run

```
rocjitsu run [--profile profile_name] [--image ] [program]
```
