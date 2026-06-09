# Mirage CLI

The mirage CLI

## mirage daemon

### mirage daemon start
start the daemon

### mirage daemon kill
kill the daemon

## mirage workload


### mirage workload create
--name filename.json

prompts interactively to help create a workload
has a bunch of arguments to prefill things you could want
combos the profile and exec creation prompts

### mirage workload run

```
mirage workload [--attach] [--keep] run [workload.json]
```

runs a workload
if --keep, don't delete on completeion
if --attach, it forwards the stdout of all the nodes to the stdout here, and the exit code is same as the head's exit code.
if not --attach, it
returns a session id of the running workload

## mirage profile
### mirage profile list
list all profiles

### mirage profile rm
delete a profile

### mirage profile create
create a profile
has lots of arguments for everything you could want to set for a profile
prompts for any questions that were not specified

### mirage profile import
import a profile from a json file

### mirage profile export
export a profile to a json file

## mirage emulator

### mirage emulator list
list the installed emulators

### mirage emulator status
get the status of an emulator

## mirage state

### mirage state builtins
(re)write the builtin topologies to `$MIRAGE_CONFIG/topology/`,
overwriting any existing files. Mirage writes any *missing*
builtins automatically on every run; use this command after
upgrading mirage to refresh the bundled set.

### mirage state purge
completely stop and purge all mirage processes and state.
Removes the runtime, state, and cache directories; pass `--all`
to also delete the config directory (profiles + topologies).

## mirage session


### mirage session create
--profile


### mirage session boot
--profile 
--name

boots a session
if profile is specified creates it as well
returns session id


### mirage session activate
--session
set a sesssion as the target for all commands

### mirage session delete
kill a session

### mirage session shell
--session
--node

start an interactive shell 

### mirage session exec
--session
--node
--env (set env var)
--workdir (set workingdir)
program to run

### mirage session attach
--session str
--node 0
--exec u32

attach to relevent places


## mirage run

```
mirage run [--profile profile_name] [--image ] [program]
```

