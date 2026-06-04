# dasboard

a spa written with React + TypeScript + Vite

talks to daemon via flatbuffers `../schema`



## /simulators
- list installed sims
- install

## /simulators/{sim}

- shows info about that simulator
- stats about what nodes
- links to gpus tab

## /simulators/{sim}/gpus
list what kinds of gpus are supported
if supports designing custom gpu button for doing that

## /simulators/{sim}/gpus/{gpu}
grapicaly create gpus

## /simulators/{sim}/gpus/{gpu}/profiles/{profile}

name
simulator
simulator_mode
gpu
num_gpus
num_nodes

## /simulators/{sim}/gpus/{gpu}/profiles/{profile}/sessions

a session is a running profile
lists sessions
delete session


## /simulators/{sim}/gpus/{gpu}/profiles/{profile}/sessions/{session}

- view outputs, stats, etc
- metrics
- delete
- view runs
- kill runs
- list process

## /simulators/{sim}/gpus/{gpu}/profiles/{profile}/sessions/{session}/runs/{run}

live terminals (xterm.js)

