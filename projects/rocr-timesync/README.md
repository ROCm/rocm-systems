# ROCR Timesync

ROCR Timesync is an open source library that support precision time
synchronization on AMD platforms. It works in conjunction with
[rocr-runtime](../rocr-runtime) to support translation of time from HSA agents
to a common global system timeline.

A goal is to support alignment of timestamps across HSA agents with precision
targets of 1 µs or less, in systems where appropriate HW mechanisms exist to
allow this (e.g., PTP/PTM)

### Background

Intro to PTP

## Overview

What we are doing

## Architecture

ROCR Runtime -> ROCR Timesync API -> ROCR Timesync library -> ROCR Timesync backend (currently based on influxdb)

ROCR Timesync is built as a library. It is linked into rocr-runtime. Timesync can operate in 2 modes:
1) in-process
2) out-of-process


#### In-process

With in-process mode, ROCR Timesync creates a datastore for timestamp deltas and directly implements the Timesync API service by querying this datastore. On process teardown, the datastore is deleted.

With out-of-process mode, ROCR Timesync establishes connectivity to a remote datastore and implements the Timesync API by calling out to that datastore. On process teardown, the datastore is _not_ deleted.
In-proce

## Implementation

In its current form, this just just provides a simple time translation interface and uses a
time-series database to store timestamps

