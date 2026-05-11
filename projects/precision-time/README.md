# Precision Time

Precision Time is an open source library that support precision synchronization
on AMD platforms. It works in conjunction with [rocr-runtime](../rocr-runtime)
to support translation of time from HSA agents to a common global system
timeline.

A goal is to support alignment of timestamps across HSA agent with precision
targets of 1 µs or less, in systems where appropriate HW mechanisms exist to
allow this (e.g., PTP/PTM)

### Background

Intro to PTP

## Overview

What we are doing

## Implementation

In it's current form, this just just provides a simple time translation interface and uses a
time-series database to store timestamps

